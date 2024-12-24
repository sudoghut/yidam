use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use reqwest::Client;
use serde_json::{json, Value};
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug)]
struct LlmError(String);

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LLM error: {}", self.0)
    }
}

impl Error for LlmError {}

#[derive(Debug, Clone)]
struct UserMessage {
    role: String,
    content: String,
}

impl UserMessage {
    fn to_json(&self) -> Value {
        json!({
            "role": self.role,
            "content": self.content
        })
    }
}

struct AppState {
    context: Mutex<Vec<UserMessage>>,
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    while let Some(Ok(msg)) = socket.recv().await {
        if let Message::Text(text) = msg {
            println!("Received: {}", text);
            let mut context = state.context.lock().await;
            context.push(UserMessage {
                role: "user".to_string(),
                content: text.clone(),
            });
            
            let full_context = json!({
                "messages": context.iter().map(|m| m.to_json()).collect::<Vec<Value>>()
            }).to_string();

            // Send start token
            if let Err(e) = socket.send(Message::Text("[START]".to_string())).await {
                eprintln!("Error sending start token: {}", e);
                break;
            }

            match llm_caller(&full_context).await {
                Ok(response) => {
                    println!("Response: {}", response);
                    context.push(UserMessage {
                        role: "assistant".to_string(),
                        content: response.clone(),
                    });
                    if context.len() > 10 {
                        context.remove(0);
                        context.remove(0);
                    }
                    
                    // Send the response in chunks
                    for chunk in response.chars().collect::<Vec<char>>().chunks(10) {
                        let chunk_str: String = chunk.iter().collect();
                        if let Err(e) = socket.send(Message::Text(chunk_str)).await {
                            eprintln!("Error sending message chunk: {}", e);
                            break;
                        }
                    }
                }
                Err(e) => {
                    let error_msg = format!("Error: {}", e);
                    if let Err(e) = socket.send(Message::Text(error_msg)).await {
                        eprintln!("Error sending error message: {}", e);
                        break;
                    }
                }
            }

            // Send end token
            if let Err(e) = socket.send(Message::Text("[END]".to_string())).await {
                eprintln!("Error sending end token: {}", e);
                break;
            }
        }
    }
}

async fn llm_caller(context: &str) -> Result<String, Arc<dyn Error + Send + Sync>> {
    let client = Client::new();
    let url = "http://localhost:11434/api/chat";

    let context_json: Value = serde_json::from_str(context)
        .map_err(|e| Arc::new(LlmError(e.to_string())) as Arc<dyn Error + Send + Sync>)?;
    println!("Context: {}", context);
    let body = json!({
        "model": "qwen2.5:14b",
        "messages": context_json["messages"]
    });

    let mut response = client
        .post(url)
        .json(&body)
        .send()
        .await
        .map_err(|e| Arc::new(LlmError(e.to_string())) as Arc<dyn Error + Send + Sync>)?;

    if response.status().is_success() {
        let mut full_response = String::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| Arc::new(LlmError(e.to_string())) as Arc<dyn Error + Send + Sync>)?
        {
            let chunk_str = String::from_utf8_lossy(&chunk);
            for line in chunk_str.lines() {
                if let Ok(json) = serde_json::from_str::<Value>(line) {
                    if let Some(response_part) = json["message"]["content"].as_str() {
                        full_response.push_str(response_part);
                        // return full_response;
                        println!("Response part: {}", full_response.clone());
                        return Ok(full_response);
                    } else {
                        eprintln!("Error parsing JSON: {}", line);
                    }
                } else {
                    eprintln!("Error parsing JSON: {}", line);
                }
            }
        }
        Ok(full_response + "[END]")
    } else {
        Err(Arc::new(LlmError("Error in calling LLM".to_string()))
            as Arc<dyn Error + Send + Sync>)
    }
}

async fn index_handler() -> impl IntoResponse {
    Html(include_str!("index.html"))
}

#[tokio::main]
async fn main() {
    let state = Arc::new(AppState {
        context: Mutex::new(Vec::new()),
    });

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/ws", get(websocket_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Listening on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}