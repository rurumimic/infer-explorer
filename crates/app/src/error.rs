use thiserror::Error;

pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("json parse failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("safetensors parse failed: {0}")]
    SafeTensor(#[from] safetensors::SafeTensorError),
    #[error("model is not loaded")]
    ModelNotLoaded,
    #[error("invalid input: {0}")]
    InvalidInput(String),
}
