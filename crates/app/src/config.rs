use serde::{Deserialize, Serialize};

use crate::error::AppResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub num_labels: usize,
    #[serde(default)]
    pub pad_token_id: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelArch {
    Bert,
    XlmRoberta,
}

impl ModelConfig {
    pub fn parse(bytes: &[u8]) -> AppResult<Self> {
        serde_json::from_slice(bytes).map_err(Into::into)
    }

    pub fn arch(&self) -> ModelArch {
        match self.model_type.as_str() {
            "xlm-roberta" | "roberta" => ModelArch::XlmRoberta,
            _ => ModelArch::Bert,
        }
    }
}
