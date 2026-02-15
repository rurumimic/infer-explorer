use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRequest {
    pub schema_version: u32,
    pub run_id: String,
    pub task: Task,
    pub model: ModelInfo,
    pub tokenization: Tokenization,
    pub options: RunOptions,
    pub inputs: RunInputs,
    pub encoded: EncodedBatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Task {
    Embedding,
    Classifier,
    Reranker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub arch: String,
    pub root_prefix: String,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenization {
    pub max_length: usize,
    pub padding: String,
    pub truncation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunOptions {
    pub embedding_pooling: String,
    pub embedding_normalize: String,
    pub top_k: usize,
    pub classifier_prob: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunInputs {
    pub texts: Vec<String>,
    pub query: String,
    pub docs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedBatch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
    pub token_type_ids: Option<Vec<Vec<u32>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub schema_version: u32,
    pub run_id: String,
    pub task: Task,
    pub env: EnvInfo,
    pub timing_ms: Timing,
    pub outputs: RunOutputs,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvInfo {
    pub runtime: String,
    pub engine: String,
    pub os: Option<String>,
    pub gpu: Option<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Timing {
    pub tokenize: f32,
    pub inference: f32,
    pub postprocess: f32,
    pub total: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunOutputs {
    pub embeddings: Option<Vec<Vec<f32>>>,
    pub logits: Option<Vec<Vec<f32>>>,
    pub probs: Option<Vec<Vec<f32>>>,
    pub scores: Option<Vec<f32>>,
    pub order: Option<Vec<usize>>,
}
