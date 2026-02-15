use std::collections::HashMap;

use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

use crate::{
    config::{ModelArch, ModelConfig},
    error::AppResult,
};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LoadDiagnostics {
    pub applied_keys: usize,
    pub missing_keys: Vec<String>,
    pub extra_keys: Vec<String>,
    pub root_prefix: String,
}

#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub config: ModelConfig,
    pub arch: ModelArch,
    pub diagnostics: LoadDiagnostics,
    pub tensors: HashMap<String, Vec<f32>>,
}

impl LoadedModel {
    pub fn from_bytes(config_bytes: &[u8], weight_bytes: &[u8], prefix_override: Option<&str>) -> AppResult<Self> {
        let config = ModelConfig::parse(config_bytes)?;
        let arch = config.arch();
        let safetensors = SafeTensors::deserialize(weight_bytes)?;
        let names = safetensors.names();
        let name_refs = names.iter().map(|v| v.as_str()).collect::<Vec<_>>();
        let root_prefix = detect_prefix(&name_refs, arch, prefix_override);

        let mut tensors = HashMap::new();
        for key in names {
            let tensor = safetensors.tensor(key)?;
            if tensor.dtype() != safetensors::Dtype::F32 {
                continue;
            }
            let bytes = tensor.data();
            let mut values = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            tensors.insert(key.to_string(), values);
        }

        let required = required_keys(arch, &root_prefix);
        let missing_keys = required
            .iter()
            .filter(|k| !tensors.contains_key(*k))
            .map(|k| (*k).to_string())
            .collect::<Vec<_>>();

        let diagnostics = LoadDiagnostics {
            applied_keys: tensors.len(),
            missing_keys,
            extra_keys: Vec::new(),
            root_prefix,
        };

        Ok(Self {
            config,
            arch,
            diagnostics,
            tensors,
        })
    }
}

fn detect_prefix(names: &[&str], arch: ModelArch, override_value: Option<&str>) -> String {
    if let Some(value) = override_value {
        return value.to_string();
    }
    if names.iter().any(|k| k.starts_with("bert.")) {
        return "bert.".to_string();
    }
    if names.iter().any(|k| k.starts_with("roberta.")) {
        return "roberta.".to_string();
    }
    match arch {
        ModelArch::Bert => "bert.".to_string(),
        ModelArch::XlmRoberta => "roberta.".to_string(),
    }
}

fn required_keys(arch: ModelArch, prefix: &str) -> Vec<String> {
    match arch {
        ModelArch::Bert => vec![
            format!("{prefix}embeddings.word_embeddings.weight"),
            format!("{prefix}classifier.weight"),
            format!("{prefix}classifier.bias"),
        ],
        ModelArch::XlmRoberta => vec![
            format!("{prefix}embeddings.word_embeddings.weight"),
            format!("{prefix}classifier.out_proj.weight"),
            format!("{prefix}classifier.out_proj.bias"),
        ],
    }
}
