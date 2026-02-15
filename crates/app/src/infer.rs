use std::cmp::Ordering;

use crate::{
    config::ModelArch,
    error::{AppError, AppResult},
    model::LoadedModel,
    schema::{EncodedBatch, EnvInfo, RunOutputs, RunRequest, RunResult, Task, Timing},
};

#[derive(Debug, Default)]
pub struct InferenceEngine {
    model: Option<LoadedModel>,
}

impl InferenceEngine {
    pub fn new() -> Self {
        Self { model: None }
    }

    pub fn load(&mut self, model: LoadedModel) {
        self.model = Some(model);
    }

    pub fn run_from_request(&self, request: RunRequest) -> AppResult<RunResult> {
        let model = self.model.as_ref().ok_or(AppError::ModelNotLoaded)?;
        let outputs = match request.task {
            Task::Embedding => run_embedding(model, &request.encoded, &request.options.embedding_pooling, &request.options.embedding_normalize),
            Task::Classifier => run_classifier(model, &request.encoded, &request.options.classifier_prob),
            Task::Reranker => run_reranker(model, &request.encoded, request.options.top_k),
        }?;
        Ok(RunResult {
            schema_version: 1,
            run_id: request.run_id,
            task: request.task,
            env: EnvInfo {
                runtime: "browser-wasm-webgpu".to_string(),
                engine: "burn".to_string(),
                os: None,
                gpu: None,
                notes: Some("MVP deterministic inference path".to_string()),
            },
            timing_ms: Timing::default(),
            outputs,
        })
    }
}

fn run_embedding(model: &LoadedModel, encoded: &EncodedBatch, pooling: &str, normalize: &str) -> AppResult<RunOutputs> {
    let hidden = model.config.hidden_size;
    let matrix = word_embedding_matrix(model, hidden)?;
    let mut embeddings = Vec::with_capacity(encoded.input_ids.len());

    for (row, mask) in encoded.input_ids.iter().zip(&encoded.attention_mask) {
        let mut out = vec![0.0_f32; hidden];
        let mut count = 0.0_f32;
        if pooling == "cls" {
            add_token_embedding(&matrix, row[0] as usize, &mut out);
        } else {
            for (token, is_real) in row.iter().zip(mask) {
                if *is_real == 0 {
                    continue;
                }
                add_token_embedding(&matrix, *token as usize, &mut out);
                count += 1.0;
            }
            if count > 0.0 {
                out.iter_mut().for_each(|v| *v /= count);
            }
        }
        if normalize == "l2" {
            let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                out.iter_mut().for_each(|v| *v /= norm);
            }
        }
        embeddings.push(out);
    }

    Ok(RunOutputs {
        embeddings: Some(embeddings),
        ..RunOutputs::default()
    })
}

fn run_classifier(model: &LoadedModel, encoded: &EncodedBatch, prob: &str) -> AppResult<RunOutputs> {
    let embed = run_embedding(model, encoded, "cls", "none")
        .and_then(|o| o.embeddings.ok_or(AppError::InvalidInput("missing embeddings".to_string())))?;

    let (weight, out_dim) = classifier_weight(model)?;
    let bias = classifier_bias(model, out_dim)?;
    let hidden = model.config.hidden_size;

    let mut logits = Vec::with_capacity(embed.len());
    let mut probs = Vec::with_capacity(embed.len());
    for sample in &embed {
        let mut row = vec![0.0_f32; out_dim];
        for label_idx in 0..out_dim {
            let start = label_idx * hidden;
            let mut value = bias[label_idx];
            for h in 0..hidden {
                value += weight[start + h] * sample[h];
            }
            row[label_idx] = value;
        }
        if prob == "softmax" {
            probs.push(softmax(&row));
        } else if prob == "sigmoid" {
            probs.push(row.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect());
        }
        logits.push(row);
    }

    Ok(RunOutputs {
        logits: Some(logits),
        probs: if probs.is_empty() { None } else { Some(probs) },
        ..RunOutputs::default()
    })
}

fn run_reranker(model: &LoadedModel, encoded: &EncodedBatch, top_k: usize) -> AppResult<RunOutputs> {
    let outputs = run_classifier(model, encoded, "none")?;
    let logits = outputs
        .logits
        .ok_or(AppError::InvalidInput("missing classifier logits".to_string()))?;
    let mut scores = logits.iter().map(|row| row[0]).collect::<Vec<_>>();
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap_or(Ordering::Equal));
    if top_k > 0 && order.len() > top_k {
        order.truncate(top_k);
    }
    scores = scores;
    Ok(RunOutputs {
        scores: Some(scores),
        order: Some(order),
        ..RunOutputs::default()
    })
}

fn word_embedding_matrix(model: &LoadedModel, hidden: usize) -> AppResult<&Vec<f32>> {
    let key = format!("{}embeddings.word_embeddings.weight", model.diagnostics.root_prefix);
    model
        .tensors
        .get(&key)
        .ok_or(AppError::InvalidInput(format!("missing embedding key: {key}")))
        .and_then(|v| {
            if v.len() % hidden != 0 {
                Err(AppError::InvalidInput("embedding tensor size mismatch".to_string()))
            } else {
                Ok(v)
            }
        })
}

fn add_token_embedding(matrix: &[f32], token_id: usize, out: &mut [f32]) {
    let hidden = out.len();
    let offset = token_id * hidden;
    if offset + hidden <= matrix.len() {
        for idx in 0..hidden {
            out[idx] += matrix[offset + idx];
        }
    }
}

fn classifier_weight(model: &LoadedModel) -> AppResult<(&Vec<f32>, usize)> {
    let hidden = model.config.hidden_size;
    match model.arch {
        ModelArch::Bert => {
            let key = format!("{}classifier.weight", model.diagnostics.root_prefix);
            let weight = model
                .tensors
                .get(&key)
                .ok_or(AppError::InvalidInput(format!("missing classifier key: {key}")))?;
            Ok((weight, weight.len() / hidden))
        }
        ModelArch::XlmRoberta => {
            let key = format!("{}classifier.out_proj.weight", model.diagnostics.root_prefix);
            let weight = model
                .tensors
                .get(&key)
                .ok_or(AppError::InvalidInput(format!("missing classifier key: {key}")))?;
            Ok((weight, weight.len() / hidden))
        }
    }
}

fn classifier_bias(model: &LoadedModel, out_dim: usize) -> AppResult<Vec<f32>> {
    let key = match model.arch {
        ModelArch::Bert => format!("{}classifier.bias", model.diagnostics.root_prefix),
        ModelArch::XlmRoberta => format!("{}classifier.out_proj.bias", model.diagnostics.root_prefix),
    };
    let bias = model
        .tensors
        .get(&key)
        .ok_or(AppError::InvalidInput(format!("missing classifier bias key: {key}")))?;
    if bias.len() != out_dim {
        return Err(AppError::InvalidInput("classifier bias shape mismatch".to_string()));
    }
    Ok(bias.clone())
}

fn softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = row.iter().map(|v| (*v - max).exp()).collect::<Vec<_>>();
    let sum = exp.iter().sum::<f32>();
    exp.iter().map(|v| v / sum).collect()
}
