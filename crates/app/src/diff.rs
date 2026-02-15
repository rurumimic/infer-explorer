use serde::{Deserialize, Serialize};

use crate::schema::{RunResult, Task};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompareReport {
    pub task: String,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub avg_cosine: Option<f32>,
    pub top1_match: Option<bool>,
    pub reranker_top10_overlap: Option<f32>,
}

pub fn compare_run_results(browser: &RunResult, reference: &RunResult) -> CompareReport {
    let mut report = CompareReport {
        task: format!("{:?}", browser.task),
        ..CompareReport::default()
    };
    match browser.task {
        Task::Embedding => {
            if let (Some(b), Some(r)) = (&browser.outputs.embeddings, &reference.outputs.embeddings) {
                let (max, mean, cosine) = tensor_metrics_2d(b, r);
                report.max_abs_diff = max;
                report.mean_abs_diff = mean;
                report.avg_cosine = Some(cosine);
            }
        }
        Task::Classifier => {
            if let (Some(b), Some(r)) = (&browser.outputs.logits, &reference.outputs.logits) {
                let (max, mean, _) = tensor_metrics_2d(b, r);
                report.max_abs_diff = max;
                report.mean_abs_diff = mean;
                report.top1_match = Some(top1(b) == top1(r));
            }
        }
        Task::Reranker => {
            if let (Some(b), Some(r)) = (&browser.outputs.scores, &reference.outputs.scores) {
                let (max, mean) = tensor_metrics_1d(b, r);
                report.max_abs_diff = max;
                report.mean_abs_diff = mean;
            }
            if let (Some(b), Some(r)) = (&browser.outputs.order, &reference.outputs.order) {
                report.reranker_top10_overlap = Some(topk_overlap(b, r, 10));
            }
        }
    }
    report
}

fn tensor_metrics_2d(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f32, f32, f32) {
    let mut max: f32 = 0.0;
    let mut sum = 0.0;
    let mut count = 0.0;
    let mut cosine_sum = 0.0;
    let mut cosine_count = 0.0;
    for (av, bv) in a.iter().zip(b) {
        let mut dot = 0.0;
        let mut an = 0.0;
        let mut bn = 0.0;
        for (x, y) in av.iter().zip(bv) {
            let diff = (*x - *y).abs();
            max = max.max(diff);
            sum += diff;
            count += 1.0;
            dot += x * y;
            an += x * x;
            bn += y * y;
        }
        if an > 0.0 && bn > 0.0 {
            cosine_sum += dot / (an.sqrt() * bn.sqrt());
            cosine_count += 1.0;
        }
    }
    (max, if count > 0.0 { sum / count } else { 0.0 }, if cosine_count > 0.0 { cosine_sum / cosine_count } else { 0.0 })
}

fn tensor_metrics_1d(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max: f32 = 0.0;
    let mut sum = 0.0;
    let mut count = 0.0;
    for (x, y) in a.iter().zip(b) {
        let diff = (*x - *y).abs();
        max = max.max(diff);
        sum += diff;
        count += 1.0;
    }
    (max, if count > 0.0 { sum / count } else { 0.0 })
}

fn top1(values: &[Vec<f32>]) -> Vec<usize> {
    values
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect()
}

fn topk_overlap(a: &[usize], b: &[usize], k: usize) -> f32 {
    let sa = a.iter().take(k).copied().collect::<std::collections::BTreeSet<_>>();
    let sb = b.iter().take(k).copied().collect::<std::collections::BTreeSet<_>>();
    let inter = sa.intersection(&sb).count();
    inter as f32 / k as f32
}
