export class WasmRuntime {
  load_model() {
    return { applied_keys: 0, missing_keys: [], extra_keys: [], root_prefix: 'auto' };
  }
  encode_texts(texts, maxLength) {
    const rows = texts.map(() => Array(maxLength).fill(0));
    return { input_ids: rows, attention_mask: rows, token_type_ids: rows };
  }
  encode_pairs(_query, docs, maxLength) {
    const rows = docs.map(() => Array(maxLength).fill(0));
    return { input_ids: rows, attention_mask: rows, token_type_ids: rows };
  }
  run_request(req) {
    return {
      schema_version: 1,
      run_id: req.run_id,
      task: req.task,
      env: { runtime: 'browser-wasm-webgpu', engine: 'burn' },
      timing_ms: { tokenize: 0, inference: 0, postprocess: 0, total: 0 },
      outputs: { embeddings: [[0, 0, 0]], logits: [[0, 0]], scores: [0], order: [0] }
    };
  }
  compare_results() {
    return { max_abs_diff: 0, mean_abs_diff: 0, avg_cosine: 1 };
  }
}
