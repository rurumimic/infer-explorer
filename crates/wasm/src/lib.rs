use infer_explorer_app::{
    AppError, InferenceEngine, LoadedModel, RunRequest, TokenizerEngine, compare_run_results,
};
use serde_wasm_bindgen::{from_value, to_value};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmRuntime {
    engine: InferenceEngine,
    tokenizer: Option<TokenizerEngine>,
}

#[wasm_bindgen]
impl WasmRuntime {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            engine: InferenceEngine::new(),
            tokenizer: None,
        }
    }

    #[wasm_bindgen]
    pub fn load_model(
        &mut self,
        config_bytes: Vec<u8>,
        tokenizer_bytes: Vec<u8>,
        weight_bytes: Vec<u8>,
        prefix_override: Option<String>,
    ) -> Result<JsValue, JsValue> {
        let tokenizer = TokenizerEngine::from_bytes(&tokenizer_bytes).map_err(err_to_js)?;
        let model = LoadedModel::from_bytes(&config_bytes, &weight_bytes, prefix_override.as_deref())
            .map_err(err_to_js)?;
        let diagnostics = model.diagnostics.clone();
        self.tokenizer = Some(tokenizer);
        self.engine.load(model);
        to_value(&diagnostics).map_err(|e| JsValue::from_str(&format!("serialize failed: {e}")))
    }

    #[wasm_bindgen]
    pub fn encode_texts(&mut self, texts: JsValue, max_length: usize) -> Result<JsValue, JsValue> {
        let list: Vec<String> = from_value(texts).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let tokenizer = self.tokenizer.as_mut().ok_or(JsValue::from_str("tokenizer is not loaded"))?;
        let encoded = tokenizer.encode_texts(&list, max_length).map_err(err_to_js)?;
        to_value(&encoded).map_err(|e| JsValue::from_str(&format!("serialize failed: {e}")))
    }

    #[wasm_bindgen]
    pub fn encode_pairs(&mut self, query: String, docs: JsValue, max_length: usize) -> Result<JsValue, JsValue> {
        let list: Vec<String> = from_value(docs).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let tokenizer = self.tokenizer.as_mut().ok_or(JsValue::from_str("tokenizer is not loaded"))?;
        let encoded = tokenizer.encode_pairs(&query, &list, max_length).map_err(err_to_js)?;
        to_value(&encoded).map_err(|e| JsValue::from_str(&format!("serialize failed: {e}")))
    }

    #[wasm_bindgen]
    pub fn run_request(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let request: RunRequest = from_value(request).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let result = self.engine.run_from_request(request).map_err(err_to_js)?;
        to_value(&result).map_err(|e| JsValue::from_str(&format!("serialize failed: {e}")))
    }

    #[wasm_bindgen]
    pub fn compare_results(&self, browser: JsValue, reference: JsValue) -> Result<JsValue, JsValue> {
        let browser = from_value(browser).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let reference = from_value(reference).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let report = compare_run_results(&browser, &reference);
        to_value(&report).map_err(|e| JsValue::from_str(&format!("serialize failed: {e}")))
    }
}

fn err_to_js(err: AppError) -> JsValue {
    JsValue::from_str(&err.to_string())
}
