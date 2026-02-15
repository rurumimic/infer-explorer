use tokenizers::{EncodeInput, Encoding, PaddingParams, Tokenizer, TruncationParams};

use crate::{error::AppError, schema::EncodedBatch};

#[derive(Debug)]
pub struct TokenizerEngine {
    inner: Tokenizer,
}

impl TokenizerEngine {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AppError> {
        let inner = Tokenizer::from_bytes(bytes)
            .map_err(|e| AppError::Tokenizer(format!("failed to parse tokenizer.json: {e}")))?;
        Ok(Self { inner })
    }

    pub fn encode_texts(
        &mut self,
        texts: &[String],
        max_length: usize,
    ) -> Result<EncodedBatch, AppError> {
        self.apply_padding(max_length);
        self.apply_truncation(max_length)?;
        let mut input_ids = Vec::with_capacity(texts.len());
        let mut attention_mask = Vec::with_capacity(texts.len());
        let mut token_type_ids = Vec::with_capacity(texts.len());
        for text in texts {
            let encoding = self
                .inner
                .encode(text.as_str(), true)
                .map_err(|e| AppError::Tokenizer(format!("encode failed: {e}")))?;
            push_encoding(&encoding, &mut input_ids, &mut attention_mask, &mut token_type_ids);
        }
        Ok(EncodedBatch {
            input_ids,
            attention_mask,
            token_type_ids: Some(token_type_ids),
        })
    }

    pub fn encode_pairs(
        &mut self,
        query: &str,
        docs: &[String],
        max_length: usize,
    ) -> Result<EncodedBatch, AppError> {
        self.apply_padding(max_length);
        self.apply_truncation(max_length)?;
        let mut input_ids = Vec::with_capacity(docs.len());
        let mut attention_mask = Vec::with_capacity(docs.len());
        let mut token_type_ids = Vec::with_capacity(docs.len());
        for doc in docs {
            let encoding = self
                .inner
                .encode(EncodeInput::Dual(query.into(), doc.as_str().into()), true)
                .map_err(|e| AppError::Tokenizer(format!("pair encode failed: {e}")))?;
            push_encoding(&encoding, &mut input_ids, &mut attention_mask, &mut token_type_ids);
        }
        Ok(EncodedBatch {
            input_ids,
            attention_mask,
            token_type_ids: Some(token_type_ids),
        })
    }

    fn apply_padding(&mut self, max_length: usize) {
        self.inner.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(max_length),
            ..PaddingParams::default()
        }));
    }

    fn apply_truncation(&mut self, max_length: usize) -> Result<(), AppError> {
        self.inner
            .with_truncation(Some(TruncationParams {
                max_length,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                ..TruncationParams::default()
            }))
            .map(|_| ())
            .map_err(|e| AppError::Tokenizer(format!("failed to set truncation: {e}")))
    }
}

fn push_encoding(
    encoding: &Encoding,
    input_ids: &mut Vec<Vec<u32>>,
    attention_mask: &mut Vec<Vec<u32>>,
    token_type_ids: &mut Vec<Vec<u32>>,
) {
    input_ids.push(encoding.get_ids().to_vec());
    attention_mask.push(encoding.get_attention_mask().to_vec());
    token_type_ids.push(encoding.get_type_ids().to_vec());
}
