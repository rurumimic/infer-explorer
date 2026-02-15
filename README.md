# infer-explorer

Burn(WebGPU/WASM) 기반 로컬 Transformer 테스트용 MVP 모노레포입니다.

> 핵심 목표: 브라우저에서 로컬 파일(`config.json`, `tokenizer.json`, `model.safetensors`)을 업로드해 추론하고, 외부 실행 결과 JSON과 Compare 화면에서 정량 비교.

## 모노레포 구성

- `crates/app`: 공용 타입, config 파서, tokenizer wrapper, 모델 로더, 추론, diff metrics
- `crates/wasm`: wasm-bindgen 바인딩 레이어
- `frontend`: SvelteKit + Tailwind UI (`/load`, `/tokenizer`, `/playground`, `/compare`)

## 모델 준비 (MVP-1)

필수 파일(로컬 디렉토리):

1. `config.json`
2. `tokenizer.json`
3. `model.safetensors` (단일 파일만 지원)
4. 선택: `id2label.json`

지원 아키텍처:

- BERT (`model_type=bert`)
- XLM-RoBERTa (`model_type=xlm-roberta`)

## 개발 실행

### 1) Rust 워크스페이스 검사

```bash
cargo check
```

### 2) 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev
```

## 화면 플로우

### `/load`

- `config.json`, `tokenizer.json`, `model.safetensors` 업로드
- 로드 진단:
  - 적용된 키 수
  - 누락 키 목록
  - root prefix auto-detect (`bert.`, `roberta.`)

### `/tokenizer`

- 단일 문장 인코딩
- pair(query/doc) 인코딩
- `encoded` JSON 확인/복사

### `/playground`

- Embedding / Classifier / Reranker 탭
- 실행 결과 확인
- `RunRequest JSON export`, `RunResult JSON export`

### `/compare`

1. `RunRequest JSON` 입력(또는 Playground 최신 요청 사용)
2. 외부 환경 결과인 `Reference RunResult JSON` 입력
3. `Run in Browser (from encoded)` 실행
4. diff metrics 확인

## RunRequest / RunResult 스키마

샘플 파일:

- `frontend/static/samples/run_request.embedding.json`
- `frontend/static/samples/run_result.reference.embedding.json`

핵심 원칙:

- `RunRequest.encoded`에 `input_ids`, `attention_mask`, `token_type_ids` 포함
- Compare는 encoded 기반 재실행으로 tokenizer 차이를 최소화

## 외부(PyTorch 등) 결과 생성 개요

1. 외부 런타임에서 동일 입력으로 추론
2. 본 레포의 `RunResult` 스키마로 JSON 저장 (`runtime=reference-external`, `engine=pytorch` 등)
3. `/compare`에서 업로드하여 브라우저 결과와 비교

## 현재 MVP 구현 노트

- 드롭아웃은 추론 경로에서 비활성 가정
- Diff metric 제공:
  - Embedding: cosine / abs diff
  - Classifier: top-1 match / logits abs diff
  - Reranker: top-k overlap / score abs diff
- tokenizer는 `tokenizers` crate 기반 로컬 JSON 로드를 사용

## 제한 사항 (MVP-1)

- sharded safetensors 미지원
- Hub 다운로드 미지원
- 서버/DB 미사용
- tokenizer 멀티스레드 최적화 미포함
