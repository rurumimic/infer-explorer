# Models

```bash
models/
└── jinaai/
    └── jina-embeddings-v2-base-en/
        ├── 1_Pooling/
        ├── config.json
        ├── # ...
        ├── model.safetensors
        └── tokenizer.json
```

## Download models to a local folder

```bash
cd models
```

Python .venv

```bash
uv venv
source .venv/bin/activate
```

Install HuggingFace CLI:

```bash
uv pip install "huggingface_hub"
```

### Download a model

```bash
mkdir -p jinaai
hf download "jinaai/jina-embeddings-v2-base-en" --local-dir jinaai/jina-embeddings-v2-base-en
```
