# Models

```bash
du -h --max-depth 1 models | sort -hr

13G     models
6.9G    models/.venv
3.2G    models/BAAI
2.1G    models/jinaai
955M    models/SamLowe
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
hf download "jinaai/jina-embeddings-v5-text-nano" --local-dir jinaai/jina-embeddings-v5-text-nano
hf download "jinaai/jina-embeddings-v2-base-en" --local-dir jinaai/jina-embeddings-v2-base-en
```

```bash
mkdir -p BAAI
hf download "BAAI/bge-reranker-base" --local-dir BAAI/bge-reranker-base
```

```bash
mkdir -p SamLowe
hf download "SamLowe/roberta-base-go_emotions" --local-dir SamLowe/roberta-base-go_emotions
```

## Test a model

### jinaai/jina-embeddings-v5-text-nano

- model card: [jinaai/jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano)

### Install dependencies

- Dao-AILab/flash-attention: [releases](https://github.com/Dao-AILab/flash-attention/releases)

check torch version:

```bash
uv pip install torch==2.9
uv pip install transformers peft
```

(optional) install flash attention:

> WARNING! Don't build flash attention from source, install the pre-built wheel for your CUDA version.

```bash
(.venv) $ python -c "import torch; print(torch.__version__)"
2.9.0+cu128

(.venv) $ python -c "import torch; print(torch.version.cuda)"
12.8
```

```bash
uv pip install flash-attn --no-build-isolation --verbose

DEBUG torch.__version__  = 2.9.0+cu128
DEBUG
DEBUG
DEBUG running bdist_wheel
DEBUG Guessing wheel URL:  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
DEBUG Raw wheel path ~/.cache/uv/builds-v0/.tmpiisx8o/.tmp-30s5cljb/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
DEBUG Released lock at `/tmp/uv-setuptools-b9314863d1eb0e4c.lock`
DEBUG Finished building: flash-attn==2.8.3
      Built flash-attn==2.8.3
DEBUG Released lock at `~/.cache/uv/sdists-v9/pypi/flash-attn/2.8.3/.lock`
Prepared 1 package without build isolation in 30.87s
Installed 1 package in 1ms
 + einops==0.8.2
 + flash-attn==2.8.3
```

#### Test the model

```py
model_id='./jinaai/jina-embeddings-v5-text-nano'

from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    _attn_implementation="flash_attention_2",  # Recommended but optional
    dtype=torch.bfloat16,  # Recommended for GPUs
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# == device(type='cuda')

model = model.to(device=device)
# == JinaEmbeddingsV5Model(...)
```

##### example: Retrieval Task

```py
query_embeddings = model.encode(
    sentences=["Overview of climate change impacts on coastal cities"],
    task="retrieval",
    prompt_name="query",
)

document_embeddings = model.encode(
    sentences=[
        "Climate change has led to rising sea levels, increased frequency of extreme weather events..."
    ],
    task="retrieval",
    prompt_name="document",
)
```

#### Reranker

- model card: [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)

```py
model_id='./BAAI/bge-reranker-base'

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

print(scores) # tensor([-7.2112,  6.2000])
```

#### Classifier

- model card: [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)

```py
model_id='./SamLowe/roberta-base-go_emotions'

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "I am so happy today!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

print(logits)

probs = torch.sigmoid(logits)
print(probs)

predicted_labels = [
    model.config.id2label[i]
    for i, p in enumerate(probs[0])
    if p > 0.5
]
print(predicted_labels) # ['joy']
```

