# NanoGPT: Decoder-Only Transformer from Scratch

An implementation of a decoder-only, GPT-style Transformer built completely from scratch in PyTorch.

Unlike projects that wrap high-level libraries like HuggingFace `transformers`, every architectural building block in this repository — character-level tokenization, positional embeddings, causal multi-head self-attention, pre-layer-norm transformer blocks, autoregressive generation with sampling strategies, and HTTP serving — is implemented directly using PyTorch primitives and individually unit-tested.

---

## Architecture Overview

NanoGPT follows the autoregressive decoder-only Transformer architecture (GPT-2 style):

- **Character Tokenizer (`nanogpt/tokenizer.py`)**: A character-level vocabulary mapping unique characters in the corpus to integer token IDs and vice versa.
- **Positional Embeddings (`nanogpt/model.py`)**: A learned positional embedding lookup table added to the token embeddings.
- **Causal Self-Attention (`nanogpt/model.py`)**: Multi-head attention with a lower-triangular causal mask preventing tokens from attending to future positions.
- **Transformer Block (`nanogpt/model.py`)**: Pre-LayerNorm block ordering (LayerNorm → Causal Self-Attention → Residual Add → LayerNorm → Feed-Forward MLP → Residual Add).
- **Full Model (`nanogpt/model.py`)**: Composes token and position embeddings, N stacked Transformer Blocks, a final LayerNorm, and a linear language-modeling head.

---

## Installation

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for running tests
```

Install the CLI:
```bash
pip install -e .
```

---

## Usage

### Training

```bash
nanogpt train
```

Run `nanogpt train --help` for the full list of flags (data path, step count, batch size, checkpoint output directory, etc.). Checkpoints are saved to the `checkpoints/` directory during training.

### Text Generation

```bash
nanogpt generate \
  --checkpoint checkpoints/checkpoint_latest.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 150 \
  --temperature 0.8 \
  --top-k 10 \
  --top-p 0.9
```

Flags:
- `--checkpoint`: path to a `.pt` checkpoint
- `--prompt`: initial string prompt
- `--max-new-tokens`: number of tokens to generate
- `--temperature`: sampling temperature (lower = more deterministic)
- `--top-k`: restrict sampling to top-k tokens (optional)
- `--top-p`: nucleus sampling threshold (optional)

### Serving via HTTP API

```bash
nanogpt serve --checkpoint checkpoints/checkpoint_latest.pt --host 0.0.0.0 --port 8000
```

Or via environment variable:
```bash
export CHECKPOINT_PATH=checkpoints/checkpoint_latest.pt
nanogpt serve --host 0.0.0.0 --port 8000
```

**Health check:**
```bash
curl http://127.0.0.1:8000/health
```
```json
{"status":"ok","model_loaded":true}
```

**Generate:**
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "abc", "max_new_tokens": 10, "temperature": 0.8, "top_k": 5}'
```
```json
{"prompt":"abc","generated_text":"abcgggggggggg"}
```

### Docker

```bash
docker build -t nanogpt-scratch .
docker run -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -e CHECKPOINT_PATH=checkpoints/checkpoint_latest.pt \
  nanogpt-scratch
```

---

## Sample Generation & Model Performance

Output below is raw, unedited text from `checkpoint_latest.pt` after a ~500-step training run:

```
Prompt: "ROMEO:"
Flags: --max-new-tokens 150 --temperature 0.8 --top-k 10 --top-p 0.9

ROMEO:
INhe beer thit tomoro ange an tholy ald t meat thele wound wit mememe me fan wheadere d we wand wou d fathe themad d momyomy thatowerd t merir winthe
```

> **Note:** This output is character-soup, not coherent English — expected for a ~500-step run on a tiny model and small corpus. Getting to coherent prose would require a larger model (more layers/heads/embedding dim), subword tokenization instead of character-level, a longer context window, and training for tens of thousands of steps on a much larger corpus.

---

## Testing

```bash
pytest -v
```

Current status: **30 passed**, covering:
- `tests/test_attention.py` — causal masking, output shapes, no future leakage
- `tests/test_block.py` — transformer block forward pass and residuals
- `tests/test_data.py` — batch shapes, train/val split, batch offsets
- `tests/test_generate.py` — temperature, top-k, top-p, end-to-end checkpoint generation
- `tests/test_model.py` — model forward shape, golden overfit test
- `tests/test_positional.py` — positional embedding shape and uniqueness
- `tests/test_serve.py` — `/health` and `/generate` endpoints, input validation
- `tests/test_tokenizer.py` — vocab building, encode/decode roundtrip, error handling

---

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request to `main`:
1. Sets up Python 3.10
2. Installs dependencies
3. Runs `pytest -v` — the build fails if any test fails

---

## Project Structure

```
.
├── nanogpt/
│   ├── __init__.py
│   ├── cli.py           # CLI entry point (train, generate, serve)
│   ├── data.py           # Dataset loading and batch sampling
│   ├── generate.py       # Sampling logic (temperature, top-k, top-p)
│   ├── model.py           # Attention, transformer block, GPT model
│   ├── serve.py           # FastAPI app (/health, /generate)
│   ├── tokenizer.py       # Character-level tokenizer
│   └── train.py           # Training loop
├── tests/
│   ├── test_attention.py
│   ├── test_block.py
│   ├── test_data.py
│   ├── test_generate.py
│   ├── test_model.py
│   ├── test_positional.py
│   ├── test_serve.py
│   └── test_tokenizer.py
├── .dockerignore
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```