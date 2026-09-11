# NanoGPT: Decoder-Only Transformer from Scratch

An implementation of a decoder-only, GPT-style Transformer built completely from scratch in PyTorch.

Unlike projects that wrap high-level libraries like HuggingFace `transformers`, every single architectural building block in this repository—character-level tokenization, positional embeddings, causal multi-head self-attention, pre-layer-norm transformer blocks, autoregressive generation with sampling strategies, and HTTP serving—is implemented directly using PyTorch primitives and individually unit-tested.

---

## Architecture Overview

NanoGPT follows the autoregressive decoder-only Transformer architecture (GPT-2 style):

- **Character Tokenizer (`nanogpt/tokenizer.py`)**: A character-level vocabulary mapping unique characters in the corpus to integer token IDs and vice versa.
- **Positional Embeddings (`nanogpt/model.py`)**: A learned positional embedding lookup table (`[block_size, n_embd]`) added directly to the token embeddings (`[vocab_size, n_embd]`).
- **Causal Self-Attention (`nanogpt/model.py`)**: Multi-Head Attention with a lower-triangular causal mask to prevent tokens from attending to future positions during training and inference.
- **Transformer Block (`nanogpt/model.py`)**: Pre-LayerNorm block ordering (`LayerNorm -> CausalSelfAttention -> Residual Add -> LayerNorm -> FeedForward MLP -> Residual Add`). The MLP uses a $4\times$ hidden expansion dimension with GELU activations.
- **Full Model (`nanogpt/model.py`)**: Composes token and position embeddings, $N$ stacked Transformer Blocks, a final LayerNorm, and a linear language modeling head mapping representations to vocabulary logits.

---

## Installation

### 1. Install Dependencies

Install the runtime dependencies:
```bash
pip install -r requirements.txt
```

For development and running unit tests, install dev dependencies:
```bash
pip install -r requirements-dev.txt
```

### 2. Install CLI Package

Install `nanogpt` as an editable CLI package:
```bash
pip install -e .
```

---

## Usage

### Training

To train a model using a YAML configuration file:

```bash
nanogpt train --config configs/tinyshakespeare.yaml
```

The training process automatically downloads the Tiny Shakespeare dataset if it is not present at `data/tiny_shakespeare.txt`, trains the model according to the hyperparameters specified in the config, logs training/validation loss, and saves checkpoints to the `checkpoints/` directory.

### Text Generation

Generate text autoregressively from a trained model checkpoint:

```bash
nanogpt generate \
  --checkpoint checkpoints/checkpoint_latest.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-k 5 \
  --top-p 0.9
```

Available flags:
- `--checkpoint`: Path to PyTorch model checkpoint (`.pt`).
- `--prompt`: Initial string prompt for generation (default: empty string).
- `--max-new-tokens`: Number of new tokens to generate (default: 100).
- `--temperature`: Sampling temperature ($>0.0$, default: 1.0). Lower values make outputs more deterministic.
- `--top-k`: Restricts token selection to top-$k$ highest probability tokens (optional).
- `--top-p`: Cumulative probability threshold for nucleus sampling (optional).
- `--device`: Compute device (`cpu` or `cuda`).

### Serving via HTTP API

Start a FastAPI HTTP server serving generation requests:

```bash
nanogpt serve --checkpoint checkpoints/checkpoint_latest.pt --host 0.0.0.0 --port 8000
```

Alternatively, set the `CHECKPOINT_PATH` environment variable:
```bash
export CHECKPOINT_PATH=checkpoints/checkpoint_latest.pt
nanogpt serve --host 0.0.0.0 --port 8000
```

#### Example HTTP Requests

**Health Check (`GET /health`):**
```bash
curl http://127.0.0.1:8000/health
```
*Response:*
```json
{"status":"ok","model_loaded":true}
```

**Generate Text (`POST /generate`):**
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "ROMEO:",
    "max_new_tokens": 50,
    "temperature": 0.8
  }'
```
*Response:*
```json
{
  "prompt": "ROMEO:",
  "generated_text": "ROMEO:\nIOeree tausl youand wantis ther owinethe.\nAnilill"
}
```

### Docker Support

Build the Docker container:
```bash
docker build -t nanogpt .
```

Run the container with a local checkpoint mounted:
```bash
docker run -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -e CHECKPOINT_PATH=checkpoints/checkpoint_latest.pt \
  nanogpt
```

---

## Sample Generations & Model Performance

The outputs below are raw, unedited text samples generated from `checkpoint_latest.pt` after a brief 500-step training run on Tiny Shakespeare using `configs/tinyshakespeare.yaml` ($d_{model}=64$, $N_{layers}=4$, $N_{heads}=4$, context length = 64).

### Sample 1: Low Temperature (`--temperature 0.2`)
```
ROMEO:
I:
The hathe and the the he the he the athe hillllllour he wime the wind bourere wime the me t me w
```

### Sample 2: Top-K Sampling (`--temperature 0.8 --top-k 5`)
```
ROMEO:
MM blllin ar ator thilou histhind we heanot h thes te mereremom ared thin be, wend wanord matom the
```

### Sample 3: Top-P Nucleus Sampling (`--temperature 0.8 --top-p 0.9`)
```
ROMEO:
RI lllol wome ounorofe harto t the ce fane th aly angouns, s t ait meaulerelesuurimameroway berer h
```

> **Note on Outputs:**
> The generated text consists of pseudo-words and character repetition. This is expected because this run is a small proof-of-concept (~500 steps on a ~1MB character corpus with a tiny 64-dimensional model). Achieving coherent English prose would require scaling model capacity ($d_{model} \ge 768$, layers $\ge 12$), using subword tokenization (e.g. BPE), increasing context length, and training over tens or hundreds of thousands of steps on a larger dataset.

---

## Testing

The project includes a comprehensive unit test suite covering every core component:

```bash
pytest -v
```

Current test status: **30 passed** across the following test suites:
- `tests/test_attention.py`: Causal masking, attention matrix output dimensions, and future-leakage prevention.
- `tests/test_block.py`: Transformer block residual forward passes and representation transformations.
- `tests/test_data.py`: Dataset downloading, non-overlapping train/val split logic, and batch offset matching.
- `tests/test_generate.py`: Temperature scaling, top-$k$ filtering, top-$p$ nucleus sampling, and end-to-end checkpoint loading.
- `tests/test_model.py`: Model forward pass shape validation and golden overfit sanity checks.
- `tests/test_positional.py`: Positional embedding lookup shapes, index uniqueness, and deterministic vector behavior.
- `tests/test_serve.py`: FastAPI `/health` and `/generate` endpoints, status codes, and input validation errors.
- `tests/test_tokenizer.py`: Character vocabulary building, encode/decode roundtrips, and unknown token exception handling.

---

## Continuous Integration (CI)

Continuous Integration is configured via GitHub Actions in `.github/workflows/ci.yml`.

On every push or pull request targeting the `main` branch, CI:
1. Provisions a Python 3.10 environment.
2. Installs requirements and the `nanogpt` package in editable mode.
3. Executes `pytest -v` to verify that all unit tests pass without regressions.

---

## Project Structure

```
.
├── configs/
│   └── tinyshakespeare.yaml    # YAML configuration for dataset, model, and training
├── nanogpt/
│   ├── __init__.py
│   ├── cli.py                  # Command-line interface entry point (train, generate, serve)
│   ├── data.py                 # Dataset download, loading, and batch sampling
│   ├── generate.py             # Autoregressive sampling logic (temperature, top-k, top-p)
│   ├── model.py                # GPT components (Attention, Transformer Block, GPT model)
│   ├── serve.py                # FastAPI HTTP application (/health, /generate)
│   ├── tokenizer.py            # Character-level tokenizer
│   ├── train.py                # Training loop and loss estimation
│   └── utils.py                # Seed utilities
├── tests/
│   ├── test_attention.py       # Causal self-attention unit tests
│   ├── test_block.py           # Transformer block unit tests
│   ├── test_data.py            # Data pipeline unit tests
│   ├── test_generate.py        # Generation & sampling logic unit tests
│   ├── test_model.py           # GPT architecture unit tests
│   ├── test_positional.py      # Positional embedding unit tests
│   ├── test_serve.py           # FastAPI serving unit tests
│   └── test_tokenizer.py       # Tokenizer unit tests
├── .dockerignore               # Docker build ignore rules
├── .gitignore                  # Git ignore rules
├── Dockerfile                  # Container build for model serving
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development and testing dependencies
└── setup.py                    # Package configuration
```
