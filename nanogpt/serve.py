import os
import contextlib
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import torch

from nanogpt.generate import load_model_from_checkpoint, generate

# Global model and tokenizer state
MODEL = None
TOKENIZER = None
DEVICE = "cpu"


def load_model_global(checkpoint_path: Optional[str] = None, config_path: Optional[str] = None):
    global MODEL, TOKENIZER, DEVICE
    if checkpoint_path is None:
        checkpoint_path = os.environ.get("CHECKPOINT_PATH", None)

    if not checkpoint_path:
        # If no checkpoint specified, we leave model uninitialized (or lazy)
        return False

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL, TOKENIZER = load_model_from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        device=DEVICE,
    )
    return True


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: load checkpoint once at startup if env var or config is provided
    try:
        load_model_global()
    except Exception as e:
        # Print error log but don't fail startup so server can return error on health/generate if missing
        print(f"Warning: Failed to load model at startup: {e}")
    yield
    # Shutdown logic if any
    pass


app = FastAPI(title="NanoGPT Serving API", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=100, gt=0, le=1000, description="Number of tokens to generate (1 to 1000)")
    temperature: float = Field(default=1.0, ge=0.0, description="Sampling temperature")
    top_k: Optional[int] = Field(default=None, gt=0, description="Top-k filtering parameter")
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0, description="Top-p nucleus filtering threshold")


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=(MODEL is not None and TOKENIZER is not None),
    )


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(request: GenerateRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please set CHECKPOINT_PATH environment variable.",
        )

    prompt_text = request.prompt
    if prompt_text:
        try:
            encoded_prompt = TOKENIZER.encode(prompt_text)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid prompt string for vocabulary: {e}",
            )
    else:
        encoded_prompt = [0]

    prompt_tensor = torch.tensor(encoded_prompt, dtype=torch.long, device=DEVICE)

    try:
        out_tokens = generate(
            MODEL,
            prompt_tensor,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
        generated_text = TOKENIZER.decode(out_tokens.tolist())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )

    return GenerateResponse(
        prompt=prompt_text,
        generated_text=generated_text,
    )
