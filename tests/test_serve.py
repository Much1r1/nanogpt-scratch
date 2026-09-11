import pytest
import torch
from fastapi.testclient import TestClient

import nanogpt.serve as serve_module
from nanogpt.model import GPT


@pytest.fixture
def dummy_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    config = {
        "model": {
            "vocab_size": 10,
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 1,
            "block_size": 8,
            "dropout": 0.0,
        }
    }
    model = GPT(**config["model"])
    chars = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "chars": chars,
        },
        checkpoint_path,
    )
    return str(checkpoint_path)


def test_health_when_unloaded(monkeypatch):
    monkeypatch.setattr(serve_module, "MODEL", None)
    monkeypatch.setattr(serve_module, "TOKENIZER", None)
    client = TestClient(serve_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_health_when_loaded(dummy_checkpoint):
    serve_module.load_model_global(dummy_checkpoint)
    client = TestClient(serve_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_generate_endpoint_success(dummy_checkpoint):
    serve_module.load_model_global(dummy_checkpoint)
    client = TestClient(serve_module.app)
    payload = {
        "prompt": "abc",
        "max_new_tokens": 5,
        "temperature": 1.0,
        "top_k": 3,
        "top_p": 0.9,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prompt" in data
    assert data["prompt"] == "abc"
    assert "generated_text" in data
    assert isinstance(data["generated_text"], str)
    assert len(data["generated_text"]) > len("abc")


def test_generate_endpoint_unloaded(monkeypatch):
    monkeypatch.setattr(serve_module, "MODEL", None)
    monkeypatch.setattr(serve_module, "TOKENIZER", None)
    client = TestClient(serve_module.app)
    payload = {"prompt": "abc"}
    response = client.post("/generate", json=payload)
    assert response.status_code == 533 or response.status_code == 503
    assert response.status_code == 503


def test_generate_endpoint_invalid_input(dummy_checkpoint):
    serve_module.load_model_global(dummy_checkpoint)
    client = TestClient(serve_module.app)

    # Missing required field 'prompt'
    res1 = client.post("/generate", json={"max_new_tokens": 5})
    assert res1.status_code == 422

    # Negative max_new_tokens
    res2 = client.post("/generate", json={"prompt": "abc", "max_new_tokens": -5})
    assert res2.status_code == 422

    # Zero max_new_tokens
    res3 = client.post("/generate", json={"prompt": "abc", "max_new_tokens": 0})
    assert res3.status_code == 422

    # max_new_tokens exceeding capped limit (e.g., > 1000)
    res4 = client.post("/generate", json={"prompt": "abc", "max_new_tokens": 1000000})
    assert res4.status_code == 422

    # Negative temperature
    res5 = client.post("/generate", json={"prompt": "abc", "temperature": -1.0})
    assert res5.status_code == 422

    # Invalid top_p (> 1.0)
    res6 = client.post("/generate", json={"prompt": "abc", "top_p": 1.5})
    assert res6.status_code == 422


def test_generate_endpoint_unknown_char(dummy_checkpoint):
    serve_module.load_model_global(dummy_checkpoint)
    client = TestClient(serve_module.app)
    # Tokenizer only knows 'a'-'j', 'z' is unknown -> 400 Bad Request
    response = client.post("/generate", json={"prompt": "xyz"})
    assert response.status_code == 400
