import os
import math
import torch
import pytest
import yaml
from nanogpt.data import get_batch, prepare_data
from nanogpt.train import train, estimate_loss
from nanogpt.model import GPT

def test_get_batch_shapes():
    data = torch.arange(100, dtype=torch.long)
    block_size = 8
    batch_size = 4
    x, y = get_batch(data, block_size=block_size, batch_size=batch_size)
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)

def test_train_val_split_non_overlapping(tmp_path):
    text_content = "Hello world! This is a test dataset for tiny shakespeare split verification."
    filepath = str(tmp_path / "test_corpus.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text_content)

    train_data, val_data, tokenizer = prepare_data(filepath, train_ratio=0.9)
    full_len = len(train_data) + len(val_data)
    assert full_len == len(tokenizer.encode(text_content))
    assert len(train_data) == int(0.9 * full_len)
    assert len(val_data) == full_len - len(train_data)

    # Check non-overlap by checking slice lengths and positions
    # train_data is [:n] and val_data is [n:] from full tensor
    raw_data = torch.tensor(tokenizer.encode(text_content), dtype=torch.long)
    assert torch.equal(train_data, raw_data[: len(train_data)])
    assert torch.equal(val_data, raw_data[len(train_data) :])

def test_get_batch_offset_by_one():
    data = torch.arange(100, dtype=torch.long)
    block_size = 10
    batch_size = 5
    torch.manual_seed(42)
    x, y = get_batch(data, block_size=block_size, batch_size=batch_size)

    for i in range(batch_size):
        start_idx = x[i, 0].item()
        assert torch.equal(x[i], data[start_idx : start_idx + block_size])
        assert torch.equal(y[i], data[start_idx + 1 : start_idx + 1 + block_size])
        assert torch.equal(x[i, 1:], y[i, :-1])

def test_training_smoke_test(tmp_path):
    corpus = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 5
    data_file = str(tmp_path / "tiny_data.txt")
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(corpus)

    config_data = {
        "data": {
            "path": data_file,
            "train_ratio": 0.9,
        },
        "model": {
            "n_embd": 16,
            "n_layer": 2,
            "n_head": 2,
            "block_size": 8,
            "dropout": 0.0,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 1e-3,
            "max_steps": 10,
            "eval_interval": 5,
            "eval_iters": 2,
            "seed": 42,
            "device": "cpu",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "checkpoint_interval": 10,
        },
    }
    config_file = str(tmp_path / "smoke_config.yaml")
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    train(config_file)

    # Assert checkpoints were created and valid
    ckpt_path = os.path.join(tmp_path / "checkpoints", "ckpt_step_10.pt")
    assert os.path.exists(ckpt_path)
    checkpoint = torch.load(ckpt_path, weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["step"] == 10

    # Verify model forward pass / loss computation produces finite loss
    train_data, val_data, tokenizer = prepare_data(data_file)
    model = GPT(vocab_size=tokenizer.vocab_size, n_embd=16, n_layer=2, n_head=2, block_size=8)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss = estimate_loss(model, val_data, block_size=8, batch_size=2, eval_iters=2, device="cpu")
    assert math.isfinite(val_loss)
