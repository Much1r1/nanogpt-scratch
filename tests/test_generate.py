import os
import pytest
import torch
import torch.nn.functional as F

from nanogpt.model import GPT
from nanogpt.tokenizer import CharTokenizer
from nanogpt.generate import sample_next_token, generate, load_model_from_checkpoint
from nanogpt.train import save_checkpoint


def test_temperature_zero_or_low_greedy():
    # Fake logits where token 3 has highest value
    logits = torch.tensor([1.0, 2.0, 0.5, 5.0, -1.0])

    # Temperature = 0 should pick index 3 (argmax)
    token_temp0 = sample_next_token(logits, temperature=0.0)
    assert token_temp0.item() == 3

    # Low temperature should also pick index 3 with high probability (repeated samples deterministic)
    samples = [sample_next_token(logits, temperature=1e-5).item() for _ in range(20)]
    assert all(s == 3 for s in samples)


def test_top_k_restricts_sampling():
    # Logits for 5 tokens: [1.0, 5.0, 4.0, 2.0, 0.0]
    # top_k = 2 should keep indices 1 (val 5.0) and 2 (val 4.0), and mask indices 0, 3, 4 to -inf
    logits = torch.tensor([1.0, 5.0, 4.0, 2.0, 0.0])
    k = 2

    samples = set()
    for _ in range(100):
        tok = sample_next_token(logits, temperature=1.0, top_k=k)
        samples.add(tok.item())

    # Verify sampled tokens are ONLY from top k set {1, 2}
    assert samples == {1, 2} or samples.issubset({1, 2})
    assert 0 not in samples and 3 not in samples and 4 not in samples


def test_top_p_restricts_sampling_nucleus():
    # Fake logits where probabilities before top_p filtering can be computed explicitly
    # Logits = [10.0, 9.0, 1.0, 0.0, -10.0]
    # Softmax probabilities: index 0 and 1 take > 99% of probability mass
    logits = torch.tensor([10.0, 9.0, 1.0, 0.0, -10.0])
    probs = F.softmax(logits, dim=-1)

    # Let's set top_p = 0.9. Token 0 has prob ~ 0.73, Token 1 has prob ~ 0.27. Cum prob for {0, 1} is ~ 1.0.
    # Excluded tokens (2, 3, 4) should have zero probability mass after top_p filtering.
    samples = set()
    for _ in range(100):
        tok = sample_next_token(logits, temperature=1.0, top_p=0.9)
        samples.add(tok.item())

    assert samples.issubset({0, 1})
    assert 2 not in samples and 3 not in samples and 4 not in samples


def test_generate_output_length():
    vocab_size = 10
    block_size = 8
    model = GPT(vocab_size=vocab_size, n_embd=16, n_layer=2, n_head=2, block_size=block_size)

    idx = torch.tensor([1, 2, 3], dtype=torch.long)
    max_new_tokens = 5
    out = generate(model, idx, max_new_tokens=max_new_tokens, temperature=1.0)

    assert out.shape == (len(idx) + max_new_tokens,)

    # Test batch input shape (B, T)
    idx_batch = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    out_batch = generate(model, idx_batch, max_new_tokens=max_new_tokens, temperature=1.0)
    assert out_batch.shape == (2, 3 + max_new_tokens)


def test_load_checkpoint_and_generate_end_to_end(tmp_path):
    chars = ["a", "b", "c", "d", "e"]
    tokenizer = CharTokenizer(chars)
    vocab_size = tokenizer.vocab_size

    config = {
        "model": {
            "vocab_size": vocab_size,
            "n_embd": 16,
            "n_layer": 2,
            "n_head": 2,
            "block_size": 8,
            "dropout": 0.0,
        }
    }

    model = GPT(**config["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_dir = tmp_path / "checkpoints"
    save_checkpoint(model, optimizer, step=1, checkpoint_dir=str(ckpt_dir), config=config, chars=chars)

    ckpt_path = os.path.join(str(ckpt_dir), "checkpoint_latest.pt")
    loaded_model, loaded_tokenizer = load_model_from_checkpoint(ckpt_path)

    prompt = "ab"
    prompt_ids = torch.tensor(loaded_tokenizer.encode(prompt), dtype=torch.long)
    out_ids = generate(loaded_model, prompt_ids, max_new_tokens=4, temperature=0.8, top_k=3, top_p=0.9)

    out_text = loaded_tokenizer.decode(out_ids.tolist())
    assert len(out_text) == len(prompt) + 4
    assert out_text.startswith(prompt)
