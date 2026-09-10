import torch
from nanogpt.model import GPT

def test_gpt_forward_shape():
    batch_size = 2
    seq_len = 16
    vocab_size = 65
    n_embd = 32
    n_layer = 2
    n_head = 4
    block_size = 64

    model = GPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
    )

    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(idx)

    assert logits.shape == (batch_size, seq_len, vocab_size)

    # Test weight tying
    assert model.lm_head.weight is model.wte.weight


def test_gpt_golden_overfit():
    torch.manual_seed(42)

    # Short fixed sequence representing ~25 tokens (e.g. from tiny_shakespeare)
    # "First Citizen:\nBefore we proceed any further, hear me speak."
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}

    token_ids = [stoi[ch] for ch in text]
    # Use sequence length of 25 tokens
    token_ids = token_ids[:25]
    seq_len = len(token_ids)

    # Input sequence (tokens 0..N-2) and Target sequence (tokens 1..N-1)
    # Or input sequence x and target sequence y
    x = torch.tensor([token_ids[:-1]], dtype=torch.long) # shape (1, 24)
    y = torch.tensor([token_ids[1:]], dtype=torch.long)  # shape (1, 24)

    model = GPT(
        vocab_size=vocab_size,
        n_embd=64,
        n_layer=2,
        n_head=4,
        block_size=32,
        dropout=0.0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(300):
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1, f"Expected loss < 0.1, got {loss.item():.4f}"
