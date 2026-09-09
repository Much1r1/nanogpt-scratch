import torch
import pytest
from nanogpt.attention import CausalSelfAttention

def test_causal_self_attention_output_shape():
    batch, seq_len, n_embd = 2, 8, 32
    n_head, block_size = 4, 16

    attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head, block_size=block_size)
    x = torch.randn(batch, seq_len, n_embd)

    out = attn(x)
    assert out.shape == (batch, seq_len, n_embd)

def test_causal_mask_is_lower_triangular():
    n_embd, n_head, block_size = 32, 4, 16
    attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head, block_size=block_size)

    # Check buffer 'bias' shape (1, 1, block_size, block_size)
    bias = attn.bias.squeeze() # (block_size, block_size)
    assert bias.shape == (block_size, block_size)

    # Check that bias is lower triangular (1 for row >= col, 0 for row < col)
    expected_mask = torch.tril(torch.ones(block_size, block_size))
    assert torch.equal(bias, expected_mask)

def test_no_future_leakage():
    batch, seq_len, n_embd = 2, 8, 32
    n_head, block_size = 4, 16

    attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head, block_size=block_size)
    attn.eval() # ensure dropout is inactive

    x1 = torch.randn(batch, seq_len, n_embd)
    x2 = x1.clone()

    # Modify tokens at position > i (e.g., position i = 3, change positions 4, 5, 6, 7)
    i = 3
    x2[:, i + 1:, :] = torch.randn(batch, seq_len - (i + 1), n_embd)

    out1 = attn(x1)
    out2 = attn(x2)

    # Outputs at position <= i must be identical between x1 and x2
    assert torch.allclose(out1[:, :i + 1, :], out2[:, :i + 1, :], atol=1e-6)
