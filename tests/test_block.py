import torch
import pytest
from nanogpt.block import Block

def test_block_forward_shape():
    batch, seq_len, n_embd = 2, 8, 32
    n_head, block_size = 4, 16

    block = Block(n_embd=n_embd, n_head=n_head, block_size=block_size)
    x = torch.randn(batch, seq_len, n_embd)

    out = block(x)
    assert out.shape == x.shape == (batch, seq_len, n_embd)

def test_block_transforms_representation():
    batch, seq_len, n_embd = 2, 8, 32
    n_head, block_size = 4, 16

    block = Block(n_embd=n_embd, n_head=n_head, block_size=block_size)
    x = torch.randn(batch, seq_len, n_embd)

    out = block(x)

    # Check that output is not identical to input (block is not a dead layer / no-op)
    assert not torch.allclose(out, x), "Block output should not be identical to input"
