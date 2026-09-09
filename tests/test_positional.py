import torch
import pytest
from nanogpt.positional import PositionalEmbedding

def test_positional_embedding_shape():
    batch, seq_len, n_embd = 4, 16, 32
    block_size = 64
    pos_emb_module = PositionalEmbedding(block_size=block_size, n_embd=n_embd)

    # Passing token embedding tensor of shape (batch, seq_len, n_embd)
    token_emb = torch.randn(batch, seq_len, n_embd)
    pos_emb = pos_emb_module(token_emb)

    assert pos_emb.shape == token_emb.shape == (batch, seq_len, n_embd)

def test_distinct_positions_produce_distinct_vectors():
    block_size, n_embd = 64, 32
    pos_emb_module = PositionalEmbedding(block_size=block_size, n_embd=n_embd)

    seq_len = 10
    token_emb = torch.randn(1, seq_len, n_embd)
    pos_emb = pos_emb_module(token_emb) # shape (1, seq_len, n_embd)

    vectors = pos_emb[0] # shape (seq_len, n_embd)

    # Check that for any pair of positions i and j, vectors[i] and vectors[j] are not identical
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert not torch.allclose(vectors[i], vectors[j]), f"Position {i} and {j} produced identical vectors"

def test_position_embedding_pure_function_of_index():
    batch, seq_len, n_embd = 3, 8, 16
    block_size = 32
    pos_emb_module = PositionalEmbedding(block_size=block_size, n_embd=n_embd)

    # Batch with two different inputs
    input1 = torch.randn(batch, seq_len, n_embd)
    input2 = torch.randn(batch, seq_len, n_embd) * 10.0 + 5.0

    pos_emb1 = pos_emb_module(input1)
    pos_emb2 = pos_emb_module(input2)

    # Position embeddings must be identical regardless of input tensor values
    assert torch.allclose(pos_emb1, pos_emb2)

    # Position embeddings for position i must be identical across all batch elements
    for b in range(batch):
        assert torch.allclose(pos_emb1[0], pos_emb1[b])
