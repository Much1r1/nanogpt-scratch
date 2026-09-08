import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.embedding = nn.Embedding(block_size, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, n_embd) or token indices of shape (batch, seq_len)
               or sequence length as integer or sequence tensor.
        Returns:
            Position embedding tensor of shape (batch, seq_len, n_embd) or (1, seq_len, n_embd) broadcastable.
        """
        if isinstance(x, torch.Tensor):
            if x.dim() == 3:
                batch_size, seq_len, _ = x.shape
            elif x.dim() == 2:
                batch_size, seq_len = x.shape
            elif x.dim() == 1:
                batch_size = 1
                seq_len = x.shape[0]
            else:
                raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
        elif isinstance(x, int):
            batch_size = 1
            seq_len = x
        else:
            raise TypeError(f"Unsupported type for x: {type(x)}")

        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.block_size}")

        device = x.device if isinstance(x, torch.Tensor) else None
        pos = torch.arange(seq_len, device=device)
        pos_emb = self.embedding(pos) # shape (seq_len, n_embd)

        # Expand to (batch, seq_len, n_embd) or keep as (1, seq_len, n_embd) broadcastable
        if batch_size > 1:
            pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            pos_emb = pos_emb.unsqueeze(0)

        return pos_emb
