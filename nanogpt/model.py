import torch
import torch.nn as nn
import torch.nn.functional as F
from nanogpt.positional import PositionalEmbedding
from nanogpt.block import Block

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = PositionalEmbedding(block_size=block_size, n_embd=n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd=n_embd, n_head=n_head, block_size=block_size, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight-tie token embedding and lm_head
        self.lm_head.weight = self.wte.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx: LongTensor of shape (batch, seq_len) with token indices
            targets: LongTensor of shape (batch, seq_len) with target token indices (optional)
        Returns:
            logits: FloatTensor of shape (batch, seq_len, vocab_size)
            loss: Optional loss scalar if targets is provided
        """
        device = idx.device
        b, t = idx.size()
        if t > self.block_size:
            raise ValueError(f"Cannot forward sequence of length {t}, block size is {self.block_size}")

        tok_emb = self.wte(idx) # (b, t, n_embd)
        pos_emb = self.wpe(idx) # (b, t, n_embd) or broadcastable (1, t, n_embd)

        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (b, t, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if targets is not None:
            return logits, loss
        return logits
