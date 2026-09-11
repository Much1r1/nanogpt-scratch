import os
import urllib.request
from typing import Tuple, Optional, Union
import torch
from nanogpt.tokenizer import CharTokenizer

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def download_data(filepath: str, url: str = TINY_SHAKESPEARE_URL) -> None:
    dirname = os.path.dirname(filepath)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath)

def load_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def prepare_data(filepath: str, train_ratio: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    if not os.path.exists(filepath):
        download_data(filepath)
    text = load_text(filepath)
    tokenizer = CharTokenizer.from_corpus(text)
    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded, dtype=torch.long)
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, tokenizer

def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size:
        raise ValueError(f"Data length ({len(data)}) must be greater than block_size ({block_size}).")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    return x, y
