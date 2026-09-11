import argparse
from typing import Optional, Union, List
import yaml
import torch
import torch.nn.functional as F

from nanogpt.model import GPT
from nanogpt.tokenizer import CharTokenizer
from nanogpt.data import prepare_data


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Sample next token index from logits given sampling controls.

    Args:
        logits: Tensor of shape (vocab_size,) or (batch_size, vocab_size)
        temperature: Temperature scaling factor. If <= 1e-8, uses argmax (greedy).
        top_k: If set, keep top k logits, mask rest to -inf.
        top_p: If set, keep smallest set of tokens whose cumulative probability >= p.

    Returns:
        Sampled token index tensor of shape () or (batch_size, 1)
    """
    is_1d = logits.dim() == 1
    if is_1d:
        logits = logits.unsqueeze(0)  # (1, vocab_size)

    logits = logits.clone()

    if temperature is None or temperature <= 1e-8:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        logits = logits / temperature

        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            # Get values of top k
            v, _ = torch.topk(logits, k, dim=-1)
            min_k_value = v[:, [-1]]
            logits[logits < min_k_value] = float("-inf")

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)

            # Mask tokens with cumulative probability above p
            # Shift cumulative probabilities to the right so the first token that exceeds top_p is kept
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Scatter back to original indices mask
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    if is_1d:
        return next_token.squeeze(0)  # ()
    return next_token  # (batch_size, 1)


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Autoregressively generate new tokens starting from context idx.

    Args:
        model: Trained GPT model
        idx: LongTensor of shape (batch_size, seq_len) or (seq_len,) with starting token ids
        max_new_tokens: Number of tokens to generate
        temperature: Temperature scaling factor
        top_k: Top-k filtering threshold
        top_p: Top-p (nucleus) filtering threshold

    Returns:
        LongTensor of shape (batch_size, seq_len + max_new_tokens) or (seq_len + max_new_tokens,)
    """
    model.eval()
    was_1d = idx.dim() == 1
    if was_1d:
        idx = idx.unsqueeze(0)  # (1, seq_len)

    for _ in range(max_new_tokens):
        # Crop context if it exceeds block_size
        idx_cond = idx[:, -model.block_size :]
        logits = model(idx_cond)  # (batch_size, t, vocab_size)
        # Take logits at last position
        logits_last = logits[:, -1, :]  # (batch_size, vocab_size)

        next_token = sample_next_token(
            logits_last,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )  # (batch_size, 1)

        idx = torch.cat((idx, next_token), dim=1)

    if was_1d:
        return idx.squeeze(0)
    return idx


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "cpu",
):
    """
    Loads checkpoint, rebuilds GPT model and tokenizer.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get("config", None)
    chars = checkpoint.get("chars", None)

    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            override_config = yaml.safe_load(f)
            if config is None:
                config = override_config
            else:
                config.update(override_config)

    if config is None:
        raise ValueError(
            "Config not found in checkpoint and no config_path provided."
        )

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    # Rebuild tokenizer
    if chars is not None:
        tokenizer = CharTokenizer(chars)
    else:
        data_path = data_cfg.get("path", "data/tiny_shakespeare.txt")
        train_ratio = data_cfg.get("train_ratio", 0.9)
        _, _, tokenizer = prepare_data(data_path, train_ratio=train_ratio)

    vocab_size = model_cfg.get("vocab_size", tokenizer.vocab_size)
    n_embd = model_cfg.get("n_embd", 64)
    n_layer = model_cfg.get("n_layer", 4)
    n_head = model_cfg.get("n_head", 4)
    block_size = model_cfg.get("block_size", 64)
    dropout = model_cfg.get("dropout", 0.0)

    model = GPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer


def generate_cli(args=None):
    parser = argparse.ArgumentParser(description="Generate text from a trained NanoGPT model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Prompt string for generation")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k filtering limit")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p nucleus filtering threshold")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file (optional)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run generation on")

    parsed_args = parser.parse_args(args)

    device = parsed_args.device
    model, tokenizer = load_model_from_checkpoint(
        parsed_args.checkpoint,
        config_path=parsed_args.config,
        device=device,
    )

    if parsed_args.prompt:
        encoded_prompt = tokenizer.encode(parsed_args.prompt)
    else:
        # Default to token 0 or newline if empty prompt
        encoded_prompt = [0]

    prompt_tensor = torch.tensor(encoded_prompt, dtype=torch.long, device=device)

    out_tokens = generate(
        model,
        prompt_tensor,
        max_new_tokens=parsed_args.max_new_tokens,
        temperature=parsed_args.temperature,
        top_k=parsed_args.top_k,
        top_p=parsed_args.top_p,
    )

    generated_text = tokenizer.decode(out_tokens.tolist())
    print(generated_text)
    return generated_text


if __name__ == "__main__":
    generate_cli()
