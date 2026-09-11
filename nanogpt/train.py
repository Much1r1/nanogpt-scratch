import os
import argparse
import yaml
import torch
from nanogpt.utils import set_seed
from nanogpt.data import prepare_data, get_batch
from nanogpt.model import GPT

@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters, device):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, block_size, batch_size, device=device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

def save_checkpoint(model, optimizer, step, checkpoint_dir, config=None, chars=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step_{step}.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config,
        "chars": chars,
    }
    torch.save(checkpoint, ckpt_path)
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)

def train(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    seed = train_cfg.get("seed", 1337)
    set_seed(seed)

    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    data_path = data_cfg.get("path", "data/tiny_shakespeare.txt")
    train_ratio = data_cfg.get("train_ratio", 0.9)
    train_data, val_data, tokenizer = prepare_data(data_path, train_ratio=train_ratio)

    vocab_size = tokenizer.vocab_size
    n_embd = model_cfg.get("n_embd", 64)
    n_layer = model_cfg.get("n_layer", 4)
    n_head = model_cfg.get("n_head", 4)
    block_size = model_cfg.get("block_size", 64)
    dropout = model_cfg.get("dropout", 0.0)

    # Put vocab_size explicitly into model config if not present
    model_cfg_full = dict(model_cfg)
    model_cfg_full["vocab_size"] = vocab_size

    model = GPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        dropout=dropout,
    ).to(device)

    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    max_steps = train_cfg.get("max_steps", 500)
    batch_size = train_cfg.get("batch_size", 32)
    eval_interval = train_cfg.get("eval_interval", 50)
    eval_iters = train_cfg.get("eval_iters", 10)
    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    checkpoint_interval = train_cfg.get("checkpoint_interval", 100)

    full_config = {
        "data": data_cfg,
        "model": model_cfg_full,
        "training": train_cfg,
    }

    model.train()
    for step in range(1, max_steps + 1):
        xb, yb = get_batch(train_data, block_size, batch_size, device=device)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == max_steps:
            train_loss = loss.item()
            val_loss = estimate_loss(model, val_data, block_size, batch_size, eval_iters, device)
            print(f"step {step:5d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if step % checkpoint_interval == 0 or step == max_steps:
            save_checkpoint(model, optimizer, step, checkpoint_dir, config=full_config, chars=tokenizer.chars)

def main():
    parser = argparse.ArgumentParser(description="Train NanoGPT model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)

if __name__ == "__main__":
    main()
