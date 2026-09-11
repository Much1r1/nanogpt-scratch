import sys
import argparse
from nanogpt.train import train
from nanogpt.generate import generate_cli

def main():
    parser = argparse.ArgumentParser(prog="nanogpt", description="NanoGPT Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train NanoGPT model")
    train_parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    generate_parser = subparsers.add_parser("generate", help="Generate text from NanoGPT model")
    generate_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    generate_parser.add_argument("--prompt", type=str, default="", help="Prompt string for generation")
    generate_parser.add_argument("--max-new-tokens", type=int, default=100, help="Number of tokens to generate")
    generate_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    generate_parser.add_argument("--top-k", type=int, default=None, help="Top-k filtering limit")
    generate_parser.add_argument("--top-p", type=float, default=None, help="Top-p nucleus filtering threshold")
    generate_parser.add_argument("--config", type=str, default=None, help="Path to config YAML file (optional)")
    generate_parser.add_argument("--device", type=str, default="cpu", help="Device to run generation on")

    args = parser.parse_args()

    if args.command == "train":
        train(args.config)
    elif args.command == "generate":
        # Pass remaining arguments to generate_cli logic
        gen_args = [
            "--checkpoint", args.checkpoint,
            "--prompt", args.prompt,
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--device", args.device,
        ]
        if args.top_k is not None:
            gen_args.extend(["--top-k", str(args.top_k)])
        if args.top_p is not None:
            gen_args.extend(["--top-p", str(args.top_p)])
        if args.config is not None:
            gen_args.extend(["--config", args.config])

        generate_cli(gen_args)

if __name__ == "__main__":
    main()
