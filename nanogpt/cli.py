import sys
import os
import argparse
import uvicorn
from nanogpt.train import train
from nanogpt.generate import generate_cli

def serve_cli(args):
    if args.checkpoint:
        os.environ["CHECKPOINT_PATH"] = args.checkpoint
    uvicorn.run("nanogpt.serve:app", host=args.host, port=args.port, reload=args.reload)

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

    serve_parser = subparsers.add_parser("serve", help="Start FastAPI HTTP server for NanoGPT model")
    serve_parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (sets CHECKPOINT_PATH env var)")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")

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
    elif args.command == "serve":
        serve_cli(args)

if __name__ == "__main__":
    main()
