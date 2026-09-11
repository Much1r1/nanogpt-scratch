import sys
import argparse
from nanogpt.train import train

def main():
    parser = argparse.ArgumentParser(prog="nanogpt", description="NanoGPT Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train NanoGPT model")
    train_parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    args = parser.parse_args()

    if args.command == "train":
        train(args.config)

if __name__ == "__main__":
    main()
