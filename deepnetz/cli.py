"""
DeepNetz CLI — One command to rule them all.

Usage:
    deepnetz run Qwen3.5-122B --gpu 8GB
    deepnetz info model.gguf
    deepnetz serve model.gguf --port 8080
    deepnetz hardware
"""

import argparse
import sys
from deepnetz.engine.hardware import detect_hardware, print_hardware
from deepnetz.engine.model import Model


def main():
    parser = argparse.ArgumentParser(
        prog="deepnetz",
        description="DeepNetz — Run massive models on minimal hardware. https://deepnetz.com"
    )
    subparsers = parser.add_subparsers(dest="command")

    # hardware
    sub_hw = subparsers.add_parser("hardware", help="Show hardware profile")

    # run
    sub_run = subparsers.add_parser("run", help="Run a model")
    sub_run.add_argument("model", help="Model path or HuggingFace ID")
    sub_run.add_argument("--gpu", default="auto", help="GPU memory budget (e.g., 8GB)")
    sub_run.add_argument("--ram", default="auto", help="RAM budget (e.g., 32GB)")
    sub_run.add_argument("--context", type=int, default=4096, help="Target context length")

    # info
    sub_info = subparsers.add_parser("info", help="Show model info + plan")
    sub_info.add_argument("model", help="Model path (GGUF file)")
    sub_info.add_argument("--gpu", default="auto")
    sub_info.add_argument("--ram", default="auto")
    sub_info.add_argument("--context", type=int, default=4096)

    args = parser.parse_args()

    if args.command == "hardware":
        hw = detect_hardware()
        print_hardware(hw)

    elif args.command in ("run", "info"):
        model = Model(
            args.model,
            gpu_budget=args.gpu,
            ram_budget=args.ram,
            target_context=args.context
        )
        if args.command == "run":
            print("  Inference engine coming in v0.2. Use 'deepnetz info' to see the plan.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
