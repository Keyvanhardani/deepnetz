"""
DeepNetz CLI — One command to rule them all.

Usage:
    deepnetz run model.gguf                          # auto-detect hardware
    deepnetz run model.gguf --gpu 8GB                # limit GPU budget
    deepnetz run model.gguf --cpu                    # force CPU-only
    deepnetz run model.gguf --context 32k            # set context length

    deepnetz serve model.gguf --port 8080            # OpenAI-compatible API
    deepnetz info model.gguf                         # show model + plan
    deepnetz hardware                                # show hardware profile
    deepnetz download Qwen3.5-35B-A3B --quant Q4_K_M # download from HuggingFace
"""

import argparse
import sys


def cmd_hardware(args):
    from deepnetz.engine.hardware import detect_hardware, print_hardware
    hw = detect_hardware()
    print_hardware(hw)


def cmd_info(args):
    from deepnetz.engine.model import Model
    model = Model(
        args.model,
        gpu_budget="0" if args.cpu else args.gpu,
        ram_budget=args.ram,
        target_context=_parse_context(args.context),
        cpu_only=args.cpu,
    )


def cmd_run(args):
    from deepnetz.engine.model import Model

    model = Model(
        args.model,
        gpu_budget="0" if args.cpu else args.gpu,
        ram_budget=args.ram,
        target_context=_parse_context(args.context),
        cpu_only=args.cpu,
    )

    model.load()

    if args.prompt:
        # Single prompt mode
        if args.stream:
            for token in model.stream(args.prompt, max_tokens=args.max_tokens):
                print(token, end="", flush=True)
            print()
        else:
            response = model.chat(args.prompt, max_tokens=args.max_tokens)
            print(response)
    else:
        # Interactive chat mode
        print("  DeepNetz Chat (type 'exit' to quit)\n")
        while True:
            try:
                user_input = input("  You: ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if not user_input:
                    continue

                print("  AI:  ", end="", flush=True)
                for token in model.stream(user_input, max_tokens=args.max_tokens):
                    print(token, end="", flush=True)
                print("\n")

            except (KeyboardInterrupt, EOFError):
                print("\n  Bye!")
                break


def cmd_serve(args):
    from deepnetz.server import create_app

    app = create_app(
        model_path=args.model,
        gpu_budget="0" if args.cpu else args.gpu,
        ram_budget=args.ram,
        target_context=_parse_context(args.context),
        cpu_only=args.cpu,
    )

    import uvicorn
    print(f"\n  DeepNetz API Server")
    print(f"  http://localhost:{args.port}/v1/chat/completions")
    print(f"  http://localhost:{args.port}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_download(args):
    from deepnetz.engine.downloader import download_model
    download_model(args.model, quant=args.quant, output_dir=args.output)


def _parse_context(ctx_str: str) -> int:
    ctx_str = ctx_str.strip().lower()
    if ctx_str.endswith("k"):
        return int(float(ctx_str[:-1]) * 1024)
    elif ctx_str.endswith("m"):
        return int(float(ctx_str[:-1]) * 1024 * 1024)
    return int(ctx_str)


def main():
    parser = argparse.ArgumentParser(
        prog="deepnetz",
        description="DeepNetz — Run massive models on minimal hardware.\nhttps://deepnetz.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # hardware
    subparsers.add_parser("hardware", help="Show hardware profile")

    # info
    p_info = subparsers.add_parser("info", help="Show model info + inference plan")
    p_info.add_argument("model", help="Path to GGUF model file")
    p_info.add_argument("--gpu", default="auto", help="GPU budget (e.g., 8GB, 0 for CPU-only)")
    p_info.add_argument("--ram", default="auto", help="RAM budget")
    p_info.add_argument("--context", default="4096", help="Context length (e.g., 4096, 32k)")
    p_info.add_argument("--cpu", action="store_true", help="Force CPU-only mode")

    # run
    p_run = subparsers.add_parser("run", help="Run a model (interactive or single prompt)")
    p_run.add_argument("model", help="Path to GGUF model file")
    p_run.add_argument("-p", "--prompt", help="Single prompt (interactive if omitted)")
    p_run.add_argument("--gpu", default="auto", help="GPU budget (e.g., 8GB)")
    p_run.add_argument("--ram", default="auto", help="RAM budget")
    p_run.add_argument("--context", default="4096", help="Context length (e.g., 4096, 32k)")
    p_run.add_argument("--cpu", action="store_true", help="Force CPU-only mode")
    p_run.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    p_run.add_argument("--stream", action="store_true", default=True, help="Stream output")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start OpenAI-compatible API server")
    p_serve.add_argument("model", help="Path to GGUF model file")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--gpu", default="auto")
    p_serve.add_argument("--ram", default="auto")
    p_serve.add_argument("--context", default="4096")
    p_serve.add_argument("--cpu", action="store_true")

    # download
    p_dl = subparsers.add_parser("download", help="Download model from HuggingFace")
    p_dl.add_argument("model", help="Model name (e.g., Qwen3.5-35B-A3B)")
    p_dl.add_argument("--quant", default="Q4_K_M", help="Quantization type")
    p_dl.add_argument("--output", default=".", help="Output directory")

    args = parser.parse_args()

    commands = {
        "hardware": cmd_hardware,
        "info": cmd_info,
        "run": cmd_run,
        "serve": cmd_serve,
        "download": cmd_download,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
