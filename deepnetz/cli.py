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


def cmd_backends(args):
    from deepnetz.backends.discovery import discover_backends, print_backends
    backends = discover_backends()
    print_backends(backends)
    for b in backends:
        models = b.list_models()
        if models:
            print(f"  {b.name} models:")
            for m in models[:10]:
                size = f"{m.size_mb}MB" if m.size_mb else ""
                print(f"    {m.name} {size}")
            if len(models) > 10:
                print(f"    ... +{len(models)-10} more")
            print()


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


def cmd_register(args):
    from deepnetz.registry.client import RegistryClient
    import getpass
    client = RegistryClient()

    username = args.username or input("  Username: ").strip()
    password = getpass.getpass("  Passwort: ")

    try:
        result = client.register(username, password)
        print(f"\n  Registriert als: {result.get('username')}")
        print(f"  API-Key: {result.get('api_key')}")
        print(f"  Gespeichert in: ~/.config/deepnetz/credentials.json\n")
    except RuntimeError as e:
        print(f"\n  Fehler: {e}\n")


def cmd_login(args):
    from deepnetz.registry.client import RegistryClient
    import getpass
    client = RegistryClient()

    username = args.username or input("  Username: ").strip()
    password = getpass.getpass("  Passwort: ")

    try:
        result = client.login(username, password)
        print(f"\n  Eingeloggt als: {result.get('username')}")
        print(f"  API-Key gespeichert.\n")
    except RuntimeError as e:
        print(f"\n  Fehler: {e}\n")


def cmd_search(args):
    from deepnetz.registry.client import RegistryClient
    client = RegistryClient()

    if not client.is_authenticated:
        print(f"\n  Bitte zuerst einloggen: deepnetz login\n")
        return

    try:
        results = client.search(args.query, limit=args.limit)
    except RuntimeError as e:
        print(f"\n  Fehler: {e}\n")
        return

    if not results:
        print(f"\n  Keine Ergebnisse für '{args.query}'.\n")
        return

    print(f"\n  Suchergebnisse für '{args.query}' ({len(results)} Treffer)")
    print(f"  {'─' * 60}")
    for m in results:
        repo = m.get("repo", "")
        dl = m.get("downloads", 0)
        dl_str = f"{dl // 1000}k" if dl > 1000 else str(dl)
        print(f"  {repo:<45} {dl_str:>8} Downloads")
    print(f"\n  Pull: deepnetz pull <repo>\n")


def cmd_pull(args):
    from deepnetz.engine.downloader import pull_model
    from deepnetz.registry.client import RegistryClient

    # Log pull to registry
    try:
        client = RegistryClient()
        if client.is_authenticated:
            client.log_pull(args.model, quant=args.quant)
    except Exception:
        pass

    pull_model(args.model, quant=args.quant)


def cmd_list(args):
    from deepnetz.engine.downloader import list_local_models

    models = list_local_models()
    if not models:
        print(f"\n  Keine lokalen Modelle.")
        print(f"  Suche:  deepnetz search Qwen")
        print(f"  Pull:   deepnetz pull Qwen3.5-35B\n")
        return

    print(f"\n  DeepNetz Modelle ({len(models)} lokal)")
    print(f"  {'─' * 55}")
    for m in models:
        name = m.get("name", "?")
        quant = m.get("quant", "?")
        size = m.get("size", 0)
        size_mb = size / (1024 * 1024) if size > 1024 * 1024 else size
        size_str = f"{size_mb / 1024:.1f} GB" if size_mb > 1024 else f"{size_mb:.0f} MB"
        status = "✓" if m.get("available") else "✗"
        print(f"  {status} {name:<22} {quant:<10} {size_str}")
    print(f"\n  Run: deepnetz run <name>\n")


def cmd_registry(args):
    from deepnetz.registry.server import create_registry_app
    import uvicorn

    app = create_registry_app()
    print(f"\n  DeepNetz Registry Server")
    print(f"  http://0.0.0.0:{args.port}/v1/search?q=Qwen")
    print(f"  http://0.0.0.0:{args.port}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def cmd_download(args):
    from deepnetz.engine.downloader import pull_model
    pull_model(args.model, quant=args.quant)


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

    # backends
    subparsers.add_parser("backends", help="Show available inference backends")

    # info
    p_info = subparsers.add_parser("info", help="Show model info + inference plan")
    p_info.add_argument("model", help="Path to GGUF model file")
    p_info.add_argument("--gpu", default="auto", help="GPU budget (e.g., 8GB, 0 for CPU-only)")
    p_info.add_argument("--ram", default="auto", help="RAM budget")
    p_info.add_argument("--context", default="4096", help="Context length (e.g., 4096, 32k)")
    p_info.add_argument("--backend", default="auto", help="Force backend (native, ollama, vllm, lmstudio, hf, remote)")
    p_info.add_argument("--cpu", action="store_true", help="Force CPU-only mode")

    # run
    p_run = subparsers.add_parser("run", help="Run a model (interactive or single prompt)")
    p_run.add_argument("model", help="Path to GGUF model file")
    p_run.add_argument("-p", "--prompt", help="Single prompt (interactive if omitted)")
    p_run.add_argument("--gpu", default="auto", help="GPU budget (e.g., 8GB)")
    p_run.add_argument("--ram", default="auto", help="RAM budget")
    p_run.add_argument("--context", default="4096", help="Context length (e.g., 4096, 32k)")
    p_run.add_argument("--backend", default="auto", help="Force backend")
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
    p_serve.add_argument("--backend", default="auto", help="Force backend")
    p_serve.add_argument("--cpu", action="store_true")

    # register
    p_reg_user = subparsers.add_parser("register", help="Account erstellen auf registry.deepnetz.com")
    p_reg_user.add_argument("--username", default="", help="Username")

    # login
    p_login = subparsers.add_parser("login", help="Einloggen bei registry.deepnetz.com")
    p_login.add_argument("--username", default="", help="Username")

    # search
    p_search = subparsers.add_parser("search", help="Modelle suchen (über Registry)")
    p_search.add_argument("query", help="Suchbegriff (z.B. Qwen, Llama, code)")
    p_search.add_argument("--limit", type=int, default=15, help="Max Ergebnisse")

    # pull
    p_pull = subparsers.add_parser("pull", help="Modell herunterladen")
    p_pull.add_argument("model", help="Modellname oder HF Repo (z.B. Qwen3.5-35B)")
    p_pull.add_argument("--quant", default="auto", help="Quantisierung (auto, Q4_K_M, Q8_0, ...)")

    # list
    p_list = subparsers.add_parser("list", help="Lokale Modelle anzeigen")

    # registry (server)
    p_reg = subparsers.add_parser("registry", help="Registry Server starten")
    p_reg.add_argument("--port", type=int, default=8090, help="Port (Standard: 8090)")

    # download (legacy)
    p_dl = subparsers.add_parser("download", help="Modell herunterladen (Alias für pull)")
    p_dl.add_argument("model", help="Modellname")
    p_dl.add_argument("--quant", default="auto", help="Quantisierung")

    args = parser.parse_args()

    commands = {
        "hardware": cmd_hardware,
        "backends": cmd_backends,
        "info": cmd_info,
        "run": cmd_run,
        "serve": cmd_serve,
        "register": cmd_register,
        "login": cmd_login,
        "search": cmd_search,
        "pull": cmd_pull,
        "list": cmd_list,
        "registry": cmd_registry,
        "download": cmd_download,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
