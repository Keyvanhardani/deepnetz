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

    # Reasoning mode
    reasoning = getattr(args, 'reasoning', False)
    if reasoning:
        from deepnetz.engine.features import format_reasoning_prompt

    # Image input
    image = getattr(args, 'image', None)

    if args.prompt:
        prompt = args.prompt
        if reasoning:
            prompt = format_reasoning_prompt(prompt, True)

        if image:
            # Vision mode
            from deepnetz.engine.features import prepare_vision_message
            msg = prepare_vision_message(prompt, image_paths=[image])
            model.conversation.append(msg)
            from deepnetz.backends.base import GenerationConfig
            config = GenerationConfig(max_tokens=args.max_tokens)
            response = model.backend.chat(model.conversation, config)
            if reasoning:
                from deepnetz.engine.features import parse_reasoning
                thinking, answer = parse_reasoning(response)
                if thinking:
                    print(f"  Thinking: {thinking[:200]}...\n")
                print(answer)
            else:
                print(response)
        elif args.stream:
            full = []
            for token in model.stream(prompt, max_tokens=args.max_tokens):
                full.append(token)
                print(token, end="", flush=True)
            print()
            if reasoning:
                from deepnetz.engine.features import parse_reasoning
                thinking, answer = parse_reasoning("".join(full))
                if thinking:
                    print(f"\n  [Reasoning: {thinking[:200]}...]")
        else:
            response = model.chat(prompt, max_tokens=args.max_tokens)
            if reasoning:
                from deepnetz.engine.features import parse_reasoning
                thinking, answer = parse_reasoning(response)
                if thinking:
                    print(f"  Thinking: {thinking[:200]}...\n")
                print(answer)
            else:
                print(response)
    else:
        # Interactive chat mode
        from deepnetz.engine.features import is_vision_model
        has_vision = is_vision_model(args.model)
        print(f"  DeepNetz Chat (type 'exit' to quit)")
        if has_vision:
            print(f"  Vision enabled — use /image <path> to send images")
        if reasoning:
            print(f"  Reasoning mode enabled")
        print()

        while True:
            try:
                user_input = input("  You: ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if not user_input:
                    continue

                # Vision command: /image path.jpg describe this
                if user_input.startswith("/image ") and has_vision:
                    parts = user_input[7:].split(" ", 1)
                    img_path = parts[0]
                    img_prompt = parts[1] if len(parts) > 1 else "Describe this image."
                    if reasoning:
                        img_prompt = format_reasoning_prompt(img_prompt, True)
                    from deepnetz.engine.features import prepare_vision_message
                    from deepnetz.backends.base import GenerationConfig
                    msg = prepare_vision_message(img_prompt, image_paths=[img_path])
                    model.conversation.append(msg)
                    config = GenerationConfig(max_tokens=args.max_tokens)
                    print("  AI:  ", end="", flush=True)
                    response = model.backend.chat(model.conversation, config)
                    print(response)
                    model.conversation.append({"role": "assistant", "content": response})
                    print()
                    continue

                prompt = user_input
                if reasoning:
                    prompt = format_reasoning_prompt(prompt, True)

                print("  AI:  ", end="", flush=True)
                full = []
                for token in model.stream(prompt, max_tokens=args.max_tokens):
                    full.append(token)
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

    if args.browser:
        # Device Flow: öffne Browser
        _device_flow_login(client)
        return

    # Fallback: Username + Password
    username = args.username or input("  Username: ").strip()
    password = getpass.getpass("  Passwort: ")

    try:
        result = client.login(username, password)
        print(f"\n  Eingeloggt als: {result.get('username')}")
        print(f"  API-Key gespeichert.\n")
    except RuntimeError as e:
        print(f"\n  Fehler: {e}\n")


def _device_flow_login(client):
    """Device auth flow: CLI → Browser → Login → CLI bekommt Key."""
    import webbrowser

    print(f"\n  DeepNetz Login")
    print(f"  ─────────────────────────────────────")

    # 1. Start device flow
    try:
        result = client._request("/v1/auth/device", method="POST")
        if not result:
            print(f"  Registry nicht erreichbar.\n")
            return
    except Exception as e:
        print(f"  Fehler: {e}\n")
        return

    device_code = result["device_code"]
    user_code = result["user_code"]
    url = result["verification_url"]

    print(f"  Code: {user_code}")
    print(f"  Öffne Browser: {url}")
    print()

    # 2. Open browser
    webbrowser.open(url)

    # 3. Poll until complete
    print(f"  Warte auf Login im Browser...", end="", flush=True)
    import time as _time
    for _ in range(60):  # max 5 min (60 * 5s)
        _time.sleep(5)
        try:
            poll = client._request(f"/v1/auth/device/{device_code}/poll")
            if poll and poll.get("status") == "complete":
                api_key = poll["api_key"]
                client._save_api_key(api_key)
                print(f"\n\n  Login erfolgreich!")
                print(f"  API-Key gespeichert.\n")
                return
        except Exception:
            pass
        print(".", end="", flush=True)

    print(f"\n\n  Timeout — Login nicht abgeschlossen.\n")


def cmd_search(args):
    from deepnetz.engine.cards import load_cards, search_cards

    # 1. Search cached cards first (instant, no login needed)
    cards = load_cards()
    results = search_cards(args.query, cards)

    if results:
        print(f"\n  DeepNetz Modelle — '{args.query}' ({len(results)} Treffer)")
        print(f"  {'─' * 65}")
        for c in results[:args.limit]:
            params = f"{c.params_b:.0f}B" if c.params_b else ""
            active = f" ({c.active_params_b:.0f}B aktiv)" if c.active_params_b != c.params_b and c.active_params_b else ""
            tags = " ".join(f"[{t}]" for t in c.tags[:3])
            quants_count = len(set(q["name"] for q in c.quants if q["name"]))
            dl = f"{c.downloads // 1000}k" if c.downloads > 1000 else str(c.downloads)
            print(f"  {c.name:<30} {params:>5}{active:<15} {quants_count} quants  {dl:>6} dl  {tags}")
        print(f"\n  Pull: deepnetz pull <name>")
        if len(results) > args.limit:
            print(f"  ({len(results) - args.limit} weitere Treffer nicht angezeigt)")
        print()
        return

    # 2. Fallback: Live HF search via registry
    try:
        from deepnetz.registry.client import RegistryClient
        client = RegistryClient()
        hf_results = client.search(args.query, limit=args.limit)
        if hf_results:
            print(f"\n  HuggingFace Suche — '{args.query}' ({len(hf_results)} Treffer)")
            print(f"  {'─' * 60}")
            for m in hf_results:
                repo = m.get("repo", "")
                dl = m.get("downloads", 0)
                dl_str = f"{dl // 1000}k" if dl > 1000 else str(dl)
                print(f"  {repo:<45} {dl_str:>8} Downloads")
            print(f"\n  Pull: deepnetz pull <repo>\n")
            return
    except Exception:
        pass

    print(f"\n  Keine Ergebnisse für '{args.query}'.")
    print(f"  Versuche: deepnetz pull <hf-repo-name>\n")


def cmd_pull(args):
    from deepnetz.engine.cards import load_cards, search_cards, recommend_quant
    from deepnetz.engine.downloader import pull_model
    from deepnetz.engine.hardware import detect_hardware

    model_name = args.model
    auto = getattr(args, 'yes', False)

    # 1. Find model card
    cards = load_cards()
    matches = search_cards(model_name, cards)
    card = matches[0] if matches else None

    if card and card.quants and not auto:
        # 2. Show model info + variants
        hw = detect_hardware()
        rec = recommend_quant(card, vram_mb=hw.total_vram_mb, ram_mb=hw.ram_mb)
        rec_name = rec["name"] if rec else "Q4_K_M"

        print(f"\n  {card.name}")
        if card.params_b:
            active = f" ({card.active_params_b:.0f}B aktiv)" if card.active_params_b != card.params_b else ""
            print(f"  {card.params_b:.0f}B Parameter{active} | {card.architecture}")
        if card.license:
            print(f"  Lizenz: {card.license}", end="")
        if card.context_length:
            print(f" | Context: {card.context_length:,}", end="")
        print()
        print(f"  {'─' * 50}")

        # Show quant variants
        seen = set()
        variants = []
        for q in card.quants:
            if q["name"] in seen or not q["name"]:
                continue
            seen.add(q["name"])
            variants.append(q)

        for i, q in enumerate(variants, 1):
            size_str = f"{q['size_mb'] / 1024:.1f} GB" if q['size_mb'] > 1024 else f"{q['size_mb']} MB"
            marker = " ← Empfohlen" if q["name"] == rec_name else ""
            print(f"  {i:>2}. {q['name']:<10} {size_str:>10}{marker}")

        # Check local + Ollama
        print(f"  {'─' * 50}")
        from deepnetz.engine.downloader import resolve_local_model
        local = resolve_local_model(model_name)
        if local:
            print(f"  Lokal: {local}")

        # Check Ollama
        try:
            from deepnetz.backends.ollama import OllamaBackend
            ob = OllamaBackend()
            if ob.detect().available:
                ollama_models = ob.list_models()
                for m in ollama_models:
                    if model_name.lower().replace("-", "").replace("_", "") in m.name.lower().replace("-", "").replace("_", ""):
                        print(f"  Ollama: {m.name}")
                        break
        except Exception:
            pass

        # 3. Ask for variant
        default_idx = 1
        for i, q in enumerate(variants, 1):
            if q["name"] == rec_name:
                default_idx = i
                break

        try:
            choice = input(f"\n  Variante [{default_idx}]: ").strip()
            if not choice:
                choice = str(default_idx)
            idx = int(choice) - 1
            if 0 <= idx < len(variants):
                selected = variants[idx]
            else:
                selected = variants[default_idx - 1]
        except (ValueError, EOFError, KeyboardInterrupt):
            selected = variants[default_idx - 1]
            print()

        print(f"\n  Pulling {card.name} {selected['name']}...")
        from deepnetz.registry.store import RegistryStore
        store = RegistryStore()
        store._pull_from_repo(selected["repo"], selected["name"])

    else:
        # No card found or --yes flag: direct pull
        if args.quant == "auto" and auto:
            pass  # Let pull_model handle it
        pull_model(model_name, quant=args.quant)

    # Log pull
    try:
        from deepnetz.registry.client import RegistryClient
        client = RegistryClient()
        if client.is_authenticated:
            client.log_pull(model_name, quant=args.quant)
    except Exception:
        pass


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


def cmd_optimize(args):
    from deepnetz.engine.optimize import install_ik_llama, analyze_model, print_analysis

    if getattr(args, 'install_ik_llama', False):
        install_ik_llama(cuda=not getattr(args, 'cpu', False))
        return

    if hasattr(args, 'model') and args.model:
        from deepnetz.engine.resolver import resolve_model
        try:
            path = resolve_model(args.model)
        except FileNotFoundError:
            path = args.model
        report = analyze_model(path)
        print_analysis(report)
    else:
        print("\n  deepnetz optimize <model>          Analyse + Empfehlungen")
        print("  deepnetz optimize --install-ik-llama  Schnellere CUDA Kernels\n")


def cmd_convert(args):
    from deepnetz.engine.converter import convert_model
    convert_model(args.source, output_dir=args.output, quant=args.quant)


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
    p_run.add_argument("--image", help="Image path for vision models")
    p_run.add_argument("--reasoning", action="store_true", help="Enable reasoning mode")
    p_run.add_argument("--draft", help="Draft model for speculative decoding (e.g. small 3B model)")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start OpenAI-compatible API server")
    p_serve.add_argument("model", nargs="?", default="", help="Path to GGUF model file (optional — load via UI)")
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
    p_login.add_argument("--browser", action="store_true", help="Login über Browser (GitHub/Google)")

    # search
    p_search = subparsers.add_parser("search", help="Modelle suchen (über Registry)")
    p_search.add_argument("query", help="Suchbegriff (z.B. Qwen, Llama, code)")
    p_search.add_argument("--limit", type=int, default=15, help="Max Ergebnisse")

    # pull
    p_pull = subparsers.add_parser("pull", help="Modell herunterladen")
    p_pull.add_argument("model", help="Modellname oder HF Repo (z.B. Qwen3.5-35B)")
    p_pull.add_argument("--quant", default="auto", help="Quantisierung (auto, Q4_K_M, Q8_0, ...)")
    p_pull.add_argument("-y", "--yes", action="store_true", help="Automatisch beste Variante wählen")

    # list
    p_list = subparsers.add_parser("list", help="Lokale Modelle anzeigen")

    # registry (server)
    p_reg = subparsers.add_parser("registry", help="Registry Server starten")
    p_reg.add_argument("--port", type=int, default=8090, help="Port (Standard: 8090)")

    # optimize
    p_opt = subparsers.add_parser("optimize", help="Modell analysieren + Optimierungen empfehlen")
    p_opt.add_argument("model", nargs="?", help="Modell zum Analysieren")
    p_opt.add_argument("--install-ik-llama", action="store_true", help="ik_llama.cpp installieren (1.3-1.5x schneller)")
    p_opt.add_argument("--cpu", action="store_true", help="Ohne CUDA kompilieren")

    # convert
    p_conv = subparsers.add_parser("convert", help="Modell konvertieren (HF → GGUF)")
    p_conv.add_argument("source", help="Quelle (HF Repo oder lokales Verzeichnis)")
    p_conv.add_argument("--quant", default="Q4_K_M", help="Quantisierung")
    p_conv.add_argument("--output", default=".", help="Ausgabeverzeichnis")

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
        "optimize": cmd_optimize,
        "convert": cmd_convert,
        "download": cmd_download,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
