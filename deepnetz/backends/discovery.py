"""
Backend Discovery — auto-detect all available inference backends.
"""

from typing import List, Optional
from deepnetz.backends.base import BackendAdapter, BackendInfo


_cached_backends = None


def discover_backends(use_cache: bool = True) -> List[BackendAdapter]:
    """Probe all known backends and return available ones.
    Results are cached after first call for speed."""
    global _cached_backends
    if use_cache and _cached_backends is not None:
        return _cached_backends

    backends = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _probe(cls_path):
        try:
            module_path, class_name = cls_path.rsplit(".", 1)
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            b = cls()
            if b.detect().available:
                return b
        except Exception:
            pass
        return None

    # Native first (no network, instant) — always check synchronously
    try:
        from deepnetz.backends.native import NativeBackend
        b = NativeBackend()
        if b.detect().available:
            backends.append(b)
    except Exception:
        pass

    # Network backends in parallel (Ollama, vLLM, LMStudio, HF)
    network_backends = [
        "deepnetz.backends.ollama.OllamaBackend",
        "deepnetz.backends.vllm.VLLMBackend",
        "deepnetz.backends.lmstudio.LMStudioBackend",
        "deepnetz.backends.huggingface.HuggingFaceBackend",
    ]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_probe, cls) for cls in network_backends]
        for future in futures:
            try:
                result = future.result(timeout=3)
                if result:
                    backends.append(result)
            except Exception:
                pass

    _cached_backends = backends
    return backends


def get_backend(name: str) -> Optional[BackendAdapter]:
    """Get a specific backend by name."""
    mapping = {
        "native": "deepnetz.backends.native.NativeBackend",
        "ollama": "deepnetz.backends.ollama.OllamaBackend",
        "vllm": "deepnetz.backends.vllm.VLLMBackend",
        "lmstudio": "deepnetz.backends.lmstudio.LMStudioBackend",
        "huggingface": "deepnetz.backends.huggingface.HuggingFaceBackend",
        "hf": "deepnetz.backends.huggingface.HuggingFaceBackend",
        "remote": "deepnetz.backends.remote.RemoteBackend",
    }

    class_path = mapping.get(name.lower())
    if not class_path:
        return None

    module_path, class_name = class_path.rsplit(".", 1)
    try:
        import importlib
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls()
    except (ImportError, AttributeError):
        return None


def select_best_backend(model_ref: str, backends: List[BackendAdapter]) -> Optional[BackendAdapter]:
    """Select the best backend for a given model reference."""
    # Protocol-specific routing
    if "://" in model_ref:
        protocol = model_ref.split("://")[0]
        protocol_map = {
            "ollama": "ollama",
            "hf": "native",
            "huggingface": "native",
            "lmstudio": "lmstudio",
            "vllm": "vllm",
            "llamacpp": "native",
            "mlx": "native",
        }
        preferred = protocol_map.get(protocol)
        if preferred:
            for b in backends:
                if b.name == preferred:
                    return b

    # For local GGUF files, check if Ollama has it loaded (faster)
    for b in backends:
        if b.name == "ollama":
            models = b.list_models()
            for m in models:
                if model_ref in m.name or m.name in model_ref:
                    return b

    # Default: prefer native (fastest, most features)
    for b in backends:
        if b.name == "native":
            return b

    # Any available backend
    return backends[0] if backends else None


def print_backends(backends: List[BackendAdapter]):
    """Pretty-print available backends."""
    print(f"\n  DeepNetz Backends")
    print(f"  {'-' * 50}")
    for b in backends:
        info = b.detect()
        models = b.list_models()
        status = "active" if info.available else "not found"
        print(f"  {info.name:<12} {info.version:<12} {len(models)} models  [{status}]")
    if not backends:
        print("  No backends found. Install llama-cpp-python or Ollama.")
    print()
