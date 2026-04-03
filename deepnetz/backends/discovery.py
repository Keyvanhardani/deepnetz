"""
Backend Discovery — auto-detect all available inference backends.
"""

from typing import List, Optional
from deepnetz.backends.base import BackendAdapter, BackendInfo


def discover_backends() -> List[BackendAdapter]:
    """Probe all known backends and return available ones."""
    backends = []

    # Native (llama-cpp-python)
    try:
        from deepnetz.backends.native import NativeBackend
        b = NativeBackend()
        if b.detect().available:
            backends.append(b)
    except Exception:
        pass

    # Ollama
    try:
        from deepnetz.backends.ollama import OllamaBackend
        b = OllamaBackend()
        if b.detect().available:
            backends.append(b)
    except Exception:
        pass

    # vLLM
    try:
        from deepnetz.backends.vllm import VLLMBackend
        b = VLLMBackend()
        if b.detect().available:
            backends.append(b)
    except Exception:
        pass

    # LM Studio
    try:
        from deepnetz.backends.lmstudio import LMStudioBackend
        b = LMStudioBackend()
        if b.detect().available:
            backends.append(b)
    except Exception:
        pass

    # HuggingFace
    try:
        from deepnetz.backends.huggingface import HuggingFaceBackend
        b = HuggingFaceBackend()
        if b.detect().available:
            backends.append(b)
    except Exception:
        pass

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
