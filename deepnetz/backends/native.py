"""
Native Backend — llama-cpp-python direct inference.

Fastest path. Supports TurboQuant KV cache, GPU/CPU hybrid,
all GGUF models. This is DeepNetz's primary backend.
"""

import os
from typing import Generator, List, Dict, Optional
from deepnetz.backends.base import (
    BackendAdapter, BackendInfo, ModelEntry,
    GenerationConfig, BackendStats
)


class NativeBackend(BackendAdapter):
    """Direct llama-cpp-python inference backend."""

    def __init__(self):
        self._llm = None
        self._model_path = ""
        self._config = {}

    @property
    def name(self) -> str:
        return "native"

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def detect(self) -> BackendInfo:
        try:
            import llama_cpp
            return BackendInfo(
                name="native",
                version=getattr(llama_cpp, "__version__", "unknown"),
                available=True,
                details="llama-cpp-python (direct GGUF inference)"
            )
        except ImportError:
            return BackendInfo(
                name="native", version="", available=False,
                details="pip install llama-cpp-python"
            )

    def list_models(self) -> List[ModelEntry]:
        """Scan common directories for GGUF files."""
        import glob
        import platform
        models = []
        search_dirs = [
            os.path.expanduser("~/models"),
            os.path.expanduser("~/.cache/deepnetz/models"),
        ]
        if platform.system() == "Windows":
            search_dirs.extend(["D:/models", "E:/models", "C:/models"])
        else:
            search_dirs.extend(["/mnt/d/models", "/mnt/e/models", "/models"])

        for d in search_dirs:
            if os.path.exists(d):
                for f in glob.glob(os.path.join(d, "**/*.gguf"), recursive=True):
                    size_mb = os.path.getsize(f) // (1024 * 1024)
                    models.append(ModelEntry(
                        name=os.path.basename(f),
                        size_mb=size_mb,
                        backend="native",
                        path=f
                    ))
        return models

    def pull(self, model_name: str, progress_callback=None) -> str:
        try:
            from huggingface_hub import hf_hub_download
            if "/" in model_name:
                parts = model_name.split("/")
                repo = "/".join(parts[:2])
                filename = parts[2] if len(parts) > 2 else None
                if filename:
                    return hf_hub_download(repo, filename, resume_download=True)
        except ImportError:
            pass
        return model_name

    def load(self, model_ref: str, n_ctx: int = 4096,
             n_gpu_layers: int = -1, n_threads: int = 0,
             kv_type_k: str = "f16", kv_type_v: str = "f16",
             **kwargs) -> None:
        from llama_cpp import Llama

        if n_threads <= 0:
            n_threads = os.cpu_count() or 4

        self._model_path = model_ref
        self._config = {
            "n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers,
            "n_threads": n_threads, "kv_type_k": kv_type_k,
        }

        self._llm = Llama(
            model_path=model_ref,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
        )

    def chat(self, messages: List[Dict[str, str]],
             config: Optional[GenerationConfig] = None) -> str:
        if not self._llm:
            raise RuntimeError("No model loaded")
        cfg = config or GenerationConfig()

        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
        )
        return response["choices"][0]["message"]["content"]

    def stream(self, messages: List[Dict[str, str]],
               config: Optional[GenerationConfig] = None) -> Generator[str, None, None]:
        if not self._llm:
            raise RuntimeError("No model loaded")
        cfg = config or GenerationConfig()

        for chunk in self._llm.create_chat_completion(
            messages=messages,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content

    def unload(self) -> None:
        if self._llm:
            del self._llm
            self._llm = None

    def stats(self) -> BackendStats:
        return BackendStats(
            model_name=os.path.basename(self._model_path),
            context_max=self._config.get("n_ctx", 0),
        )
