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
        self._evictor = None

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

        # Map KV type names to GGML type IDs
        type_map = {"f16": 1, "f32": 0, "q8_0": 8, "q4_0": 2, "q4_1": 3,
                    "q5_0": 6, "q5_1": 7, "turbo4_0": 41, "turbo3_0": 42, "turbo2_0": 43}
        type_k_id = type_map.get(kv_type_k, 1)
        type_v_id = type_map.get(kv_type_v, 1)

        llama_kwargs = {
            "model_path": model_ref,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_threads": n_threads,
            "verbose": False,
        }

        # Only pass type_k/type_v when using KV compression (not f16)
        # Explicitly setting f16 (type_id=1) triggers a slow path in llama-cpp
        use_kv_quant = kv_type_k != "f16" and kv_type_v != "f16"
        if use_kv_quant:
            try:
                self._llm = Llama(type_k=type_k_id, type_v=type_v_id, **llama_kwargs)
            except TypeError:
                self._llm = Llama(**llama_kwargs)
        else:
            self._llm = Llama(**llama_kwargs)

        # Initialize eviction if context is large
        if n_ctx >= 2048:
            try:
                from deepnetz.cache.eviction import AttentionSinkEvictor, EvictionConfig
                self._evictor = AttentionSinkEvictor(EvictionConfig(
                    max_cache_tokens=n_ctx,
                    sink_tokens=4,
                    window_size=min(512, n_ctx // 4),
                ))
            except ImportError:
                pass

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

    def stream_speculative(self, draft_backend, messages: List[Dict[str, str]],
                           config: Optional[GenerationConfig] = None,
                           k: int = 5) -> Generator[str, None, None]:
        """Stream with speculative decoding using a draft model."""
        if not self._llm:
            raise RuntimeError("No model loaded")
        cfg = config or GenerationConfig()

        from deepnetz.engine.speculative import speculative_generate_from_backends
        yield from speculative_generate_from_backends(
            target_backend=self,
            draft_backend=draft_backend,
            messages=messages,
            max_tokens=cfg.max_tokens,
            k=k,
            temperature=cfg.temperature,
        )

    def unload(self) -> None:
        if self._llm:
            try:
                self._llm.close()
            except Exception:
                pass
            self._llm = None
            import gc
            gc.collect()

    def stats(self) -> BackendStats:
        return BackendStats(
            model_name=os.path.basename(self._model_path),
            context_max=self._config.get("n_ctx", 0),
        )
