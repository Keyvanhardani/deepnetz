"""
Ollama Backend — connects to Ollama's REST API.

Ollama must be installed and running (localhost:11434).
Supports pull, chat, streaming, model management.
"""

import json
import urllib.request
import urllib.error
from typing import Generator, List, Dict, Optional
from deepnetz.backends.base import (
    BackendAdapter, BackendInfo, ModelEntry,
    GenerationConfig, BackendStats
)


class OllamaBackend(BackendAdapter):
    """Ollama REST API backend adapter."""

    def __init__(self, host: str = "http://localhost:11434"):
        self._host = host.rstrip("/")
        self._model = ""
        self._loaded = False

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def is_loaded(self) -> bool:
        return self._loaded and bool(self._model)

    def _request(self, endpoint: str, data: dict = None,
                 method: str = "GET", stream: bool = False):
        url = f"{self._host}{endpoint}"
        if data:
            req = urllib.request.Request(
                url, data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
                method=method or "POST"
            )
        else:
            req = urllib.request.Request(url, method=method)

        try:
            resp = urllib.request.urlopen(req, timeout=120)
            if stream:
                return resp  # caller reads line by line
            return json.loads(resp.read().decode())
        except (urllib.error.URLError, ConnectionResetError, OSError):
            return None

    def detect(self) -> BackendInfo:
        resp = self._request("/api/version")
        if resp:
            return BackendInfo(
                name="ollama",
                version=resp.get("version", "unknown"),
                available=True,
                details=f"Ollama {resp.get('version', '')} at {self._host}"
            )
        return BackendInfo(
            name="ollama", version="", available=False,
            details="Ollama not running. Install: https://ollama.ai"
        )

    def list_models(self) -> List[ModelEntry]:
        resp = self._request("/api/tags")
        if not resp:
            return []
        models = []
        for m in resp.get("models", []):
            size_mb = m.get("size", 0) // (1024 * 1024)
            name = m.get("name", "")
            details = m.get("details", {})
            models.append(ModelEntry(
                name=name,
                size_mb=size_mb,
                family=details.get("family", ""),
                quant=details.get("quantization_level", ""),
                backend="ollama",
                parameters_b=float(details.get("parameter_size", "0").replace("B", "")),
            ))
        return models

    def pull(self, model_name: str, progress_callback=None) -> str:
        resp = self._request("/api/pull", {"name": model_name, "stream": False})
        if resp and resp.get("status") == "success":
            return model_name
        return ""

    def load(self, model_ref: str, n_ctx: int = 4096,
             n_gpu_layers: int = -1, n_threads: int = 0,
             kv_type_k: str = "f16", kv_type_v: str = "f16",
             **kwargs) -> None:
        self._model = model_ref
        # Ollama loads on first request — warmup (non-blocking if fails)
        try:
            self._request("/api/generate", {
                "model": model_ref,
                "prompt": "hi",
                "options": {"num_predict": 1, "num_ctx": min(n_ctx, 2048)},
                "stream": False,
            })
        except Exception:
            pass  # Model will load on first real request
        self._loaded = True

    def chat(self, messages: List[Dict[str, str]],
             config: Optional[GenerationConfig] = None) -> str:
        cfg = config or GenerationConfig()
        resp = self._request("/api/chat", {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "repeat_penalty": cfg.repeat_penalty,
                "num_predict": cfg.max_tokens,
            }
        })
        if resp:
            return resp.get("message", {}).get("content", "")
        return ""

    def stream(self, messages: List[Dict[str, str]],
               config: Optional[GenerationConfig] = None) -> Generator[str, None, None]:
        cfg = config or GenerationConfig()
        data = json.dumps({
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "repeat_penalty": cfg.repeat_penalty,
                "num_predict": cfg.max_tokens,
            }
        }).encode()

        url = f"{self._host}/api/chat"
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            for line in resp:
                if line:
                    try:
                        chunk = json.loads(line.decode())
                    except json.JSONDecodeError:
                        continue
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
        except Exception as e:
            yield f"\n[Error: Ollama connection failed — {type(e).__name__}: {e}]"

    def unload(self) -> None:
        self._model = ""
        self._loaded = False

    def stats(self) -> BackendStats:
        return BackendStats(model_name=self._model)
