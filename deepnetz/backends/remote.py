"""
Remote Backend — connects to any OpenAI-compatible API endpoint.

Works with: OpenAI, Anthropic, Groq, Together, any local server.
"""

import json
import urllib.request
from typing import Generator, List, Dict, Optional
from deepnetz.backends.base import (
    BackendAdapter, BackendInfo, ModelEntry,
    GenerationConfig, BackendStats
)


class RemoteBackend(BackendAdapter):
    """Generic OpenAI-compatible API backend."""

    def __init__(self, host: str = "http://localhost:8080",
                 api_key: str = ""):
        self._host = host.rstrip("/")
        self._api_key = api_key
        self._model = ""

    @property
    def name(self) -> str:
        return "remote"

    @property
    def is_loaded(self) -> bool:
        return bool(self._model)

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def detect(self) -> BackendInfo:
        try:
            req = urllib.request.Request(
                f"{self._host}/v1/models", headers=self._headers()
            )
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())
            n = len(data.get("data", []))
            return BackendInfo(name="remote", version="api", available=True,
                               models_count=n, details=f"Remote API at {self._host}")
        except Exception:
            return BackendInfo(name="remote", version="", available=False,
                              details=f"No API at {self._host}")

    def list_models(self) -> List[ModelEntry]:
        try:
            req = urllib.request.Request(
                f"{self._host}/v1/models", headers=self._headers()
            )
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())
            return [ModelEntry(name=m["id"], backend="remote")
                    for m in data.get("data", [])]
        except Exception:
            return []

    def pull(self, model_name: str, progress_callback=None) -> str:
        return model_name  # Remote models don't need downloading

    def load(self, model_ref: str, **kwargs) -> None:
        self._model = model_ref

    def chat(self, messages: List[Dict[str, str]],
             config: Optional[GenerationConfig] = None) -> str:
        cfg = config or GenerationConfig()
        data = json.dumps({
            "model": self._model,
            "messages": messages,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self._host}/v1/chat/completions",
            data=data, headers=self._headers()
        )
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read().decode())
        return result["choices"][0]["message"]["content"]

    def stream(self, messages: List[Dict[str, str]],
               config: Optional[GenerationConfig] = None) -> Generator[str, None, None]:
        cfg = config or GenerationConfig()
        data = json.dumps({
            "model": self._model,
            "messages": messages,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            f"{self._host}/v1/chat/completions",
            data=data, headers=self._headers()
        )
        resp = urllib.request.urlopen(req, timeout=120)
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    content = chunk["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    pass

    def unload(self) -> None:
        self._model = ""

    def stats(self) -> BackendStats:
        return BackendStats(model_name=self._model)
