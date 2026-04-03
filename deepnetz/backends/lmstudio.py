"""
LM Studio Backend — connects via lms CLI or REST API.

LM Studio runs as a daemon (llmster) with OpenAI-compatible API.
"""

import json
import subprocess
import urllib.request
import urllib.error
import os
import glob
import platform
from typing import Generator, List, Dict, Optional
from deepnetz.backends.base import (
    BackendAdapter, BackendInfo, ModelEntry,
    GenerationConfig, BackendStats
)


class LMStudioBackend(BackendAdapter):
    """LM Studio backend via REST API or lms CLI."""

    def __init__(self, host: str = "http://localhost:1234"):
        self._host = host.rstrip("/")
        self._model = ""

    @property
    def name(self) -> str:
        return "lmstudio"

    @property
    def is_loaded(self) -> bool:
        return bool(self._model)

    def detect(self) -> BackendInfo:
        # Check lms CLI
        try:
            result = subprocess.run(["lms", "version"],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                return BackendInfo(name="lmstudio", version=version,
                                  available=True, details=f"LM Studio {version}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check API
        try:
            req = urllib.request.Request(f"{self._host}/v1/models")
            resp = urllib.request.urlopen(req, timeout=3)
            return BackendInfo(name="lmstudio", version="api",
                               available=True, details=f"LM Studio at {self._host}")
        except Exception:
            pass

        # Check if installed but not running
        home = os.path.expanduser("~")
        lms_paths = [
            os.path.join(home, ".cache", "lm-studio"),
            os.path.join(home, "AppData", "Roaming", "LM Studio"),
        ]
        for p in lms_paths:
            if os.path.exists(p):
                return BackendInfo(name="lmstudio", version="installed",
                                  available=True, details="LM Studio installed (not running)")

        return BackendInfo(name="lmstudio", version="", available=False,
                          details="LM Studio not found. https://lmstudio.ai")

    def list_models(self) -> List[ModelEntry]:
        models = []
        # Try API first
        try:
            req = urllib.request.Request(f"{self._host}/v1/models")
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())
            for m in data.get("data", []):
                models.append(ModelEntry(name=m["id"], backend="lmstudio"))
            return models
        except Exception:
            pass

        # Scan local cache
        home = os.path.expanduser("~")
        cache_dirs = [
            os.path.join(home, ".cache", "lm-studio", "models"),
            os.path.join(home, "AppData", "Roaming", "LM Studio", "models"),
        ]
        for d in cache_dirs:
            if os.path.exists(d):
                for f in glob.glob(os.path.join(d, "**/*.gguf"), recursive=True):
                    models.append(ModelEntry(
                        name=os.path.basename(f),
                        size_mb=os.path.getsize(f) // (1024*1024),
                        backend="lmstudio",
                        path=f
                    ))
        return models

    def pull(self, model_name: str, progress_callback=None) -> str:
        try:
            subprocess.run(["lms", "get", model_name],
                          capture_output=True, timeout=300)
        except Exception:
            pass
        return model_name

    def load(self, model_ref: str, n_ctx: int = 4096, **kwargs) -> None:
        self._model = model_ref
        try:
            subprocess.run(["lms", "load", model_ref, "--yes"],
                          capture_output=True, timeout=60)
        except Exception:
            pass

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
            data=data, headers={"Content-Type": "application/json"}
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
            data=data, headers={"Content-Type": "application/json"}
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
        try:
            subprocess.run(["lms", "unload", "--yes"],
                          capture_output=True, timeout=10)
        except Exception:
            pass
        self._model = ""

    def stats(self) -> BackendStats:
        return BackendStats(model_name=self._model)
