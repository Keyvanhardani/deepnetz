"""
vLLM Backend — connects to vLLM OpenAI-compatible server.

Detects running vLLM instance or starts one via subprocess.
"""

import json
import subprocess
import urllib.request
import urllib.error
from typing import Generator, List, Dict, Optional
from deepnetz.backends.base import (
    BackendAdapter, BackendInfo, ModelEntry,
    GenerationConfig, BackendStats
)


class VLLMBackend(BackendAdapter):
    """vLLM inference backend via OpenAI-compatible API."""

    def __init__(self, host: str = "http://localhost:8000"):
        self._host = host.rstrip("/")
        self._model = ""
        self._process = None

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def is_loaded(self) -> bool:
        return bool(self._model)

    def detect(self) -> BackendInfo:
        # Check if vLLM is installed
        try:
            result = subprocess.run(["vllm", "--version"],
                                    capture_output=True, text=True, timeout=5)
            version = result.stdout.strip() or "installed"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Check Python import
            try:
                import vllm
                version = getattr(vllm, "__version__", "installed")
            except ImportError:
                return BackendInfo(name="vllm", version="", available=False,
                                  details="pip install vllm")

        # Check if server is running
        try:
            req = urllib.request.Request(f"{self._host}/v1/models")
            resp = urllib.request.urlopen(req, timeout=3)
            data = json.loads(resp.read().decode())
            n_models = len(data.get("data", []))
            return BackendInfo(name="vllm", version=version, available=True,
                               models_count=n_models,
                               details=f"vLLM server at {self._host}")
        except Exception:
            return BackendInfo(name="vllm", version=version, available=True,
                               details="vLLM installed (server not running)")

    def list_models(self) -> List[ModelEntry]:
        try:
            req = urllib.request.Request(f"{self._host}/v1/models")
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())
            return [ModelEntry(name=m["id"], backend="vllm")
                    for m in data.get("data", [])]
        except Exception:
            return []

    def pull(self, model_name: str, progress_callback=None) -> str:
        return model_name  # vLLM downloads from HF on serve

    def load(self, model_ref: str, n_ctx: int = 4096, **kwargs) -> None:
        self._model = model_ref
        # Check if already serving this model
        models = self.list_models()
        for m in models:
            if model_ref in m.name:
                return

        # Start vLLM server
        cmd = ["vllm", "serve", model_ref,
               "--max-model-len", str(n_ctx),
               "--port", self._host.split(":")[-1]]
        try:
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            import time
            time.sleep(10)  # wait for server startup
        except FileNotFoundError:
            raise RuntimeError("vLLM CLI not found. pip install vllm")

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
                chunk = json.loads(line[6:])
                content = chunk["choices"][0].get("delta", {}).get("content", "")
                if content:
                    yield content

    def unload(self) -> None:
        if self._process:
            self._process.terminate()
            self._process = None
        self._model = ""

    def stats(self) -> BackendStats:
        return BackendStats(model_name=self._model)
