"""
DeepNetz Model — orchestrates backend selection, planning, and inference.
"""

import os
from typing import Optional, Generator, List, Dict
from deepnetz.engine.hardware import detect_hardware, print_hardware
from deepnetz.engine.planner import ModelSpec, plan_inference, print_plan
from deepnetz.backends.base import BackendAdapter, GenerationConfig
from deepnetz.backends.discovery import discover_backends, select_best_backend, get_backend


class Model:
    """
    Load and run a model with automatic hardware optimization.

    Usage:
        model = Model("model.gguf")                    # auto everything
        model = Model("ollama://qwen3.5:35b")           # use Ollama
        model = Model("model.gguf", cpu_only=True)       # CPU only
        model = Model("model.gguf", backend="ollama")    # force backend

        response = model.chat("Hello!")
        for token in model.stream("Tell me a story"):
            print(token, end="")
    """

    def __init__(self, model_ref: str,
                 gpu_budget: str = "auto",
                 ram_budget: str = "auto",
                 target_context: int = 4096,
                 cpu_only: bool = False,
                 backend: str = "auto",
                 verbose: bool = True):

        self.model_ref = model_ref
        self.verbose = verbose
        self.conversation: List[Dict[str, str]] = []

        # Detect hardware
        self.hw = detect_hardware()
        if cpu_only:
            self.hw.gpus = []
            self.hw.total_vram_mb = 0
            self.hw.has_cuda = False
        if verbose:
            print_hardware(self.hw)

        # Parse budgets
        gpu_mb = self._parse_budget(gpu_budget, self.hw.total_vram_mb)
        ram_mb = self._parse_budget(ram_budget, self.hw.ram_mb)
        if gpu_mb == 0:
            self.hw.gpus = []
            self.hw.total_vram_mb = 0
            self.hw.has_cuda = False

        # Discover backends
        self.backends = discover_backends()
        if verbose and self.backends:
            from deepnetz.backends.discovery import print_backends
            print_backends(self.backends)

        # Select backend
        if backend != "auto":
            self.backend = get_backend(backend)
            if not self.backend:
                raise RuntimeError(f"Backend '{backend}' not available")
        else:
            self.backend = select_best_backend(model_ref, self.backends)
            if not self.backend:
                raise RuntimeError("No inference backend available. Install llama-cpp-python or Ollama.")

        # Resolve model path — handle protocol URLs for ALL backends
        self.model_path = model_ref
        if "://" in model_ref:
            protocol, path = model_ref.split("://", 1)
            if self.backend.name == "native":
                # Native backend needs a local file path — resolve via resolver
                from deepnetz.engine.resolver import resolve_model
                try:
                    self.model_path = resolve_model(model_ref)
                except (FileNotFoundError, ValueError, ImportError) as e:
                    raise RuntimeError(
                        f"Cannot resolve '{model_ref}' for native backend: {e}\n\n"
                        f"Hints:\n"
                        f"  ollama://  → Make sure Ollama is running (ollama serve)\n"
                        f"  hf://      → pip install huggingface_hub\n"
                        f"  Or use a local file: deepnetz run /path/to/model.gguf"
                    )
            else:
                # Non-native backends: strip protocol prefix
                # e.g. ollama://qwen3.5:35b → qwen3.5:35b
                self.model_path = path
        elif self.backend.name == "native":
            # No protocol — try to resolve local path
            from deepnetz.engine.resolver import resolve_model
            try:
                self.model_path = resolve_model(model_ref)
            except FileNotFoundError:
                pass

        # Read GGUF specs (if local file)
        self.spec = None
        if os.path.exists(self.model_path):
            from deepnetz.engine.gguf_reader import gguf_to_model_spec, print_model_info
            self.spec = gguf_to_model_spec(self.model_path)
            if verbose:
                print_model_info(self.spec)

        # Plan inference
        if self.spec:
            self.plan = plan_inference(
                self.spec, self.hw,
                target_context=target_context,
                gpu_budget_mb=gpu_mb,
                ram_budget_mb=ram_mb
            )
            if verbose:
                print_plan(self.plan, self.spec)
        else:
            self.plan = None

        if verbose:
            print(f"  Backend: {self.backend.name}\n")

    def load(self):
        """Pre-load model."""
        if self.plan and self.backend.name == "native":
            self.backend.load(
                self.model_path,
                n_ctx=self.plan.max_context,
                n_gpu_layers=self.plan.n_gpu_layers,
                n_threads=self.hw.cpu_cores,
                kv_type_k=self.plan.kv_type_k,
                kv_type_v=self.plan.kv_type_v,
            )
        else:
            self.backend.load(self.model_path)
        return self

    def chat(self, prompt: str, max_tokens: int = 512,
             temperature: float = 0.7) -> str:
        """Chat with conversation history."""
        if not self.backend.is_loaded:
            self.load()

        self.conversation.append({"role": "user", "content": prompt})
        config = GenerationConfig(
            max_tokens=max_tokens, temperature=temperature
        )
        response = self.backend.chat(self.conversation, config)
        self.conversation.append({"role": "assistant", "content": response})
        return response

    def stream(self, prompt: str, max_tokens: int = 512,
               temperature: float = 0.7) -> Generator:
        """Stream with conversation history."""
        if not self.backend.is_loaded:
            self.load()

        self.conversation.append({"role": "user", "content": prompt})
        config = GenerationConfig(
            max_tokens=max_tokens, temperature=temperature
        )
        full_response = []
        for token in self.backend.stream(self.conversation, config):
            full_response.append(token)
            yield token
        self.conversation.append({"role": "assistant", "content": "".join(full_response)})

    def reset(self):
        """Clear conversation history."""
        self.conversation = []

    @staticmethod
    def _parse_budget(budget: str, auto_value: int) -> int:
        if budget == "auto":
            return auto_value
        budget = budget.strip().upper()
        if budget.endswith("GB"):
            return int(float(budget[:-2]) * 1024)
        elif budget.endswith("MB"):
            return int(float(budget[:-2]))
        return int(budget)
