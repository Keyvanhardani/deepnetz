"""
DeepNetz Model — the main entry point.

Handles model loading, inference planning, and execution.
Supports GPU, CPU, and hybrid GPU+CPU modes.
"""

import os
from typing import Optional, Generator
from deepnetz.engine.hardware import detect_hardware, print_hardware, GPUInfo
from deepnetz.engine.planner import ModelSpec, plan_inference, print_plan
from deepnetz.engine.gguf_reader import gguf_to_model_spec, print_model_info
from deepnetz.engine.backend import DeepNetzBackend


class Model:
    """
    Load and run a model with automatic hardware optimization.

    Usage:
        # Auto-detect everything
        model = Model("path/to/model.gguf")

        # Explicit budgets
        model = Model("model.gguf", gpu_budget="8GB", ram_budget="32GB")

        # CPU-only
        model = Model("model.gguf", gpu_budget="0")

        # Chat
        response = model.chat("Hello!")

        # Stream
        for token in model.stream("Tell me a story"):
            print(token, end="", flush=True)
    """

    def __init__(self, model_path: str,
                 gpu_budget: str = "auto",
                 ram_budget: str = "auto",
                 target_context: int = 4096,
                 cpu_only: bool = False,
                 verbose: bool = True):

        self.model_path = model_path
        self.verbose = verbose

        # Detect hardware
        self.hw = detect_hardware()

        # Force CPU-only if requested
        if cpu_only:
            self.hw.gpus = []
            self.hw.total_vram_mb = 0
            self.hw.has_cuda = False

        if verbose:
            print_hardware(self.hw)

        # Parse budgets
        gpu_mb = self._parse_budget(gpu_budget, self.hw.total_vram_mb)
        ram_mb = self._parse_budget(ram_budget, self.hw.ram_mb)

        # Force CPU-only if gpu_budget is 0
        if gpu_mb == 0:
            self.hw.gpus = []
            self.hw.total_vram_mb = 0
            self.hw.has_cuda = False

        # Read model specs from GGUF
        if os.path.exists(model_path):
            self.spec = gguf_to_model_spec(model_path)
            if verbose:
                print_model_info(self.spec)
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Use 'deepnetz download <model>' to fetch from HuggingFace."
            )

        # Plan inference
        self.plan = plan_inference(
            self.spec, self.hw,
            target_context=target_context,
            gpu_budget_mb=gpu_mb,
            ram_budget_mb=ram_mb
        )
        if verbose:
            print_plan(self.plan, self.spec)

        # Create backend
        self.backend = DeepNetzBackend(
            model_path=model_path,
            plan=self.plan,
            hw=self.hw,
            spec=self.spec
        )

    def load(self):
        """Pre-load the model (otherwise loaded on first use)."""
        self.backend.load(verbose=self.verbose)
        return self

    def chat(self, prompt: str, max_tokens: int = 512,
             temperature: float = 0.7) -> str:
        """Chat with the model. Returns full response."""
        return self.backend.chat(prompt, max_tokens, temperature, stream=False)

    def stream(self, prompt: str, max_tokens: int = 512,
               temperature: float = 0.7) -> Generator:
        """Stream response token by token."""
        return self.backend.chat(prompt, max_tokens, temperature, stream=True)

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        """Raw text completion (no chat template)."""
        return self.backend.complete(prompt, max_tokens)

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
