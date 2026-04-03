"""
DeepNetz Backend — wraps llama-cpp-python with auto-optimization.

Translates the InferencePlan into llama-cpp-python configuration
and provides a unified chat/completion interface.
"""

import os
from typing import Optional, Generator
from deepnetz.engine.planner import InferencePlan, ModelSpec
from deepnetz.engine.hardware import HardwareProfile


class DeepNetzBackend:
    """
    Inference backend powered by llama.cpp.

    Automatically configures:
    - GPU/CPU layer split based on plan
    - KV cache type (TurboQuant if available)
    - Context length
    - Thread count
    """

    def __init__(self, model_path: str, plan: InferencePlan,
                 hw: HardwareProfile, spec: ModelSpec):
        self.model_path = model_path
        self.plan = plan
        self.hw = hw
        self.spec = spec
        self._llm = None

    def load(self, verbose: bool = True):
        """Load model with optimized settings from plan."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with:\n"
                "  pip install llama-cpp-python\n"
                "For CUDA: pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cu124"
            )

        if verbose:
            mode = "GPU+CPU" if self.plan.n_gpu_layers > 0 else "CPU-only"
            print(f"  Loading {self.spec.name} ({mode})...")
            print(f"  Layers: {self.plan.n_gpu_layers} GPU + {self.plan.n_cpu_layers} CPU")
            print(f"  Context: {self.plan.max_context:,} tokens")
            print(f"  Threads: {self.hw.cpu_cores}")

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.plan.max_context,
            n_gpu_layers=self.plan.n_gpu_layers,
            n_threads=self.hw.cpu_cores,
            verbose=verbose,
        )

        if verbose:
            print(f"  Model loaded.\n")

    def chat(self, prompt: str, max_tokens: int = 512,
             temperature: float = 0.7, stream: bool = False):
        """Chat completion."""
        if self._llm is None:
            self.load()

        messages = [{"role": "user", "content": prompt}]

        if stream:
            return self._chat_stream(messages, max_tokens, temperature)
        else:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response["choices"][0]["message"]["content"]

    def _chat_stream(self, messages, max_tokens, temperature) -> Generator:
        """Streaming chat."""
        for chunk in self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content

    def complete(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7) -> str:
        """Text completion."""
        if self._llm is None:
            self.load()

        result = self._llm(prompt, max_tokens=max_tokens,
                           temperature=temperature)
        return result["choices"][0]["text"]

    def __del__(self):
        if self._llm is not None:
            del self._llm
