"""
Backend Adapter Interface — all backends implement this protocol.

Each backend (Native, Ollama, vLLM, LM Studio, HF, Remote) provides
the same interface so DeepNetz can switch transparently.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generator, List, Dict, Optional, Any


@dataclass
class BackendInfo:
    """Information about a detected backend."""
    name: str
    version: str
    available: bool
    models_count: int = 0
    details: str = ""


@dataclass
class ModelEntry:
    """A model available in a backend."""
    name: str
    size_mb: int = 0
    family: str = ""
    quant: str = ""
    backend: str = ""
    path: str = ""
    parameters_b: float = 0
    context_length: int = 0


@dataclass
class GenerationConfig:
    """Inference configuration."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.0
    repeat_penalty: float = 1.1
    stream: bool = True


@dataclass
class BackendStats:
    """Runtime statistics from a backend."""
    model_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    context_used: int = 0
    context_max: int = 0
    kv_cache_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class BackendAdapter(ABC):
    """Abstract base class for all inference backends."""

    @abstractmethod
    def detect(self) -> BackendInfo:
        """Check if this backend is available on the system."""
        ...

    @abstractmethod
    def list_models(self) -> List[ModelEntry]:
        """List all models available in this backend."""
        ...

    @abstractmethod
    def pull(self, model_name: str, progress_callback=None) -> str:
        """Download/pull a model. Returns local path or identifier."""
        ...

    @abstractmethod
    def load(self, model_ref: str, n_ctx: int = 4096,
             n_gpu_layers: int = -1, n_threads: int = 0,
             kv_type_k: str = "f16", kv_type_v: str = "f16",
             **kwargs) -> None:
        """Load a model with given configuration."""
        ...

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]],
             config: Optional[GenerationConfig] = None) -> str:
        """Chat completion. Returns full response text."""
        ...

    @abstractmethod
    def stream(self, messages: List[Dict[str, str]],
               config: Optional[GenerationConfig] = None) -> Generator[str, None, None]:
        """Streaming chat completion. Yields tokens."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload current model, free resources."""
        ...

    @abstractmethod
    def stats(self) -> BackendStats:
        """Get current runtime statistics."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'native', 'ollama', 'vllm')."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded."""
        return False
