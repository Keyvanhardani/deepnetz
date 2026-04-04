"""
Model Manager — handles model lifecycle, loading, unloading, switching.
"""

from typing import Optional, List, Dict
from deepnetz.engine.model import Model
from deepnetz.engine.hardware import detect_hardware
from deepnetz.backends.discovery import discover_backends


class ModelManager:
    """Manages model lifecycle for the API server."""

    def __init__(self, gpu_budget: str = "auto", ram_budget: str = "auto",
                 target_context: int = 4096, cpu_only: bool = False,
                 default_backend: str = "auto"):
        self.gpu_budget = gpu_budget
        self.ram_budget = ram_budget
        self.target_context = target_context
        self.cpu_only = cpu_only
        self.default_backend = default_backend

        self._active_model: Optional[Model] = None
        self._active_model_ref: str = ""
        self.hw = detect_hardware()
        self._backends = discover_backends()

    @property
    def active_model(self) -> Optional[Model]:
        return self._active_model

    @property
    def is_loaded(self) -> bool:
        return self._active_model is not None

    def load_model(self, model_ref: str, backend: str = "",
                   cpu_only: bool = None, target_context: int = 0) -> Model:
        """Load a model. Unloads current model first if any."""
        if self._active_model:
            self.unload_model()

        bk = backend or self.default_backend
        use_cpu = cpu_only if cpu_only is not None else self.cpu_only
        ctx = target_context if target_context > 0 else self.target_context

        model = Model(
            model_ref,
            gpu_budget=self.gpu_budget,
            ram_budget=self.ram_budget,
            target_context=ctx,
            cpu_only=use_cpu,
            backend=bk,
            verbose=False,
        )
        model.load()
        self._active_model = model
        self._active_model_ref = model_ref
        return model

    def unload_model(self):
        """Unload current model."""
        if self._active_model and self._active_model.backend:
            try:
                self._active_model.backend.unload()
            except Exception:
                pass
        self._active_model = None
        self._active_model_ref = ""

    def get_active(self) -> Optional[Model]:
        return self._active_model

    def list_available_models(self) -> List[Dict]:
        """List models from all discovered backends."""
        models = []
        for b in self._backends:
            try:
                for m in b.list_models():
                    models.append({
                        "id": m.name,
                        "backend": m.backend,
                        "size_mb": m.size_mb,
                        "path": getattr(m, 'path', ''),
                    })
            except Exception:
                continue
        return models

    @property
    def backends(self):
        return self._backends

    @property
    def model_ref(self) -> str:
        return self._active_model_ref
