"""
DeepNetz Model Registry — Ollama-style model management.

Models are defined by JSON config files (Modelfiles).
Blobs (GGUF weights) are content-addressed in a local store.
Configs can be shipped with the package, fetched from a remote catalog,
or created by users.
"""

from deepnetz.registry.store import RegistryStore
from deepnetz.registry.config import ModelConfig

__all__ = ["RegistryStore", "ModelConfig"]
