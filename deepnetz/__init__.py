"""
DeepNetz — Run massive models on minimal hardware.

Combines TurboQuant KV cache compression, smart token eviction,
dynamic layer offloading, and hardware auto-tuning into one framework.

https://deepnetz.com
https://github.com/Keyvanhardani/deepnetz
"""

__version__ = "1.2.1"
__author__ = "Keyvan Hardani"

def __getattr__(name):
    if name == "Model":
        from deepnetz.engine.model import Model
        return Model
    raise AttributeError(f"module 'deepnetz' has no attribute {name}")

__all__ = ["Model"]
