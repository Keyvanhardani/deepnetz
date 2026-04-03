"""
DeepNetz — Run massive models on minimal hardware.

Combines TurboQuant KV cache compression, smart token eviction,
dynamic layer offloading, and hardware auto-tuning into one framework.

https://deepnetz.com
https://github.com/Keyvanhardani/deepnetz
"""

__version__ = "0.1.0"
__author__ = "Keyvan Hardani"

from deepnetz.engine.model import Model

__all__ = ["Model"]
