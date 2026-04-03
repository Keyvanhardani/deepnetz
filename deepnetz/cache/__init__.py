"""
DeepNetz Cache — intelligent KV cache management stack.

Components:
- TurboQuant: WHT + Lloyd-Max quantization (3.6x compression)
- Eviction: Attention-sink-aware token pruning (2-4x savings)
- Merging: Merge similar tokens (1.5-2x savings)
- Combined: all three stacked = up to 10x+ memory reduction
"""

from deepnetz.cache.eviction import AttentionSinkEvictor, EvictionConfig
from deepnetz.cache.merging import KVMerger, MergeConfig
from deepnetz.cache.turboquant import TurboQuantConfig, recommend_kv_config

__all__ = [
    "AttentionSinkEvictor", "EvictionConfig",
    "KVMerger", "MergeConfig",
    "TurboQuantConfig", "recommend_kv_config",
]
