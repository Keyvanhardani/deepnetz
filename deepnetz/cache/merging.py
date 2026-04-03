"""
KV Merging — merge similar tokens instead of hard eviction.

Based on CaM (Cache Merging) and D2O:
- CaM: merge low-importance KV pairs into neighboring important ones
- D2O: distinguish active/passive tokens, merge passive ones

Preserves more information than eviction at same memory budget.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math


@dataclass
class MergeConfig:
    """Configuration for KV merging."""
    strategy: str = "weighted"     # "weighted", "average", "importance"
    similarity_threshold: float = 0.85  # merge tokens with cosine sim > this
    max_merges_per_step: int = 32
    protect_recent: int = 128      # don't merge most recent N tokens
    protect_sinks: int = 4         # don't merge first N tokens


class KVMerger:
    """
    Merges similar KV cache entries to reduce memory without hard eviction.

    Strategy: find pairs of adjacent tokens with high cosine similarity
    in their key vectors, merge their values by weighted average.
    """

    def __init__(self, config: MergeConfig):
        self.config = config
        self.total_merged = 0

    def find_merge_candidates(self, key_vectors: List[List[float]],
                               importance_scores: Optional[List[float]] = None
                               ) -> List[Tuple[int, int, float]]:
        """
        Find pairs of tokens suitable for merging.
        Returns list of (idx_a, idx_b, similarity) sorted by similarity.
        """
        candidates = []
        n = len(key_vectors)
        protect_start = self.config.protect_sinks
        protect_end = n - self.config.protect_recent

        if protect_end <= protect_start:
            return []

        for i in range(protect_start, protect_end - 1):
            sim = self._cosine_similarity(key_vectors[i], key_vectors[i + 1])
            if sim >= self.config.similarity_threshold:
                candidates.append((i, i + 1, sim))

        # Sort by highest similarity first
        candidates.sort(key=lambda x: -x[2])
        return candidates[:self.config.max_merges_per_step]

    def merge_entries(self, values_a: List[float], values_b: List[float],
                      weight_a: float = 0.5, weight_b: float = 0.5
                      ) -> List[float]:
        """Weighted merge of two value vectors."""
        total = weight_a + weight_b
        if total < 1e-12:
            total = 1.0
        return [(a * weight_a + b * weight_b) / total
                for a, b in zip(values_a, values_b)]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def stats(self) -> dict:
        return {
            "strategy": self.config.strategy,
            "total_merged": self.total_merged,
            "similarity_threshold": self.config.similarity_threshold,
        }
