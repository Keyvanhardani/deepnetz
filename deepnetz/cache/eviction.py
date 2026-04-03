"""
Token Eviction — attention-sink-aware KV cache management.

Based on StreamingLLM: keep first N tokens (attention sinks) +
most recent M tokens, evict middle tokens by lowest importance.

Works with llama-cpp-python's kv_cache API.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvictionConfig:
    """Configuration for token eviction."""
    strategy: str = "attention_sink"  # "attention_sink", "sliding_window", "scored"
    sink_tokens: int = 4              # first N tokens always kept
    window_size: int = 512            # recent tokens to keep
    max_cache_tokens: int = 2048      # total cache budget
    eviction_batch: int = 64          # evict this many at once


class AttentionSinkEvictor:
    """
    Implements StreamingLLM-style eviction:
    Keep first K tokens (attention sinks) + last W tokens (recent window).
    Evict everything in between when cache is full.
    """

    def __init__(self, config: EvictionConfig):
        self.config = config
        self.total_tokens = 0
        self.evicted_count = 0

    def should_evict(self, current_tokens: int) -> bool:
        """Check if eviction is needed."""
        return current_tokens >= self.config.max_cache_tokens

    def compute_eviction_range(self, current_tokens: int) -> Optional[tuple]:
        """
        Returns (start, end) range of token positions to evict.
        Returns None if no eviction needed.
        """
        if not self.should_evict(current_tokens):
            return None

        # Keep: [0..sink_tokens) + [current - window_size..current)
        # Evict: [sink_tokens..current - window_size)
        start = self.config.sink_tokens
        end = current_tokens - self.config.window_size

        if end <= start:
            return None  # not enough tokens to evict

        # Evict in batches
        evict_end = min(start + self.config.eviction_batch, end)
        self.evicted_count += (evict_end - start)
        return (start, evict_end)

    def apply_eviction(self, llm, seq_id: int = 0) -> int:
        """
        Apply eviction to a llama-cpp-python Llama instance.
        Returns number of tokens evicted.
        """
        try:
            n_tokens = llm.n_tokens
            eviction = self.compute_eviction_range(n_tokens)
            if eviction is None:
                return 0

            start, end = eviction
            n_evict = end - start

            # Remove tokens from KV cache
            llm._ctx.kv_cache_seq_rm(seq_id, start, end)

            # Shift remaining tokens to fill the gap
            llm._ctx.kv_cache_seq_add(seq_id, end, n_tokens, -n_evict)

            self.total_tokens = n_tokens - n_evict
            return n_evict

        except (AttributeError, Exception):
            return 0  # llama-cpp-python version doesn't support KV cache ops

    @property
    def stats(self) -> dict:
        return {
            "strategy": self.config.strategy,
            "total_evicted": self.evicted_count,
            "sink_tokens": self.config.sink_tokens,
            "window_size": self.config.window_size,
            "max_cache": self.config.max_cache_tokens,
        }


class ScoredEvictor:
    """
    Score-based eviction: evict tokens with lowest cumulative attention.
    More precise than sliding window but requires attention score tracking.
    """

    def __init__(self, config: EvictionConfig):
        self.config = config
        self.scores: List[float] = []
        self.evicted_count = 0

    def update_scores(self, attention_weights: List[float]):
        """Accumulate attention scores per token position."""
        for i, w in enumerate(attention_weights):
            if i < len(self.scores):
                self.scores[i] += w
            else:
                self.scores.append(w)

    def find_lowest_scored(self, n_evict: int) -> List[int]:
        """Find positions of lowest-scored tokens (excluding sinks)."""
        sink = self.config.sink_tokens
        recent = len(self.scores) - self.config.window_size

        # Only consider middle tokens
        candidates = [(score, pos) for pos, score in enumerate(self.scores)
                       if sink <= pos < recent]
        candidates.sort()  # lowest first
        return [pos for _, pos in candidates[:n_evict]]

    @property
    def stats(self) -> dict:
        return {
            "strategy": "scored",
            "total_evicted": self.evicted_count,
            "tracked_positions": len(self.scores),
        }
