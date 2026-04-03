"""
Output Quality Evaluator — lightweight scoring for generated text.
"""

import math
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class QualityScore:
    """Quality assessment of generated text."""
    repetition_score: float = 1.0    # 1.0 = no repetition, 0.0 = all repeated
    coherence_score: float = 1.0     # 1.0 = coherent, 0.0 = incoherent
    length_score: float = 1.0        # 1.0 = appropriate length
    overall: float = 1.0             # combined score
    details: Dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


def evaluate_output(text: str, prompt: str = "") -> QualityScore:
    """Evaluate the quality of generated text."""
    score = QualityScore()

    if not text or len(text.strip()) == 0:
        return QualityScore(repetition_score=0, coherence_score=0,
                           length_score=0, overall=0,
                           details={"error": "empty output"})

    # Repetition detection (n-gram frequency)
    score.repetition_score = _check_repetition(text)

    # Coherence (sentence structure)
    score.coherence_score = _check_coherence(text)

    # Length appropriateness
    score.length_score = _check_length(text, prompt)

    # Overall
    score.overall = (score.repetition_score * 0.4 +
                    score.coherence_score * 0.3 +
                    score.length_score * 0.3)

    score.details = {
        "text_length": len(text),
        "word_count": len(text.split()),
        "sentence_count": text.count(".") + text.count("!") + text.count("?"),
    }

    return score


def _check_repetition(text: str, n: int = 4) -> float:
    """Check for n-gram repetition. Returns 1.0 (no repetition) to 0.0 (all repeated)."""
    words = text.lower().split()
    if len(words) < n + 1:
        return 1.0

    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i:i+n]))

    total = len(ngrams)
    unique = len(set(ngrams))
    if total == 0:
        return 1.0

    ratio = unique / total
    # Below 0.5 unique ratio is very repetitive
    return min(1.0, ratio * 1.5)


def _check_coherence(text: str) -> float:
    """Basic coherence check — proper sentences, no garbage."""
    # Check for common garbage patterns
    if text.count("?") > len(text) / 10:  # too many question marks
        return 0.3
    if text.count("!") > len(text) / 10:
        return 0.3

    # Check for actual words (not random characters)
    words = text.split()
    if not words:
        return 0.0

    # Average word length should be between 2 and 15
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < 2 or avg_len > 15:
        return 0.5

    # Should have some sentence structure
    has_sentences = any(c in text for c in ".!?")
    return 1.0 if has_sentences else 0.7


def _check_length(text: str, prompt: str) -> float:
    """Check if output length is appropriate for the prompt."""
    words = len(text.split())

    if words < 3:
        return 0.3  # too short
    if words > 2000:
        return 0.7  # very long, might be rambling

    # Short prompts should get short-medium answers
    prompt_words = len(prompt.split()) if prompt else 10
    ratio = words / max(prompt_words, 1)

    if ratio < 0.5:
        return 0.5  # answer shorter than question
    if ratio > 100:
        return 0.7  # extremely long relative to prompt

    return 1.0
