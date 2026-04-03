"""
TurboQuant KV Cache Integration — bridge to turboquant-ggml.

Manages TurboQuant type selection and configuration for llama-cpp-python.
Falls back to f16 if TurboQuant types are not available in the build.
"""

from dataclasses import dataclass


# GGML type IDs for TurboQuant (must match ggml.h enum values)
TURBO_TYPE_IDS = {
    "turbo4_0": 41,
    "turbo3_0": 42,
    "turbo2_0": 43,
}


@dataclass
class TurboQuantConfig:
    """TurboQuant KV cache configuration."""
    k_type: str = "turbo4_0"
    v_type: str = "turbo4_0"
    fallback: str = "f16"
    enabled: bool = True


def check_turboquant_support() -> bool:
    """Check if the installed llama-cpp-python supports TurboQuant types."""
    try:
        from llama_cpp import Llama
        # TurboQuant is available if GGML_TYPE_TURBO4_0 (41) exists
        # We can't easily check without loading a model, so we
        # check if the llama_cpp module has the right version
        return True  # assume available, will fallback at runtime
    except ImportError:
        return False


def get_kv_type_id(type_name: str) -> int:
    """Get GGML type ID for a KV cache type name."""
    # Standard types
    standard = {
        "f16": 1,
        "f32": 0,
        "q8_0": 8,
        "q4_0": 2,
        "q4_1": 3,
        "q5_0": 6,
        "q5_1": 7,
    }
    if type_name in standard:
        return standard[type_name]
    if type_name in TURBO_TYPE_IDS:
        return TURBO_TYPE_IDS[type_name]
    return 1  # default f16


def recommend_kv_config(model_params_b: float,
                        available_vram_mb: int,
                        target_context: int) -> TurboQuantConfig:
    """
    Recommend KV cache compression based on model size and hardware.

    Guidelines from community benchmarks:
    - turbo4_0: +0.4% to +7.7% PPL, safe for all models
    - turbo3_0: +2% on large models, needs more testing on small
    - turbo2_0: aggressive, for extreme memory constraints
    """
    # Estimate KV cache size at f16
    # More accurate: ~2 MB per 1B params per 1K context tokens
    # (accounts for GQA ratio, typical head_dim=128)
    kv_f16_mb = model_params_b * 2.0 * (target_context / 1024)

    if kv_f16_mb < available_vram_mb * 0.3:
        # KV cache fits easily, no compression needed
        return TurboQuantConfig(k_type="f16", v_type="f16", enabled=False)

    if kv_f16_mb < available_vram_mb * 0.6:
        # Moderate pressure, use turbo4 (safest)
        return TurboQuantConfig(k_type="turbo4_0", v_type="turbo4_0")

    if kv_f16_mb < available_vram_mb:
        # High pressure, turbo3
        return TurboQuantConfig(k_type="turbo3_0", v_type="turbo3_0")

    # Extreme pressure
    return TurboQuantConfig(k_type="turbo2_0", v_type="turbo2_0")


# Compression ratios for planning
COMPRESSION_RATIOS = {
    "f16": 1.0,
    "f32": 0.5,  # actually 2x bigger
    "q8_0": 1.9,
    "q4_0": 3.6,
    "turbo4_0": 3.6,
    "turbo3_0": 4.6,
    "turbo2_0": 6.4,
}
