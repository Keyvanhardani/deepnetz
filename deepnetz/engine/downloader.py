"""
Model downloader — fetch GGUF models from HuggingFace.

Auto-selects the right quantization based on available hardware.
"""

import os
from deepnetz.engine.hardware import detect_hardware


# Common model mappings (name → HuggingFace repo + file patterns)
MODEL_REGISTRY = {
    "qwen3.5-35b": {
        "repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "pattern": "Qwen3.5-35B-A3B-{quant}.gguf",
    },
    "qwen3.5-122b": {
        "repo": "unsloth/Qwen3.5-122B-A10B-GGUF",
        "pattern": "Qwen3.5-122B-A10B-UD-{quant}.gguf",
    },
    "llama-3.3-70b": {
        "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "pattern": "Llama-3.3-70B-Instruct-{quant}.gguf",
    },
    "llama-3.2-3b": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "pattern": "Llama-3.2-3B-Instruct-{quant}.gguf",
    },
    "gemma-3-27b": {
        "repo": "unsloth/gemma-3-27b-it-GGUF",
        "pattern": "gemma-3-27b-it-{quant}.gguf",
    },
    "qwen2.5-3b": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "pattern": "qwen2.5-3b-instruct-{quant_lower}.gguf",
    },
}


def recommend_quant(model_name: str) -> str:
    """Recommend quantization based on available hardware."""
    hw = detect_hardware()
    total_mem = hw.total_vram_mb + hw.ram_mb

    # Rough size estimates for different quants of common models
    if "70b" in model_name.lower() or "122b" in model_name.lower():
        if total_mem > 48000:
            return "Q4_K_M"
        elif total_mem > 32000:
            return "IQ2_M"
        else:
            return "IQ2_XXS"
    elif "35b" in model_name.lower() or "27b" in model_name.lower():
        if total_mem > 32000:
            return "Q4_K_M"
        elif total_mem > 16000:
            return "Q2_K"
        else:
            return "IQ2_M"
    else:  # Small models
        if hw.total_vram_mb > 8000:
            return "Q8_0"
        else:
            return "Q4_K_M"


def download_model(model_name: str, quant: str = "auto",
                   output_dir: str = ".") -> str:
    """Download a model from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    # Resolve model name
    key = model_name.lower().replace(" ", "-")
    if key not in MODEL_REGISTRY:
        # Try as direct HuggingFace repo
        print(f"  Unknown model '{model_name}'. Trying as HuggingFace repo...")
        print(f"  Use format: deepnetz download <repo>/<file>")
        return ""

    entry = MODEL_REGISTRY[key]
    repo = entry["repo"]

    if quant == "auto":
        quant = recommend_quant(model_name)
        print(f"  Auto-selected quantization: {quant}")

    filename = entry["pattern"].format(quant=quant, quant_lower=quant.lower())

    print(f"  Downloading {filename}")
    print(f"  From: {repo}")
    print(f"  To:   {output_dir}/")

    path = hf_hub_download(
        repo,
        filename,
        local_dir=output_dir,
        resume_download=True
    )

    print(f"  Done: {path}")
    return path
