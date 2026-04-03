"""
Model Optimizer + Converter — make models smaller, faster, better.

Supports:
    deepnetz optimize model.gguf --preset balanced      # Smart requantization
    deepnetz optimize model.gguf --preset quality        # Best quality, smaller size
    deepnetz optimize model.gguf --preset speed          # Fastest inference
    deepnetz optimize model.gguf --apex                  # APEX MoE optimization
    deepnetz convert hf://user/repo --to gguf --quant Q4_K_M   # HF → GGUF

Optimization strategies:
    - Importance-based quantization (keep critical layers high precision)
    - APEX for MoE (per-expert precision)
    - KV cache optimization recommendation
    - Layer importance analysis
"""

import os
import subprocess
import shutil
from typing import Optional


def convert_model(source: str, output_dir: str = ".",
                  to_format: str = "gguf",
                  quant: str = "Q4_K_M",
                  verbose: bool = True) -> str:
    """
    Convert a model to GGUF format with quantization.

    Steps:
    1. Download from HuggingFace if needed
    2. Convert to GGUF (F16)
    3. Quantize to target precision

    Returns path to output GGUF file.
    """
    if to_format != "gguf":
        raise ValueError(f"Unsupported target format: {to_format}. Only 'gguf' is supported.")

    # Step 1: Resolve source
    model_dir = _resolve_source(source, verbose)

    # Step 2: Find or download convert tool
    convert_script = _find_convert_script()

    # Step 3: Convert to F16 GGUF
    model_name = os.path.basename(model_dir.rstrip("/"))
    f16_path = os.path.join(output_dir, f"{model_name}-F16.gguf")

    if not os.path.exists(f16_path):
        if verbose:
            print(f"\n  Converting to GGUF (F16)...")
            print(f"  Source:  {model_dir}")
            print(f"  Output:  {f16_path}")

        if convert_script:
            _run_convert_script(convert_script, model_dir, f16_path, verbose)
        else:
            _convert_with_gguf_py(model_dir, f16_path, verbose)
    else:
        if verbose:
            print(f"  F16 GGUF already exists: {f16_path}")

    # Step 4: Quantize
    if quant.upper() == "F16":
        if verbose:
            print(f"\n  Done: {f16_path}")
        return f16_path

    quant_path = os.path.join(output_dir, f"{model_name}-{quant}.gguf")

    if os.path.exists(quant_path):
        if verbose:
            print(f"  Quantized GGUF already exists: {quant_path}")
        return quant_path

    if verbose:
        print(f"\n  Quantizing to {quant}...")

    quantize_bin = _find_quantize_bin()
    if quantize_bin:
        _run_quantize(quantize_bin, f16_path, quant_path, quant, verbose)
    else:
        if verbose:
            print(f"  llama-quantize not found. Install llama.cpp for quantization.")
            print(f"  Returning F16 version instead.")
        return f16_path

    if verbose:
        size_mb = os.path.getsize(quant_path) / (1024 * 1024)
        print(f"\n  Done: {quant_path} ({size_mb:.0f} MB)")

    return quant_path


def _resolve_source(source: str, verbose: bool) -> str:
    """Resolve source to a local directory with model files."""
    # Local directory
    if os.path.isdir(source):
        return source

    # HuggingFace repo
    if "/" in source or source.startswith("hf://"):
        repo = source.replace("hf://", "")
        if verbose:
            print(f"  Downloading from HuggingFace: {repo}")
        try:
            from huggingface_hub import snapshot_download
            local = snapshot_download(repo, ignore_patterns=["*.gguf", "*.bin"])
            return local
        except ImportError:
            raise ImportError("pip install huggingface_hub")
        except Exception as e:
            raise FileNotFoundError(f"HuggingFace repo nicht gefunden: {repo}\n  {e}")

    raise FileNotFoundError(f"Source nicht gefunden: {source}")


def _find_convert_script() -> Optional[str]:
    """Find llama.cpp convert script."""
    candidates = [
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
        "/usr/local/bin/convert_hf_to_gguf.py",
        shutil.which("convert_hf_to_gguf.py"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _find_quantize_bin() -> Optional[str]:
    """Find llama-quantize binary."""
    candidates = [
        os.path.expanduser("~/llama.cpp/build/bin/llama-quantize"),
        os.path.expanduser("~/llama.cpp/llama-quantize"),
        shutil.which("llama-quantize"),
        "/usr/local/bin/llama-quantize",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _run_convert_script(script: str, model_dir: str, output: str, verbose: bool):
    """Run llama.cpp convert script."""
    cmd = ["python3", script, model_dir, "--outfile", output, "--outtype", "f16"]
    if verbose:
        print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=not verbose, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed: {result.stderr if not verbose else 'see output'}")


def _convert_with_gguf_py(model_dir: str, output: str, verbose: bool):
    """Convert using gguf-py library directly (fallback)."""
    try:
        # Try using the gguf Python package
        import gguf
        if verbose:
            print(f"  Using gguf-py library for conversion")
        # gguf-py doesn't have a direct HF→GGUF converter
        # Fall back to suggesting manual conversion
        raise ImportError("Direct conversion not available")
    except ImportError:
        raise RuntimeError(
            "Kein Converter gefunden.\n\n"
            "Optionen:\n"
            "  1. llama.cpp installieren: git clone https://github.com/ggml-org/llama.cpp\n"
            "  2. gguf-py installieren: pip install gguf\n"
            "  3. Fertige GGUF von HuggingFace: deepnetz pull <model>"
        )


def _run_quantize(quantize_bin: str, input_path: str, output_path: str,
                  quant_type: str, verbose: bool):
    """Run llama-quantize."""
    cmd = [quantize_bin, input_path, output_path, quant_type]
    if verbose:
        print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=not verbose, text=True, timeout=7200)
    if result.returncode != 0:
        raise RuntimeError(f"Quantization failed: {result.stderr if not verbose else 'see output'}")


# Available quantization types (llama.cpp)
QUANT_TYPES = {
    "F16": "Full precision (16-bit float)",
    "Q8_0": "8-bit (highest quality, 1.9x compression)",
    "Q6_K": "6-bit (very high quality, 2.7x compression)",
    "Q5_K_M": "5-bit medium (high quality, 3.0x compression)",
    "Q4_K_M": "4-bit medium (balanced, 3.6x compression)",
    "Q4_K_S": "4-bit small (good quality, 3.6x compression)",
    "Q3_K_M": "3-bit medium (decent quality, 4.6x compression)",
    "Q2_K": "2-bit (lowest quality, 6.4x compression)",
    "IQ2_M": "2-bit imatrix (better quality, 6.4x)",
    "IQ2_XXS": "2-bit imatrix extra small (7.5x)",
}
