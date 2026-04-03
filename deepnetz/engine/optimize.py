"""
DeepNetz Optimizer — tools for faster inference.

Includes:
    - ik_llama.cpp installation helper (1.3-1.5x faster kernels)
    - Model analysis (layer importance, MoE detection)
    - Optimization recommendations
"""

import os
import subprocess
import shutil
from typing import Dict, List, Optional


def install_ik_llama(cuda: bool = True) -> bool:
    """
    Install ik_llama.cpp (optimized llama.cpp fork) for faster inference.

    Features over stock llama.cpp:
    - Better CUDA kernels for Turing+ GPUs
    - FlashMLA for DeepSeek models
    - Fused RoPE + attention operations
    - ~1.3-1.5x faster generation

    Returns True if successful.
    """
    print("\n  DeepNetz Optimizer — Installing ik_llama.cpp")
    print("  ─────────────────────────────────────────────")

    cmake_args = "-DGGML_CUDA=on" if cuda else ""

    cmd = (
        f'CMAKE_ARGS="{cmake_args}" '
        f'LLAMA_CPP_GIT_URL="https://github.com/ikawrakow/ik_llama.cpp.git" '
        f'pip install llama-cpp-python --force-reinstall --no-cache-dir'
    )

    print(f"  Running: {cmd}")
    print(f"  This will take 5-10 minutes...\n")

    try:
        result = subprocess.run(
            cmd, shell=True, timeout=1200,
            capture_output=False,
        )
        if result.returncode == 0:
            print("\n  ik_llama.cpp installed successfully!")
            print("  Expected speedup: 1.3-1.5x on NVIDIA GPUs")
            return True
        else:
            print("\n  Installation failed. Keeping current llama-cpp-python.")
            return False
    except subprocess.TimeoutExpired:
        print("\n  Installation timed out.")
        return False


def analyze_model(model_path: str) -> Dict:
    """
    Analyze a GGUF model and recommend optimizations.

    Returns optimization report.
    """
    report = {
        "model": os.path.basename(model_path),
        "size_mb": 0,
        "recommendations": [],
    }

    if not os.path.exists(model_path):
        report["error"] = "File not found"
        return report

    report["size_mb"] = os.path.getsize(model_path) // (1024 * 1024)

    # Read GGUF metadata
    try:
        from deepnetz.engine.gguf_reader import gguf_to_model_spec
        spec = gguf_to_model_spec(model_path)
        report["params_b"] = spec.n_params_b
        report["layers"] = spec.n_layers
        report["is_moe"] = spec.is_moe
        report["context"] = spec.context_length
        report["heads"] = {"q": spec.n_heads, "kv": spec.n_kv_heads}
    except Exception:
        report["error"] = "Could not read GGUF metadata"
        return report

    # Recommendations
    recs = report["recommendations"]

    # MoE optimization
    if spec.is_moe:
        from deepnetz.engine.features import recommend_apex_variant
        apex = recommend_apex_variant(spec.name)
        if apex:
            recs.append({
                "type": "apex",
                "priority": "high",
                "description": f"APEX variant available: {apex}",
                "expected_improvement": "50% smaller, same quality",
            })
        recs.append({
            "type": "moe_kv",
            "priority": "medium",
            "description": "MoE model — KV cache compression recommended (q4_0)",
            "expected_improvement": "3.6x less KV cache memory",
        })

    # Large model optimization
    if spec.n_params_b > 30:
        recs.append({
            "type": "speculative",
            "priority": "high",
            "description": "Large model — speculative decoding recommended",
            "expected_improvement": "1.5-2x faster generation",
            "command": f"deepnetz run {model_path} --draft small-model.gguf",
        })

    # GQA optimization
    if spec.n_kv_heads < spec.n_heads:
        gqa_ratio = spec.n_heads // spec.n_kv_heads
        recs.append({
            "type": "gqa",
            "priority": "info",
            "description": f"GQA model ({gqa_ratio}:1) — efficient KV cache",
            "expected_improvement": f"{gqa_ratio}x less KV memory vs MHA",
        })

    # KV cache compression
    if report["size_mb"] > 5000:
        recs.append({
            "type": "kv_quant",
            "priority": "high",
            "description": "Large model — use KV cache quantization (q8_0 or q4_0)",
            "expected_improvement": "1.9x-3.6x less KV memory",
        })

    # ik_llama.cpp
    recs.append({
        "type": "ik_llama",
        "priority": "medium",
        "description": "Install ik_llama.cpp for optimized CUDA kernels",
        "expected_improvement": "1.3-1.5x faster on NVIDIA GPUs",
        "command": "deepnetz optimize --install-ik-llama",
    })

    return report


def print_analysis(report: Dict):
    """Pretty-print model analysis report."""
    print(f"\n  DeepNetz Model Analysis")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Model:      {report.get('model', '?')}")
    print(f"  Size:       {report.get('size_mb', 0)} MB")
    if 'params_b' in report:
        print(f"  Parameters: {report['params_b']:.1f}B {'(MoE)' if report.get('is_moe') else ''}")
        print(f"  Layers:     {report.get('layers', '?')}")
        print(f"  Context:    {report.get('context', '?'):,}")
        h = report.get('heads', {})
        print(f"  Heads:      {h.get('q', '?')} Q / {h.get('kv', '?')} KV")

    recs = report.get("recommendations", [])
    if recs:
        print(f"\n  Optimization Recommendations ({len(recs)})")
        print(f"  ─────────────────────────────────────────────")
        for r in recs:
            prio = {"high": "!!!", "medium": " ! ", "info": "   "}.get(r["priority"], "   ")
            print(f"  {prio} {r['description']}")
            print(f"       → {r['expected_improvement']}")
            if "command" in r:
                print(f"       $ {r['command']}")
            print()
    else:
        print(f"\n  No specific optimizations recommended.")
