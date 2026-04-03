"""
Local Model Scanner — discovers models from all installed backends.
"""

import os
import glob
import platform
from typing import List
from deepnetz.backends.base import ModelEntry


def scan_all_local_models() -> List[ModelEntry]:
    """Scan all known locations for locally available models."""
    models = []
    seen_paths = set()

    # Scan GGUF files in common directories
    search_dirs = [
        os.path.expanduser("~/models"),
        os.path.expanduser("~/.cache/deepnetz/models"),
    ]
    if platform.system() == "Windows":
        search_dirs.extend(["D:/models", "E:/models", "C:/models",
                           os.path.expanduser("~/Desktop")])
    else:
        search_dirs.extend(["/mnt/d/models", "/mnt/e/models",
                           "/models", "/opt/models"])

    for d in search_dirs:
        if os.path.exists(d):
            for f in glob.glob(os.path.join(d, "**/*.gguf"), recursive=True):
                if f not in seen_paths:
                    seen_paths.add(f)
                    try:
                        size_mb = os.path.getsize(f) // (1024 * 1024)
                    except OSError:
                        size_mb = 0
                    models.append(ModelEntry(
                        name=os.path.basename(f),
                        size_mb=size_mb,
                        backend="native",
                        path=f,
                    ))

    # Scan Ollama models
    ollama_dir = _find_ollama_dir()
    if ollama_dir:
        blob_dir = os.path.join(ollama_dir, "blobs")
        if os.path.exists(blob_dir):
            for f in os.listdir(blob_dir):
                path = os.path.join(blob_dir, f)
                if os.path.isfile(path) and path not in seen_paths:
                    try:
                        size_mb = os.path.getsize(path) // (1024 * 1024)
                    except OSError:
                        size_mb = 0
                    if size_mb > 100:  # skip small config files
                        seen_paths.add(path)
                        models.append(ModelEntry(
                            name=f"ollama-blob-{f[:12]}",
                            size_mb=size_mb,
                            backend="ollama",
                            path=path,
                        ))

    # Scan LM Studio cache
    for lms_dir in [
        os.path.expanduser("~/.cache/lm-studio/models"),
        os.path.expanduser("~/AppData/Roaming/LM Studio/models"),
    ]:
        if os.path.exists(lms_dir):
            for f in glob.glob(os.path.join(lms_dir, "**/*.gguf"), recursive=True):
                if f not in seen_paths:
                    seen_paths.add(f)
                    models.append(ModelEntry(
                        name=os.path.basename(f),
                        size_mb=os.path.getsize(f) // (1024 * 1024),
                        backend="lmstudio",
                        path=f,
                    ))

    return models


def _find_ollama_dir():
    home = os.path.expanduser("~")
    for d in [os.path.join(home, ".ollama", "models"),
              "/usr/share/ollama/.ollama/models"]:
        if os.path.exists(d):
            return d
    return None


# Extended model registry for download recommendations
MODEL_CATALOG = [
    {"name": "Qwen3.5-35B-A3B", "repo": "unsloth/Qwen3.5-35B-A3B-GGUF", "params": "35B", "type": "MoE", "tags": ["chat", "reasoning"], "recommended_quant": {"8gb": "Q4_K_M", "16gb": "Q4_K_M", "32gb": "Q4_K_XL"}},
    {"name": "Qwen3.5-122B-A10B", "repo": "unsloth/Qwen3.5-122B-A10B-GGUF", "params": "122B", "type": "MoE", "tags": ["chat", "reasoning", "large"], "recommended_quant": {"32gb": "IQ2_XXS", "64gb": "Q4_K_M"}},
    {"name": "Llama-3.3-70B", "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF", "params": "70B", "type": "Dense", "tags": ["chat", "code", "reasoning"], "recommended_quant": {"32gb": "IQ2_M", "64gb": "Q4_K_M"}},
    {"name": "Llama-3.2-3B", "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF", "params": "3B", "type": "Dense", "tags": ["chat", "fast"], "recommended_quant": {"8gb": "Q8_0", "16gb": "Q8_0"}},
    {"name": "Gemma-3-27B", "repo": "unsloth/gemma-3-27b-it-GGUF", "params": "27B", "type": "Dense", "tags": ["chat", "multilingual"], "recommended_quant": {"16gb": "Q4_K_M", "32gb": "Q8_0"}},
    {"name": "Qwen2.5-3B", "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF", "params": "3B", "type": "Dense", "tags": ["chat", "fast"], "recommended_quant": {"8gb": "Q8_0"}},
    {"name": "Qwen3-4B", "repo": "unsloth/Qwen3-4B-GGUF", "params": "4B", "type": "Dense", "tags": ["chat", "fast"], "recommended_quant": {"8gb": "Q4_K_M"}},
    {"name": "DeepSeek-R1-14B", "repo": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF", "params": "14B", "type": "Dense", "tags": ["reasoning", "math"], "recommended_quant": {"16gb": "Q4_K_M"}},
    {"name": "Phi-4-14B", "repo": "unsloth/phi-4-GGUF", "params": "14B", "type": "Dense", "tags": ["chat", "code"], "recommended_quant": {"16gb": "Q4_K_M"}},
    {"name": "Mistral-7B", "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "params": "7B", "type": "Dense", "tags": ["chat", "code"], "recommended_quant": {"8gb": "Q4_K_M", "16gb": "Q8_0"}},
]
