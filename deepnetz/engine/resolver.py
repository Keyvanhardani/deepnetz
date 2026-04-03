"""
Universal Model Resolver — load models from any source.

Supports:
    ./model.gguf                    → local file
    hf://user/repo/file.gguf        → HuggingFace
    ollama://model:tag              → Ollama local registry
    lmstudio://model                → LM Studio local cache
    https://url/model.gguf          → direct URL download

Also auto-detects models already downloaded by Ollama or LM Studio.
"""

import os
import glob
import platform
from pathlib import Path
from typing import Optional, Tuple


def resolve_model(model_ref: str, output_dir: str = ".") -> str:
    """
    Resolve a model reference to a local GGUF file path.
    Downloads if necessary.
    Returns absolute path to the GGUF file.
    """
    # Local file
    if os.path.exists(model_ref):
        return os.path.abspath(model_ref)

    # Protocol-based
    if "://" in model_ref:
        protocol, path = model_ref.split("://", 1)

        if protocol == "hf":
            return _resolve_huggingface(path, output_dir)
        elif protocol == "ollama":
            return _resolve_ollama(path)
        elif protocol == "lmstudio":
            return _resolve_lmstudio(path)
        elif protocol in ("http", "https"):
            return _resolve_url(model_ref, output_dir)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

    # Try to find in common locations
    found = _search_local(model_ref)
    if found:
        return found

    # Last resort: try as HuggingFace search
    raise FileNotFoundError(
        f"Model not found: {model_ref}\n\n"
        f"Try one of:\n"
        f"  deepnetz run ./path/to/model.gguf\n"
        f"  deepnetz run hf://user/repo/file.gguf\n"
        f"  deepnetz run ollama://qwen3.5:35b\n"
        f"  deepnetz download {model_ref}\n"
    )


def _resolve_huggingface(path: str, output_dir: str) -> str:
    """Download from HuggingFace. Format: user/repo/file.gguf or user/repo"""
    try:
        from huggingface_hub import hf_hub_download, HfApi
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    parts = path.split("/")
    if len(parts) >= 3:
        repo = "/".join(parts[:2])
        filename = "/".join(parts[2:])
    elif len(parts) == 2:
        repo = path
        # Auto-find the best GGUF file
        api = HfApi()
        files = [f.rfilename for f in api.list_repo_tree(repo)
                 if f.rfilename.endswith(".gguf") and "Q4_K_M" in f.rfilename]
        if not files:
            files = [f.rfilename for f in api.list_repo_tree(repo)
                     if f.rfilename.endswith(".gguf")]
        if not files:
            raise FileNotFoundError(f"No GGUF files found in {repo}")
        filename = files[0]
        print(f"  Auto-selected: {filename}")
    else:
        raise ValueError(f"Invalid HuggingFace path: {path}")

    print(f"  Downloading {filename} from {repo}...")
    return hf_hub_download(repo, filename, local_dir=output_dir, resume_download=True)


def _resolve_ollama(tag: str) -> str:
    """Find model in Ollama's local registry."""
    # Ollama stores models in:
    #   Linux:   ~/.ollama/models/
    #   macOS:   ~/.ollama/models/
    #   Windows: C:\Users\<user>\.ollama\models\

    ollama_dir = _get_ollama_dir()
    if not ollama_dir:
        raise FileNotFoundError(
            "Ollama directory not found. Is Ollama installed?\n"
            "Install: https://ollama.ai"
        )

    # Ollama model structure:
    #   models/manifests/registry.ollama.ai/library/<model>/<tag>
    #   models/blobs/sha256-<hash>

    model_name = tag.split(":")[0]
    model_tag = tag.split(":")[1] if ":" in tag else "latest"

    manifest_dir = os.path.join(ollama_dir, "manifests", "registry.ollama.ai",
                                "library", model_name, model_tag)

    if os.path.exists(manifest_dir):
        # Read manifest to find blob hash
        import json
        manifest_file = os.path.join(manifest_dir, "manifest.json") if os.path.isdir(manifest_dir) else manifest_dir
        if os.path.isdir(manifest_dir):
            # Find manifest file
            for f in os.listdir(manifest_dir):
                manifest_file = os.path.join(manifest_dir, f)
                break

        if os.path.exists(manifest_file):
            with open(manifest_file) as f:
                manifest = json.load(f)

            # Find the model layer (largest blob)
            blobs_dir = os.path.join(ollama_dir, "blobs")
            for layer in manifest.get("layers", []):
                digest = layer.get("digest", "").replace(":", "-")
                blob_path = os.path.join(blobs_dir, digest)
                if os.path.exists(blob_path) and layer.get("mediaType", "").endswith("model"):
                    print(f"  Found Ollama model: {blob_path}")
                    return blob_path

    # Fallback: search by name in blobs
    blobs_dir = os.path.join(ollama_dir, "blobs")
    if os.path.exists(blobs_dir):
        # Find largest blob (likely the model)
        blobs = [(f, os.path.getsize(os.path.join(blobs_dir, f)))
                 for f in os.listdir(blobs_dir)]
        blobs.sort(key=lambda x: x[1], reverse=True)
        if blobs:
            print(f"  Warning: Using largest Ollama blob (may not be exact match)")
            return os.path.join(blobs_dir, blobs[0][0])

    raise FileNotFoundError(f"Ollama model not found: {tag}")


def _resolve_lmstudio(name: str) -> str:
    """Find model in LM Studio's local cache."""
    lmstudio_dir = _get_lmstudio_dir()
    if not lmstudio_dir:
        raise FileNotFoundError("LM Studio directory not found.")

    # LM Studio stores models in:
    #   ~/.cache/lm-studio/models/<author>/<model>/

    # Search for matching GGUF files
    pattern = os.path.join(lmstudio_dir, "**", f"*{name}*.gguf")
    matches = glob.glob(pattern, recursive=True)

    if matches:
        # Pick the best match (prefer Q4_K_M)
        for m in matches:
            if "Q4_K_M" in m:
                print(f"  Found LM Studio model: {m}")
                return m
        print(f"  Found LM Studio model: {matches[0]}")
        return matches[0]

    raise FileNotFoundError(f"LM Studio model not found: {name}")


def _resolve_url(url: str, output_dir: str) -> str:
    """Download model from direct URL."""
    import urllib.request
    filename = url.split("/")[-1].split("?")[0]
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"  Already downloaded: {output_path}")
        return output_path

    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, output_path)
    return output_path


def _search_local(name: str) -> Optional[str]:
    """Search for a model in common locations."""
    search_dirs = [
        ".",
        os.path.expanduser("~/models"),
        os.path.expanduser("~/.cache/deepnetz/models"),
    ]

    # Add OS-specific paths
    if platform.system() == "Windows":
        search_dirs.extend([
            "D:/models", "E:/models", "C:/models",
            os.path.expanduser("~/Desktop"),
        ])
    else:
        search_dirs.extend([
            "/mnt/d/models", "/mnt/e/models",
            "/models", "/opt/models",
        ])

    # Also check Ollama and LM Studio dirs
    ollama = _get_ollama_dir()
    if ollama:
        search_dirs.append(ollama)
    lmstudio = _get_lmstudio_dir()
    if lmstudio:
        search_dirs.append(lmstudio)

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        pattern = os.path.join(d, "**", f"*{name}*.gguf")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]

    return None


def _get_ollama_dir() -> Optional[str]:
    """Get Ollama models directory."""
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".ollama", "models"),
        "/usr/share/ollama/.ollama/models",
    ]
    if platform.system() == "Windows":
        candidates.insert(0, os.path.join(home, ".ollama", "models"))

    for d in candidates:
        if os.path.exists(d):
            return d
    return None


def _get_lmstudio_dir() -> Optional[str]:
    """Get LM Studio models directory."""
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".cache", "lm-studio", "models"),
        os.path.join(home, "AppData", "Roaming", "LM Studio", "models"),
    ]
    for d in candidates:
        if os.path.exists(d):
            return d
    return None
