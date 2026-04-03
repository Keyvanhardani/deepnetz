"""
Registry Store — local blob storage + HuggingFace as model source.

No hardcoded catalog. HuggingFace IS the catalog.
Users pull any model by name, we search HF, download the GGUF.

Directory layout:
    ~/.cache/deepnetz/registry/
        blobs/              Content-addressed GGUF files
            sha256-<hash>
        state/              Download state per model
            <name>_<quant>.json
"""

import hashlib
import json
import os
import time
from typing import Dict, List, Optional

from deepnetz.registry.config import ModelConfig


class RegistryStore:
    """Local blob store. Source of truth for models = HuggingFace."""

    def __init__(self, base_dir: str = ""):
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), ".cache", "deepnetz")
        self._base = base_dir
        self._blobs_dir = os.path.join(base_dir, "registry", "blobs")
        self._state_dir = os.path.join(base_dir, "registry", "state")

        os.makedirs(self._blobs_dir, exist_ok=True)
        os.makedirs(self._state_dir, exist_ok=True)

    # ── Pull ───────────────────────────────────────────────────────

    def pull(self, query: str, quant: str = "auto") -> str:
        """
        Pull any model. Searches HuggingFace, downloads GGUF.

        Supports:
            pull("Qwen3.5-35B")                → HF search
            pull("unsloth/Qwen3.5-35B-A3B-GGUF") → direct repo
            pull("Qwen3.5-35B", quant="Q8_0")    → specific quant
        """
        # Check if already downloaded
        existing = self.resolve(query, quant if quant != "auto" else "")
        if existing:
            print(f"  Bereits vorhanden: {existing}")
            return existing

        # Determine quant
        if quant == "auto":
            from deepnetz.engine.downloader import recommend_quant
            quant = recommend_quant(query)
            print(f"  Quantisierung: {quant} (automatisch gewählt)")

        # Direct HF repo (contains "/")
        if "/" in query:
            return self._pull_from_repo(query, quant)

        # Search HuggingFace
        return self._search_and_pull(query, quant)

    def _search_and_pull(self, query: str, quant: str) -> str:
        """Search HF for GGUF models, pick best match, download."""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError("pip install huggingface_hub")

        api = HfApi()
        print(f"  Suche auf HuggingFace: {query} ...")

        # Search for GGUF repos
        try:
            results = list(api.list_models(
                search=f"{query} GGUF",
                sort="downloads",
                direction=-1,
                limit=10,
            ))
        except Exception as e:
            raise FileNotFoundError(f"HuggingFace-Suche fehlgeschlagen: {e}")

        if not results:
            raise FileNotFoundError(
                f"Nichts gefunden für '{query}' auf HuggingFace.\n\n"
                f"  Versuche:\n"
                f"    deepnetz pull user/repo-name    Direktes Repo\n"
                f"    deepnetz search {query}         Suche anzeigen"
            )

        # Filter: prefer repos with "gguf" or "GGUF" in name
        gguf_repos = [r for r in results if "gguf" in r.modelId.lower()]
        candidates = gguf_repos if gguf_repos else results

        # Pick best: prefer repos from known uploaders, then by downloads
        preferred_uploaders = {"bartowski", "unsloth", "Qwen", "meta-llama",
                               "google", "mistralai", "microsoft", "deepseek-ai"}
        best = None
        for r in candidates:
            uploader = r.modelId.split("/")[0] if "/" in r.modelId else ""
            if uploader in preferred_uploaders:
                best = r
                break
        if not best:
            best = candidates[0]

        print(f"  Gefunden: {best.modelId}")
        return self._pull_from_repo(best.modelId, quant)

    def _pull_from_repo(self, repo: str, quant: str) -> str:
        """Pull GGUF from a specific HF repo."""
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            raise ImportError("pip install huggingface_hub")

        api = HfApi()

        # List GGUF files in repo
        try:
            all_files = list(api.list_repo_tree(repo, recursive=True))
            gguf_files = [f.rfilename for f in all_files
                          if f.rfilename.endswith(".gguf")]
        except Exception as e:
            raise FileNotFoundError(f"Repo nicht gefunden: {repo}\n  {e}")

        if not gguf_files:
            raise FileNotFoundError(f"Keine GGUF-Dateien in {repo}")

        # Pick file matching requested quant
        target = self._pick_gguf(gguf_files, quant)
        name = _normalize(repo.split("/")[-1])

        # Check if this exact file was already downloaded
        existing = self.get_blob_path(name, quant)
        if existing:
            print(f"  Bereits vorhanden: {existing}")
            return existing

        # Extract file size info
        file_info = next((f for f in all_files
                          if f.rfilename == target), None)
        size_str = ""
        if file_info and hasattr(file_info, 'size') and file_info.size:
            size_gb = file_info.size / (1024 ** 3)
            size_str = f" ({size_gb:.1f} GB)"

        print(f"\n  DeepNetz Pull")
        print(f"  ─────────────────────────────────────")
        print(f"  Repo:    {repo}")
        print(f"  Datei:   {target}{size_str}")
        print(f"  Quant:   {quant}")
        print()

        # Download
        tmp_dir = os.path.join(self._base, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        path = hf_hub_download(
            repo, target,
            local_dir=tmp_dir,
            resume_download=True,
        )
        path = os.path.abspath(path)

        # Store as blob
        blob_path = self.store_blob(name, quant, path)

        size_mb = os.path.getsize(blob_path) / (1024 * 1024)
        s = f"{size_mb / 1024:.1f} GB" if size_mb > 1024 else f"{size_mb:.0f} MB"
        print(f"\n  Fertig: {s}")
        print(f"  Blob:   {blob_path}")
        return blob_path

    def _pick_gguf(self, files: List[str], quant: str) -> str:
        """Pick the best GGUF file matching the requested quantization."""
        q = quant.upper()

        # Exact quant match
        for f in files:
            if q in f.upper():
                return f

        # Fallback priorities
        fallbacks = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0", "Q2_K"]
        for fb in fallbacks:
            for f in files:
                if fb in f.upper():
                    print(f"  {quant} nicht verfügbar, nehme: {fb}")
                    return f

        # Last resort: first file
        print(f"  {quant} nicht verfügbar, nehme: {files[0]}")
        return files[0]

    # ── Search (über Registry Server) ─────────────────────────────

    @staticmethod
    def search(query: str, limit: int = 15) -> List[Dict]:
        """Suche über den DeepNetz Registry Server."""
        try:
            from deepnetz.registry.client import RegistryClient
            client = RegistryClient()
            return client.search(query, limit)
        except Exception:
            return []

    # ── Blob Management ────────────────────────────────────────────

    def get_blob_path(self, name: str, quant: str = "") -> Optional[str]:
        """Get path to downloaded blob. None if not downloaded."""
        state = self._get_state(name, quant)
        if state:
            blob_path = state.get("blob_path", "")
            if blob_path and os.path.exists(blob_path):
                return blob_path
        return None

    def store_blob(self, name: str, quant: str, source_path: str) -> str:
        """Move a downloaded file into the blob store. Returns blob path."""
        digest = _sha256_file(source_path)
        blob_name = f"sha256-{digest}"
        blob_path = os.path.join(self._blobs_dir, blob_name)

        if not os.path.exists(blob_path):
            os.rename(source_path, blob_path)
        elif source_path != blob_path and os.path.exists(source_path):
            os.remove(source_path)

        self._save_state(name, quant, {
            "blob_path": blob_path,
            "digest": digest,
            "size": os.path.getsize(blob_path),
            "downloaded_at": time.time(),
        })
        return blob_path

    def list_local(self) -> List[Dict]:
        """List all locally downloaded models."""
        models = []
        for f in sorted(os.listdir(self._state_dir)):
            if not f.endswith(".json"):
                continue
            try:
                with open(os.path.join(self._state_dir, f)) as fh:
                    state = json.load(fh)
                parts = f[:-5].rsplit("_", 1)
                name = parts[0] if len(parts) > 1 else f[:-5]
                quant = parts[1] if len(parts) > 1 else ""
                blob_path = state.get("blob_path", "")
                models.append({
                    "name": name,
                    "quant": quant,
                    "blob_path": blob_path,
                    "size": state.get("size", 0),
                    "available": os.path.exists(blob_path) if blob_path else False,
                })
            except (json.JSONDecodeError, OSError):
                continue
        return models

    def remove(self, name: str, quant: str = ""):
        """Remove a downloaded model."""
        state = self._get_state(name, quant)
        if state:
            blob_path = state.get("blob_path", "")
            if blob_path and os.path.exists(blob_path):
                os.remove(blob_path)
        key = _normalize(name)
        q = quant or "default"
        state_path = os.path.join(self._state_dir, f"{key}_{q}.json")
        if os.path.exists(state_path):
            os.remove(state_path)

    # ── Resolve (for deepnetz run) ─────────────────────────────────

    def resolve(self, name: str, quant: str = "") -> Optional[str]:
        """Resolve a model name to a local blob path. None if not downloaded."""
        key = _normalize(name)
        # Try exact name+quant
        if quant:
            path = self.get_blob_path(key, quant)
            if path:
                return path
        # Try any quant for this name
        for f in os.listdir(self._state_dir):
            if f.startswith(key + "_") and f.endswith(".json"):
                try:
                    with open(os.path.join(self._state_dir, f)) as fh:
                        state = json.load(fh)
                    blob_path = state.get("blob_path", "")
                    if blob_path and os.path.exists(blob_path):
                        return blob_path
                except (json.JSONDecodeError, OSError):
                    continue
        return None

    # ── Internal ───────────────────────────────────────────────────

    def _get_state(self, name: str, quant: str = "") -> Optional[Dict]:
        key = _normalize(name)
        q = quant or "default"
        path = os.path.join(self._state_dir, f"{key}_{q}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def _save_state(self, name: str, quant: str, state: Dict):
        key = _normalize(name)
        q = quant or "default"
        path = os.path.join(self._state_dir, f"{key}_{q}.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2)


def _normalize(name: str) -> str:
    return name.lower().strip().replace(" ", "-").replace(":", "-").replace("_", "-")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]
