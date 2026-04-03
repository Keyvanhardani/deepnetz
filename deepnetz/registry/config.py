"""
Model Config — JSON-based model definition (like Ollama Modelfiles).

A config defines:
  - Where the model comes from (HF repo, URL, etc.)
  - Available quantizations and their filenames
  - Model metadata (family, params, context length, tags)
  - Chat template and stop tokens

Example config:
{
  "name": "qwen3.5-35b",
  "family": "qwen",
  "params": "35B (MoE, 3B active)",
  "source": {
    "type": "huggingface",
    "repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
    "files": {
      "Q4_K_M": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
      "Q2_K": "Qwen3.5-35B-A3B-Q2_K.gguf"
    }
  },
  "default_quant": "Q4_K_M",
  "context_length": 32768,
  "tags": ["chat", "reasoning"]
}
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """A model definition — describes where to get a model and how to use it."""
    name: str
    family: str = ""
    params: str = ""
    source_type: str = "huggingface"  # huggingface, url, local
    source_repo: str = ""
    source_files: Dict[str, str] = field(default_factory=dict)  # quant → filename
    default_quant: str = "Q4_K_M"
    context_length: int = 4096
    tags: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "params": self.params,
            "source": {
                "type": self.source_type,
                "repo": self.source_repo,
                "files": self.source_files,
            },
            "default_quant": self.default_quant,
            "context_length": self.context_length,
            "tags": self.tags,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        source = data.get("source", {})
        return cls(
            name=data.get("name", ""),
            family=data.get("family", ""),
            params=data.get("params", ""),
            source_type=source.get("type", "huggingface"),
            source_repo=source.get("repo", ""),
            source_files=source.get("files", {}),
            default_quant=data.get("default_quant", "Q4_K_M"),
            context_length=data.get("context_length", 4096),
            tags=data.get("tags", []),
            description=data.get("description", ""),
        )

    @classmethod
    def from_file(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_filename(self, quant: str = "") -> str:
        """Get the filename for a specific quantization."""
        q = quant or self.default_quant
        if q in self.source_files:
            return self.source_files[q]
        # Try case-insensitive
        for k, v in self.source_files.items():
            if k.lower() == q.lower():
                return v
        # Fallback: first available
        if self.source_files:
            return next(iter(self.source_files.values()))
        return ""

    def available_quants(self) -> List[str]:
        """List available quantizations."""
        return list(self.source_files.keys())
