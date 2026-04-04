"""
Model Card Generator — auto-generate model cards from HuggingFace.

Generates structured model cards for 200+ popular LLMs.
Cards are cached locally and served via the registry server.

Usage:
    from deepnetz.engine.cards import generate_all_cards, search_cards, load_cards

    # Generate cards (admin/one-time)
    cards = generate_all_cards(hf_token="hf_...")

    # Search cached cards
    results = search_cards("qwen code", cards)
"""

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class QuantVariant:
    name: str          # "Q4_K_M"
    size_mb: int = 0
    repo: str = ""
    filename: str = ""


@dataclass
class ModelCard:
    id: str
    name: str
    family: str = ""
    params_b: float = 0
    active_params_b: float = 0
    architecture: str = "Dense Transformer"
    license: str = ""
    context_length: int = 4096
    quants: List[Dict] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    hf_repos: List[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelCard":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Popular Model Families ────────────────────────────────────────────

POPULAR_MODELS = [
    # Qwen family
    {"search": "Qwen3.5 GGUF", "family": "qwen", "prefix": "Qwen3.5"},
    {"search": "Qwen3 GGUF", "family": "qwen", "prefix": "Qwen3"},
    {"search": "Qwen2.5 Instruct GGUF", "family": "qwen", "prefix": "Qwen2.5"},
    {"search": "Qwen2.5-Coder GGUF", "family": "qwen", "prefix": "Qwen2.5-Coder"},
    {"search": "QwQ GGUF", "family": "qwen", "prefix": "QwQ"},
    # Llama family
    {"search": "Llama-3.3 GGUF", "family": "llama", "prefix": "Llama-3.3"},
    {"search": "Llama-3.2 GGUF", "family": "llama", "prefix": "Llama-3.2"},
    {"search": "Llama-3.1 Instruct GGUF", "family": "llama", "prefix": "Llama-3.1"},
    {"search": "Llama-4 GGUF", "family": "llama", "prefix": "Llama-4"},
    # Gemma family
    {"search": "gemma-4 GGUF", "family": "gemma", "prefix": "Gemma-4"},
    {"search": "gemma-3 GGUF", "family": "gemma", "prefix": "Gemma-3"},
    {"search": "gemma-3n GGUF", "family": "gemma", "prefix": "Gemma-3n"},
    # DeepSeek
    {"search": "DeepSeek-R1 GGUF", "family": "deepseek", "prefix": "DeepSeek-R1"},
    {"search": "DeepSeek-V3 GGUF", "family": "deepseek", "prefix": "DeepSeek-V3"},
    {"search": "DeepSeek-Coder GGUF", "family": "deepseek", "prefix": "DeepSeek-Coder"},
    # Mistral
    {"search": "Mistral Instruct GGUF", "family": "mistral", "prefix": "Mistral"},
    {"search": "Mixtral GGUF", "family": "mistral", "prefix": "Mixtral"},
    {"search": "Mistral-Small GGUF", "family": "mistral", "prefix": "Mistral-Small"},
    # Phi
    {"search": "phi-4 GGUF", "family": "phi", "prefix": "Phi-4"},
    {"search": "Phi-3.5 GGUF", "family": "phi", "prefix": "Phi-3.5"},
    {"search": "Phi-4-mini GGUF", "family": "phi", "prefix": "Phi-4-mini"},
    # Command-R
    {"search": "command-r GGUF", "family": "cohere", "prefix": "Command-R"},
    # StarCoder
    {"search": "starcoder2 GGUF", "family": "starcoder", "prefix": "StarCoder2"},
    # Yi
    {"search": "Yi-1.5 GGUF", "family": "yi", "prefix": "Yi-1.5"},
    # InternLM
    {"search": "InternLM GGUF", "family": "internlm", "prefix": "InternLM"},
    # Falcon
    {"search": "Falcon GGUF", "family": "falcon", "prefix": "Falcon"},
    # Vision models
    {"search": "Qwen3-VL GGUF", "family": "qwen", "prefix": "Qwen3-VL"},
    {"search": "Qwen2.5-VL GGUF", "family": "qwen", "prefix": "Qwen2.5-VL"},
    {"search": "LLaVA GGUF", "family": "llava", "prefix": "LLaVA"},
    {"search": "MiniCPM-V GGUF", "family": "minicpm", "prefix": "MiniCPM-V"},
    {"search": "Pixtral GGUF", "family": "mistral", "prefix": "Pixtral"},
    # Code models
    {"search": "CodeLlama GGUF", "family": "llama", "prefix": "CodeLlama"},
    {"search": "CodeGemma GGUF", "family": "gemma", "prefix": "CodeGemma"},
    # NVIDIA
    {"search": "Nemotron GGUF", "family": "nvidia", "prefix": "Nemotron"},
    # Smaller
    {"search": "SmolLM GGUF", "family": "huggingface", "prefix": "SmolLM"},
    {"search": "TinyLlama GGUF", "family": "tinyllama", "prefix": "TinyLlama"},
]

PREFERRED_UPLOADERS = {"bartowski", "unsloth", "Qwen", "meta-llama", "google",
                       "mistralai", "microsoft", "deepseek-ai", "lmstudio-community",
                       "ggml-org", "TheBloke", "mudler"}

QUANT_NAMES = ["Q2_K", "Q3_K_M", "Q3_K_S", "Q4_K_M", "Q4_K_S", "Q4_0",
               "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "IQ2_M", "IQ2_XXS",
               "IQ3_M", "IQ4_XS", "F16"]


# ── Card Generation ──────────────────────────────────────────────────

def generate_card_from_repo(repo_id: str, family: str = "",
                            hf_token: str = "") -> Optional[ModelCard]:
    """Generate a ModelCard from a HuggingFace GGUF repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token if hf_token else None)

        info = api.model_info(repo_id)
        files = list(api.list_repo_tree(repo_id, recursive=True))
    except Exception:
        return None

    # Find GGUF files
    gguf_files = [f for f in files if hasattr(f, 'rfilename') and f.rfilename.endswith(".gguf")]
    if not gguf_files:
        return None

    # Extract quant variants
    quants = []
    for f in gguf_files:
        name = f.rfilename
        size_mb = getattr(f, 'size', 0) // (1024 * 1024) if getattr(f, 'size', 0) else 0
        quant_name = _extract_quant(name)
        if quant_name:
            quants.append({
                "name": quant_name,
                "size_mb": size_mb,
                "repo": repo_id,
                "filename": name,
            })

    if not quants:
        return None

    # Sort by size
    quants.sort(key=lambda q: q["size_mb"])

    # Extract model info from repo name
    model_name = repo_id.split("/")[-1].replace("-GGUF", "").replace("-gguf", "")
    model_id = model_name.lower().replace(" ", "-")

    # Detect params from name
    params_b = _extract_params(model_name)
    active_b = params_b

    # Detect architecture
    arch = "Dense Transformer"
    tags = []
    name_lower = model_name.lower()

    if any(x in name_lower for x in ["a3b", "a4b", "a10b", "a22b", "moe", "mixtral"]):
        arch = "MoE Transformer"
        active_b = _extract_active_params(model_name) or params_b
        tags.append("moe")

    if any(x in name_lower for x in ["vl", "vision", "llava", "minicpm-v", "pixtral"]):
        tags.append("vision")
    if any(x in name_lower for x in ["coder", "code", "starcoder"]):
        tags.append("code")
    if any(x in name_lower for x in ["r1", "reasoning", "qwq", "think"]):
        tags.append("reasoning")
    if any(x in name_lower for x in ["instruct", "chat", "it"]):
        tags.append("chat")

    # Get metadata
    model_tags = getattr(info, 'tags', []) or []
    license_tag = ""
    for t in model_tags:
        if t.startswith("license:"):
            license_tag = t.replace("license:", "")
            break

    context = 4096
    for t in model_tags:
        if "128k" in t or "131072" in t: context = 131072
        elif "64k" in t or "65536" in t: context = 65536
        elif "32k" in t or "32768" in t: context = 32768
        elif "8k" in t or "8192" in t: context = 8192

    return ModelCard(
        id=model_id,
        name=model_name,
        family=family,
        params_b=params_b,
        active_params_b=active_b,
        architecture=arch,
        license=license_tag,
        context_length=context,
        quants=quants,
        tags=tags,
        downloads=getattr(info, 'downloads', 0) or 0,
        likes=getattr(info, 'likes', 0) or 0,
        hf_repos=[repo_id],
        updated_at=str(getattr(info, 'lastModified', '')),
    )


def generate_all_cards(hf_token: str = "", output_dir: str = "",
                       verbose: bool = True) -> List[ModelCard]:
    """Generate model cards for all popular models."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    if not output_dir:
        output_dir = os.path.join(os.path.expanduser("~"), ".cache", "deepnetz", "cards")
    os.makedirs(output_dir, exist_ok=True)

    api = HfApi(token=hf_token if hf_token else None)
    cards = []
    seen_ids = set()

    if verbose:
        print(f"\n  DeepNetz Model Card Generator")
        print(f"  {'─' * 50}")

    for entry in POPULAR_MODELS:
        search_q = entry["search"]
        family = entry["family"]

        if verbose:
            print(f"  Searching: {search_q}...", end=" ", flush=True)

        try:
            results = list(api.list_models(search=search_q, sort="downloads", limit=8))
        except Exception:
            if verbose:
                print("FAILED")
            continue

        count = 0
        for r in results:
            repo = r.modelId
            uploader = repo.split("/")[0] if "/" in repo else ""

            # Prefer known uploaders
            if uploader not in PREFERRED_UPLOADERS and count > 0:
                continue

            # Skip if we already have this model
            base_id = repo.split("/")[-1].lower().replace("-gguf", "").replace(" ", "-")
            if base_id in seen_ids:
                continue

            card = generate_card_from_repo(repo, family=family, hf_token=hf_token)
            if card and card.quants:
                seen_ids.add(card.id)
                cards.append(card)
                count += 1

                # Save to file
                card_path = os.path.join(output_dir, f"{card.id}.json")
                with open(card_path, "w") as f:
                    json.dump(card.to_dict(), f, indent=2)

        if verbose:
            print(f"{count} cards")

    if verbose:
        print(f"\n  Total: {len(cards)} model cards generated")
        print(f"  Saved to: {output_dir}\n")

    return cards


# ── Card Loading & Search ────────────────────────────────────────────

def load_cards(cards_dir: str = "") -> Dict[str, ModelCard]:
    """Load all cached model cards from disk."""
    if not cards_dir:
        cards_dir = os.path.join(os.path.expanduser("~"), ".cache", "deepnetz", "cards")

    cards = {}
    if not os.path.exists(cards_dir):
        return cards

    for f in os.listdir(cards_dir):
        if f.endswith(".json"):
            try:
                with open(os.path.join(cards_dir, f)) as fh:
                    data = json.load(fh)
                card = ModelCard.from_dict(data)
                cards[card.id] = card
            except (json.JSONDecodeError, OSError):
                continue

    return cards


def search_cards(query: str, cards: Dict[str, ModelCard] = None,
                 tags: List[str] = None) -> List[ModelCard]:
    """Search model cards by name, family, or tags."""
    if cards is None:
        cards = load_cards()

    q = query.lower()
    words = q.split()
    results = []

    for card in cards.values():
        score = 0
        searchable = f"{card.name} {card.family} {' '.join(card.tags)} {card.architecture}".lower()

        # All words must match somewhere
        if words and not all(w in searchable for w in words):
            if not any(w in searchable for w in words):
                # No match at all
                pass
            else:
                # Partial match
                score += sum(3 for w in words if w in searchable)
        elif words:
            score += 10  # All words matched

        name_lower = card.name.lower()
        family_lower = card.family.lower()

        # Exact name match bonus
        if q in name_lower:
            score += 10
        if q == name_lower:
            score += 20

        # Family match bonus
        if q in family_lower:
            score += 5

        # Tag match bonus
        if q in " ".join(card.tags):
            score += 3

        # Tag filter
        if tags:
            if not all(t in card.tags for t in tags):
                continue

        if score > 0:
            results.append((score, card))

    results.sort(key=lambda x: (-x[0], -x[1].downloads))
    return [r[1] for r in results]


def recommend_quant(card: ModelCard, vram_mb: int = 0,
                    ram_mb: int = 0) -> Optional[Dict]:
    """Recommend the best quantization variant for given hardware."""
    if not card.quants:
        return None

    total_mem = vram_mb + ram_mb
    if total_mem == 0:
        # Default recommendation
        for q in card.quants:
            if q["name"] == "Q4_K_M":
                return q
        return card.quants[len(card.quants) // 2]  # Middle quant

    # Find the largest quant that fits in memory (with 20% headroom)
    budget = total_mem * 0.8
    best = None
    for q in reversed(card.quants):  # Largest first
        if q["size_mb"] < budget:
            best = q
            break

    return best or card.quants[0]  # Smallest if nothing fits


# ── Helpers ──────────────────────────────────────────────────────────

def _extract_quant(filename: str) -> str:
    """Extract quantization name from GGUF filename."""
    for q in QUANT_NAMES:
        if q.upper() in filename.upper():
            return q
    # Check for UD (Ultra Dynamic) quants
    if "UD-" in filename:
        for q in QUANT_NAMES:
            if q.upper() in filename.upper():
                return f"UD-{q}"
    return ""


def _extract_params(name: str) -> float:
    """Extract parameter count from model name."""
    match = re.search(r'(\d+\.?\d*)[Bb]', name)
    if match:
        return float(match.group(1))
    return 0


def _extract_active_params(name: str) -> float:
    """Extract active parameters for MoE models (e.g. A3B → 3.0)."""
    match = re.search(r'A(\d+\.?\d*)[Bb]', name)
    if match:
        return float(match.group(1))
    return 0
