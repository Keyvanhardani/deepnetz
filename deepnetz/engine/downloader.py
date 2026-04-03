"""
Model Downloader — thin wrapper around RegistryStore.
"""

from typing import Optional, List, Dict
from deepnetz.engine.hardware import detect_hardware


def recommend_quant(model_name: str) -> str:
    """Recommend quantization based on available hardware."""
    hw = detect_hardware()
    total_mem = hw.total_vram_mb + hw.ram_mb
    name = model_name.lower()

    if any(x in name for x in ["70b", "72b", "122b", "235b", "671b"]):
        if total_mem > 48000: return "Q4_K_M"
        elif total_mem > 32000: return "IQ2_M"
        else: return "IQ2_XXS"
    elif any(x in name for x in ["24b", "27b", "30b", "32b", "35b"]):
        if total_mem > 32000: return "Q4_K_M"
        elif total_mem > 16000: return "Q2_K"
        else: return "IQ2_M"
    elif any(x in name for x in ["12b", "14b"]):
        return "Q4_K_M" if total_mem > 16000 else "Q2_K"
    else:
        return "Q8_0" if hw.total_vram_mb > 8000 else "Q4_K_M"


def pull_model(model_name: str, quant: str = "auto") -> str:
    from deepnetz.registry.store import RegistryStore
    return RegistryStore().pull(model_name, quant)


def resolve_local_model(name: str) -> Optional[str]:
    from deepnetz.registry.store import RegistryStore
    return RegistryStore().resolve(name)


def list_local_models() -> List[Dict]:
    from deepnetz.registry.store import RegistryStore
    return RegistryStore().list_local()


def search_models(query: str) -> List[Dict]:
    """Search via registry server, fallback to HF."""
    from deepnetz.registry.store import RegistryStore
    return RegistryStore.search(query)


def _find_in_catalog(name: str) -> Optional[Dict]:
    """Check if model exists locally."""
    from deepnetz.registry.store import RegistryStore
    path = RegistryStore().resolve(name)
    return {"name": name, "path": path} if path else None


# Legacy
def download_model(model_name: str, quant: str = "auto", output_dir: str = ".") -> str:
    return pull_model(model_name, quant=quant)
