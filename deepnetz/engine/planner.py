"""
Budget planner — decides how to fit a model on available hardware.

Given a model's requirements and hardware profile, computes:
- How many layers to offload to GPU
- Which KV cache compression to use
- Optimal context length
- Whether token eviction is needed
"""

from dataclasses import dataclass
from typing import Optional
from deepnetz.engine.hardware import HardwareProfile


@dataclass
class ModelSpec:
    """Model specifications extracted from GGUF metadata."""
    name: str
    file_size_mb: int
    n_params_b: float
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    context_length: int
    is_moe: bool = False
    n_active_params_b: Optional[float] = None


@dataclass
class InferencePlan:
    """Computed plan for running a model on given hardware."""
    # Layer placement
    n_gpu_layers: int
    n_cpu_layers: int

    # KV cache config
    kv_type_k: str  # "f16", "turbo4_0", "turbo3_0", "turbo2_0"
    kv_type_v: str
    kv_cache_mb: float

    # Context
    max_context: int
    use_eviction: bool
    eviction_strategy: Optional[str]  # "attention_sink", "paged", None

    # Estimates
    est_prompt_tps: float
    est_gen_tps: float
    est_total_memory_mb: float

    # Warnings
    warnings: list


def plan_inference(model: ModelSpec, hw: HardwareProfile,
                   target_context: int = 4096,
                   gpu_budget_mb: Optional[int] = None,
                   ram_budget_mb: Optional[int] = None) -> InferencePlan:
    """
    Compute optimal inference plan for model on hardware.

    Strategy:
    1. Estimate model weight memory
    2. Estimate KV cache at target context
    3. Try to fit as many layers on GPU as possible
    4. Apply KV compression if needed
    5. Enable eviction if context still too large
    """
    warnings = []

    # Check for MoE + APEX recommendation
    from deepnetz.engine.features import is_moe_model, recommend_apex_variant
    if is_moe_model(model.name):
        apex_repo = recommend_apex_variant(model.name)
        if apex_repo:
            warnings.append(f"MoE model detected. APEX variant available: deepnetz pull {apex_repo}")

    gpu_budget = gpu_budget_mb or (hw.total_vram_mb - 500)  # 500MB overhead
    ram_budget = ram_budget_mb or (hw.ram_mb - 4096)  # 4GB for OS

    # Model weights memory (already quantized in GGUF)
    weight_mb = model.file_size_mb

    # KV cache per token per layer (in bytes)
    # 2 (K+V) * n_kv_heads * head_dim * bytes_per_element
    kv_bytes_per_token_per_layer_f16 = 2 * model.n_kv_heads * model.head_dim * 2  # fp16
    kv_total_f16_mb = (kv_bytes_per_token_per_layer_f16 * model.n_layers *
                       target_context) / (1024 * 1024)

    # Try compression levels (standard GGML types — works with any llama-cpp-python)
    kv_configs = [
        ("f16", "f16", 1.0),       # No compression
        ("q8_0", "q8_0", 1.0 / 1.9),  # 1.9x compression, minimal quality loss
        ("q4_0", "q4_0", 1.0 / 3.6),  # 3.6x compression, good for large models
        ("q4_1", "q4_1", 1.0 / 3.4),  # Slightly better quality than q4_0
    ]

    best_plan = None

    for kv_k, kv_v, kv_ratio in kv_configs:
        kv_mb = kv_total_f16_mb * kv_ratio

        # How many layers fit on GPU?
        available_gpu = gpu_budget - kv_mb if hw.has_cuda else 0
        weight_per_layer = weight_mb / max(model.n_layers, 1)

        if available_gpu > 0:
            n_gpu = min(model.n_layers, int(available_gpu / weight_per_layer))
        else:
            n_gpu = 0

        n_cpu = model.n_layers - n_gpu

        total_memory = weight_mb + kv_mb
        fits_ram = total_memory < ram_budget

        if not fits_ram:
            continue

        # Speed estimates (very rough)
        if n_gpu > 0:
            gpu_fraction = n_gpu / max(model.n_layers, 1)
            est_gen = 5.0 * gpu_fraction + 0.5 * (1 - gpu_fraction)
        else:
            est_gen = 0.5 if weight_mb > 15000 else 2.0

        # Check if eviction needed
        use_eviction = False
        eviction_strategy = None
        actual_context = target_context

        if kv_mb > ram_budget * 0.3:  # KV uses > 30% of RAM
            use_eviction = True
            eviction_strategy = "attention_sink"
            kv_mb *= 0.5  # eviction halves effective KV
            warnings.append(f"Token eviction enabled (KV cache would use {kv_total_f16_mb * kv_ratio:.0f} MB)")

        plan = InferencePlan(
            n_gpu_layers=n_gpu,
            n_cpu_layers=n_cpu,
            kv_type_k=kv_k,
            kv_type_v=kv_v,
            kv_cache_mb=kv_mb,
            max_context=actual_context,
            use_eviction=use_eviction,
            eviction_strategy=eviction_strategy,
            est_prompt_tps=est_gen * 10,
            est_gen_tps=est_gen,
            est_total_memory_mb=total_memory,
            warnings=warnings.copy()
        )

        if kv_k == "f16":
            best_plan = plan  # prefer f16 if it fits
            break
        elif best_plan is None:
            best_plan = plan
            if kv_k == "q8_0":
                break  # q8_0 is good enough for most cases

    if best_plan is None:
        # Nothing fits — use most aggressive compression
        best_plan = InferencePlan(
            n_gpu_layers=0,
            n_cpu_layers=model.n_layers,
            kv_type_k="q4_0",
            kv_type_v="q4_0",
            kv_cache_mb=kv_total_f16_mb / 3.6,
            max_context=min(target_context, 2048),
            use_eviction=True,
            eviction_strategy="attention_sink",
            est_prompt_tps=1.0,
            est_gen_tps=0.3,
            est_total_memory_mb=weight_mb + kv_total_f16_mb / 6.4,
            warnings=["Model barely fits. Expect slow inference.",
                      "Context reduced to 2048."]
        )

    return best_plan


def print_plan(plan: InferencePlan, model: ModelSpec):
    """Pretty-print the inference plan."""
    print(f"\n  DeepNetz Inference Plan — {model.name}")
    print(f"  {'-' * 50}")
    print(f"  Layers:     {plan.n_gpu_layers} GPU + {plan.n_cpu_layers} CPU")
    print(f"  KV Cache:   K={plan.kv_type_k}, V={plan.kv_type_v} ({plan.kv_cache_mb:.0f} MB)")
    print(f"  Context:    {plan.max_context:,} tokens")
    if plan.use_eviction:
        print(f"  Eviction:   {plan.eviction_strategy}")
    print(f"  Memory:     ~{plan.est_total_memory_mb / 1024:.1f} GB total")
    print(f"  Est. Speed: ~{plan.est_gen_tps:.1f} tok/s generation")
    if plan.warnings:
        print(f"  Warnings:")
        for w in plan.warnings:
            print(f"    - {w}")
    print()
