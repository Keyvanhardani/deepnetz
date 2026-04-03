"""
DeepNetz Model — the main entry point.

Handles model loading, inference planning, and execution.
"""

from deepnetz.engine.hardware import detect_hardware, print_hardware
from deepnetz.engine.planner import ModelSpec, plan_inference, print_plan


class Model:
    """
    Load and run a model with automatic hardware optimization.

    Usage:
        model = Model("Qwen3.5-122B", gpu_budget="8GB")
        response = model.chat("Hello!")
    """

    def __init__(self, model_path: str,
                 gpu_budget: str = "auto",
                 ram_budget: str = "auto",
                 target_context: int = 4096,
                 verbose: bool = True):
        self.model_path = model_path
        self.verbose = verbose

        # Detect hardware
        self.hw = detect_hardware()
        if verbose:
            print_hardware(self.hw)

        # Parse budgets
        gpu_mb = self._parse_budget(gpu_budget, self.hw.total_vram_mb)
        ram_mb = self._parse_budget(ram_budget, self.hw.ram_mb)

        # TODO: Read model spec from GGUF metadata
        # For now, placeholder
        self.spec = ModelSpec(
            name=model_path,
            file_size_mb=0,
            n_params_b=0,
            n_layers=0,
            n_heads=0,
            n_kv_heads=0,
            head_dim=128,
            context_length=target_context
        )

        # Plan inference
        self.plan = plan_inference(
            self.spec, self.hw,
            target_context=target_context,
            gpu_budget_mb=gpu_mb,
            ram_budget_mb=ram_mb
        )
        if verbose:
            print_plan(self.plan, self.spec)

    def chat(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a response. TODO: implement with llama.cpp backend."""
        raise NotImplementedError(
            "DeepNetz v0.1 — inference engine coming soon. "
            "For now, use the planner to see how your model would run."
        )

    @staticmethod
    def _parse_budget(budget: str, auto_value: int) -> int:
        if budget == "auto":
            return auto_value
        budget = budget.strip().upper()
        if budget.endswith("GB"):
            return int(float(budget[:-2]) * 1024)
        elif budget.endswith("MB"):
            return int(float(budget[:-2]))
        return int(budget)
