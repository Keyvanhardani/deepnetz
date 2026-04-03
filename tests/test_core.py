"""
DeepNetz Test Suite — core functionality tests.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestHardware:
    def test_detect(self):
        from deepnetz.engine.hardware import detect_hardware
        hw = detect_hardware()
        assert hw.ram_mb > 0
        assert hw.cpu_cores > 0

    def test_print(self, capsys):
        from deepnetz.engine.hardware import detect_hardware, print_hardware
        hw = detect_hardware()
        print_hardware(hw)
        captured = capsys.readouterr()
        assert "RAM" in captured.out or "CPU" in captured.out


class TestGGUFReader:
    def test_read_nonexistent(self):
        from deepnetz.engine.gguf_reader import read_gguf_metadata
        with pytest.raises(FileNotFoundError):
            read_gguf_metadata("/nonexistent/model.gguf")

    def test_model_spec_local(self):
        """Test with a local model if available."""
        from deepnetz.engine.gguf_reader import gguf_to_model_spec
        # Try common locations
        for path in ["/mnt/d/models/qwen2.5-3b-instruct-q4_k_m.gguf",
                     "D:/models/qwen2.5-3b-instruct-q4_k_m.gguf"]:
            if os.path.exists(path):
                spec = gguf_to_model_spec(path)
                assert spec.n_layers > 0
                assert spec.n_heads > 0
                assert spec.head_dim > 0
                return
        pytest.skip("No local GGUF model available")


class TestPlanner:
    def test_plan_small_model(self):
        from deepnetz.engine.planner import ModelSpec, plan_inference
        from deepnetz.engine.hardware import HardwareProfile

        hw = HardwareProfile(gpus=[], total_vram_mb=0, ram_mb=32768,
                            cpu_cores=8, os="Linux", has_cuda=False)
        spec = ModelSpec(name="test", file_size_mb=2000, n_params_b=3,
                        n_layers=36, n_heads=16, n_kv_heads=2,
                        head_dim=128, context_length=32768)

        plan = plan_inference(spec, hw, target_context=4096)
        assert plan.n_cpu_layers == 36
        assert plan.n_gpu_layers == 0
        assert plan.max_context > 0

    def test_plan_with_gpu(self):
        from deepnetz.engine.planner import ModelSpec, plan_inference
        from deepnetz.engine.hardware import HardwareProfile, GPUInfo

        hw = HardwareProfile(
            gpus=[GPUInfo("RTX 4060", 8192, "8.9", 0)],
            total_vram_mb=8192, ram_mb=32768,
            cpu_cores=8, os="Linux", has_cuda=True
        )
        spec = ModelSpec(name="test", file_size_mb=2000, n_params_b=3,
                        n_layers=36, n_heads=16, n_kv_heads=2,
                        head_dim=128, context_length=32768)

        plan = plan_inference(spec, hw, target_context=4096)
        assert plan.n_gpu_layers > 0


class TestBackendDiscovery:
    def test_discover(self):
        from deepnetz.backends.discovery import discover_backends
        backends = discover_backends()
        assert isinstance(backends, list)
        # At least native should be available
        names = [b.name for b in backends]
        assert "native" in names or "huggingface" in names

    def test_get_backend(self):
        from deepnetz.backends.discovery import get_backend
        b = get_backend("native")
        if b:
            info = b.detect()
            assert info.name == "native"


class TestCache:
    def test_eviction_config(self):
        from deepnetz.cache.eviction import AttentionSinkEvictor, EvictionConfig
        ev = AttentionSinkEvictor(EvictionConfig(max_cache_tokens=1024))
        assert not ev.should_evict(500)
        assert ev.should_evict(1100)

    def test_eviction_range(self):
        from deepnetz.cache.eviction import AttentionSinkEvictor, EvictionConfig
        ev = AttentionSinkEvictor(EvictionConfig(
            sink_tokens=4, window_size=512, max_cache_tokens=1024
        ))
        result = ev.compute_eviction_range(1100)
        assert result is not None
        start, end = result
        assert start == 4  # after sinks
        assert end > start

    def test_turboquant_recommend(self):
        from deepnetz.cache.turboquant import recommend_kv_config
        cfg = recommend_kv_config(3, 8192, 4096)
        assert cfg.k_type in ("f16", "turbo4_0", "turbo3_0", "turbo2_0")


class TestTools:
    def test_registry(self):
        from deepnetz.tools.registry import ToolRegistry
        reg = ToolRegistry()
        tools = reg.list_tools()
        assert len(tools) >= 1
        assert tools[0].name == "web_search"

    def test_openai_schema(self):
        from deepnetz.tools.registry import ToolRegistry
        reg = ToolRegistry()
        schemas = reg.to_openai_tools()
        assert len(schemas) >= 1
        assert schemas[0]["type"] == "function"
        assert "name" in schemas[0]["function"]

    def test_parse_tool_calls(self):
        from deepnetz.tools.registry import ToolRegistry
        reg = ToolRegistry()
        text = '<tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call>'
        calls = reg.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "web_search"


class TestEvaluator:
    def test_good_text(self):
        from deepnetz.engine.evaluator import evaluate_output
        score = evaluate_output("This is a well-formed sentence. It has good structure.")
        assert score.overall > 0.7

    def test_empty_text(self):
        from deepnetz.engine.evaluator import evaluate_output
        score = evaluate_output("")
        assert score.overall == 0

    def test_repetitive_text(self):
        from deepnetz.engine.evaluator import evaluate_output
        text = "word " * 100
        score = evaluate_output(text)
        assert score.repetition_score < 0.5


class TestMonitor:
    def test_get_stats(self):
        from deepnetz.engine.monitor import get_monitor
        m = get_monitor()
        stats = m.get_stats()
        assert stats.ram_total_mb > 0 or stats.cpu_cores > 0

    def test_to_dict(self):
        from deepnetz.engine.monitor import get_monitor
        m = get_monitor()
        d = m.get_stats().to_dict()
        assert "cpu" in d
        assert "ram" in d
        assert "gpu" in d


class TestResolver:
    def test_local_file(self):
        from deepnetz.engine.resolver import resolve_model
        # Should raise for nonexistent
        with pytest.raises(FileNotFoundError):
            resolve_model("nonexistent_model_xyz.gguf")

    def test_protocol_parse(self):
        from deepnetz.engine.resolver import resolve_model
        # ollama:// should fail gracefully (Ollama not installed)
        with pytest.raises((FileNotFoundError, Exception)):
            resolve_model("ollama://nonexistent:latest")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
