# DeepNetz

**Run massive models on minimal hardware.**

```bash
pip install deepnetz

deepnetz run model.gguf                         # auto-detect hardware
deepnetz run model.gguf --cpu                    # CPU-only
deepnetz run model.gguf --gpu 8GB                # GPU with budget
deepnetz run ollama://qwen3.5:35b                # from Ollama
deepnetz run hf://unsloth/Qwen3.5-35B-A3B-GGUF  # from HuggingFace
deepnetz run lmstudio://qwen3.5-35b             # from LM Studio
deepnetz serve model.gguf --port 8080            # OpenAI-compatible API
```

## What it does

One framework. 6 backends. Any model. Any hardware.

| You have | Typical setup | With DeepNetz optimization |
|----------|--------------|---------------------------|
| RTX 4060 8GB + 32GB RAM | 35B model via Ollama | Same model, 3.6x less KV cache, longer context |
| 32GB RAM, no GPU | 7B model, slow | Auto-optimized CPU inference + KV compression |
| RTX 3090 24GB + 64GB RAM | 70B model | Same model, optimized layer split + cache |

## Quick start

```bash
pip install deepnetz

# Show your hardware + available backends
deepnetz hardware
deepnetz backends

# Run a model (auto-detects everything)
deepnetz run ./model.gguf

# Load from anywhere
deepnetz run ollama://qwen3.5:35b
deepnetz run hf://unsloth/Qwen3.5-35B-A3B-GGUF
deepnetz run lmstudio://qwen3.5-35b

# CPU-only / GPU budget
deepnetz run model.gguf --cpu
deepnetz run model.gguf --gpu 8GB --context 32k

# Single prompt
deepnetz run model.gguf -p "Explain gravity"

# API server with Web UI
deepnetz serve model.gguf --port 8080
# Dashboard: http://localhost:8080/
# Chat:      http://localhost:8080/chat
# Models:    http://localhost:8080/models
# API:       http://localhost:8080/v1/chat/completions

# Download models
deepnetz download Qwen3.5-35B --quant Q4_K_M
```

## Python API

```python
from deepnetz import Model

# Auto everything
model = Model("model.gguf")
response = model.chat("Hello!")

# CPU-only
model = Model("model.gguf", cpu_only=True)

# Specific backend
model = Model("model.gguf", backend="ollama")

# Streaming
for token in model.stream("Tell me a story"):
    print(token, end="", flush=True)
```

## 6 Backends

DeepNetz auto-detects which backends are installed and uses the best one:

| Backend | Source | How it connects |
|---------|--------|----------------|
| **Native** | llama-cpp-python | Direct GGUF inference (fastest) |
| **Ollama** | Ollama REST API | `localhost:11434` |
| **vLLM** | vLLM Python/CLI | `vllm serve` or running instance |
| **LM Studio** | lms CLI / REST | `localhost:1234` |
| **HuggingFace** | transformers | Pipeline (safetensors only) |
| **Remote** | Any OpenAI API | Custom endpoint |

```bash
deepnetz backends   # shows what's available on your system
```

## KV Cache Optimization

DeepNetz stacks compression techniques for up to 10x memory reduction:

```
122B model, 32K context:
  KV Cache (naive):        ~16 GB → doesn't fit
  + TurboQuant (3.6x):       4.4 GB
  + Token Eviction (2x):     2.2 GB
  + KV Merging (1.5x):       1.5 GB → fits!
```

| Technique | Based on | Effect |
|-----------|----------|--------|
| **TurboQuant** | [Google, ICLR 2026](https://arxiv.org/abs/2504.19874) | 3.6x KV compression |
| **Attention Sinks** | [StreamingLLM](https://arxiv.org/abs/2309.17453) | Fixed memory for infinite context |
| **Token Eviction** | [PagedEviction](https://aclanthology.org/2026.findings-eacl.168.pdf) | Remove unimportant tokens |
| **KV Merging** | CaM / D2O | Merge similar tokens |

## Web UI

`deepnetz serve model.gguf` starts a web dashboard at `http://localhost:8080/`:

- **Dashboard** — Live CPU, RAM, GPU, VRAM, temperature monitoring
- **Chat** — Streaming chat interface
- **Models** — Browse and manage models from all backends

## Tool Calling

Built-in internet search, extensible tool framework:

```python
from deepnetz.tools.registry import ToolRegistry

registry = ToolRegistry()  # web_search built-in
result = registry.execute("web_search", {"query": "latest news"})
```

OpenAI-compatible function calling via `/v1/chat/completions`.

## Benchmarks

Tested on 9 models from 3B to 122B on RTX 4060 (8GB) + 32GB RAM:

| Model | PPL Delta | Speed | KV Compression |
|-------|-----------|-------|---------------|
| Llama-3.2-3B | +0.4% | — | 3.6x |
| Gemma-3-27B | +2.0% | 2.3 tok/s | 3.6x |
| Qwen3.5-35B | +2.7% | 7.4 tok/s | 3.6x |
| Llama-3.3-70B | — | 0.7 tok/s | — |
| Qwen3.5-122B | — | 1.3 tok/s | — |

## Architecture

```
deepnetz/
├── __init__.py                  # from deepnetz import Model
├── cli.py                       # CLI (run/serve/info/hardware/backends/download)
├── server.py                    # FastAPI + WebSocket + OpenAI API
├── errors.py                    # Error hierarchy
├── engine/
│   ├── model.py                 # Main orchestrator
│   ├── hardware.py              # GPU/CPU/RAM detection
│   ├── monitor.py               # Real-time system stats
│   ├── planner.py               # Budget → inference plan
│   ├── gguf_reader.py           # GGUF metadata extraction
│   ├── resolver.py              # Universal model resolver (8 sources)
│   ├── downloader.py            # HuggingFace download
│   ├── scanner.py               # Local model discovery
│   ├── session.py               # SQLite conversation persistence
│   └── evaluator.py             # Output quality scoring
├── backends/
│   ├── base.py                  # Adapter interface
│   ├── native.py                # llama-cpp-python
│   ├── ollama.py                # Ollama REST API
│   ├── vllm.py                  # vLLM
│   ├── lmstudio.py              # LM Studio
│   ├── huggingface.py           # transformers
│   ├── remote.py                # Any OpenAI API
│   └── discovery.py             # Auto-detect backends
├── cache/
│   ├── turboquant.py            # TurboQuant KV compression
│   ├── eviction.py              # Attention sink eviction
│   └── merging.py               # KV entry merging
├── tools/
│   ├── base.py                  # Tool protocol
│   ├── search.py                # Web search (DuckDuckGo)
│   └── registry.py              # Tool management + parser
└── ui/
    ├── routes.py                # Web UI routes
    ├── static/                  # JS, CSS
    └── templates/               # Dashboard, Chat, Models HTML
```

## What makes it different

| Feature | Ollama | LM Studio | vLLM | **DeepNetz** |
|---------|--------|-----------|------|-------------|
| Load from anywhere | Own registry | Own catalog | HuggingFace | **All of them** |
| KV Cache Compression | No | No | No | **TurboQuant 3.6x** |
| Multi-Backend | No | No | No | **6 backends** |
| Hardware Auto-Tuning | Basic | Basic | No | **Budget planner** |
| Web UI + Monitoring | No | Yes (closed) | No | **Yes** |
| Tool Calling | No | No | Yes | **Yes + Search** |
| CPU Optimized | Yes | Yes | No | **Yes + KV compression** |
| Quality Scoring | No | No | No | **Yes** |

## Author

**Keyvan Hardani** — [keyvan.ai](https://keyvan.ai) | [deepnetz.com](https://deepnetz.com) | [GitHub](https://github.com/Keyvanhardani) | [LinkedIn](https://linkedin.com/in/keyvanhardani)

## Contributing

PRs welcome! See [open issues](https://github.com/Keyvanhardani/deepnetz/issues).

```bash
git clone https://github.com/Keyvanhardani/deepnetz.git
cd deepnetz
pip install -e ".[server]"
pytest tests/
```

## License

MIT — use it, fork it, build on it.
