# DeepNetz

**Run massive models on minimal hardware.**

```bash
pip install deepnetz

deepnetz run model.gguf                         # auto-detect hardware
deepnetz run model.gguf --cpu                    # CPU-only
deepnetz run model.gguf --gpu 8GB                # GPU with budget
deepnetz run ollama://qwen3.5:35b                # load from Ollama
deepnetz run hf://unsloth/Qwen3.5-35B-A3B-GGUF  # load from HuggingFace
deepnetz serve model.gguf --port 8080            # OpenAI-compatible API
```

DeepNetz combines cutting-edge research into one framework that makes large language models run on consumer hardware — no A100 required.

## Quick start

```bash
# Install
pip install deepnetz

# Check your hardware
deepnetz hardware

# Local GGUF file
deepnetz run ./model.gguf

# Load from Ollama (reads from ~/.ollama/models/)
deepnetz run ollama://qwen3.5:35b

# Load from HuggingFace (auto-downloads)
deepnetz run hf://unsloth/Qwen3.5-35B-A3B-GGUF

# Load from LM Studio cache
deepnetz run lmstudio://qwen3.5-35b

# CPU-only / GPU with budget
deepnetz run model.gguf --cpu
deepnetz run model.gguf --gpu 8GB --context 32k

# Interactive chat
deepnetz run model.gguf
#   You: What is quantum computing?
#   AI:  Quantum computing uses quantum mechanics to...

# Single prompt
deepnetz run model.gguf -p "Explain gravity in one sentence"

# OpenAI-compatible API server
deepnetz serve model.gguf --port 8080
# Then: curl http://localhost:8080/v1/chat/completions ...

# Download from HuggingFace
deepnetz download Qwen3.5-35B --quant Q4_K_M
```

## Python API

```python
from deepnetz import Model

# Auto-detect hardware, optimize automatically
model = Model("model.gguf")
response = model.chat("Hello!")

# CPU-only with custom context
model = Model("model.gguf", cpu_only=True, target_context=8192)

# GPU with budget
model = Model("model.gguf", gpu_budget="8GB", ram_budget="32GB")

# Streaming
for token in model.stream("Tell me a story"):
    print(token, end="", flush=True)
```

## What it does

| You have | Without DeepNetz | With DeepNetz |
|----------|-----------------|---------------|
| RTX 4060 8GB + 32GB RAM | 8B model, 4K context | 122B model, 32K context |
| 32GB RAM, no GPU | 7B model, 4K context | 35B model, 8K context |
| RTX 3090 24GB + 64GB RAM | 70B model, 8K context | 122B model, 128K context |

## How it works

DeepNetz auto-detects your hardware, reads model metadata, and computes an optimal inference plan:

```
$ deepnetz info Qwen3.5-122B-A10B-IQ2_XXS.gguf --gpu 8GB

  DeepNetz Hardware Profile
  ────────────────────────────────────────
  OS:       Linux
  CPU:      16 cores
  RAM:      31 GB
  GPU 0:    NVIDIA GeForce RTX 4060 (8188 MB)

  Model: Qwen3.5-122B-A10B
  ────────────────────────────────────────
  Parameters:  ~122B (MoE, 10B active)
  Layers:      96
  Heads:       64 Q / 4 KV
  Head dim:    128
  Context:     262,144
  File size:   34.1 GB

  DeepNetz Inference Plan
  ──────────────────────────────────────────────────
  Layers:     0 GPU + 96 CPU
  KV Cache:   K=turbo4_0, V=turbo4_0 (compressed)
  Context:    4,096 tokens
  Memory:     ~34.2 GB total
  Est. Speed: ~1.3 tok/s generation
```

### The optimization stack

DeepNetz stacks multiple techniques. Each gives 2-4x savings. Combined, they multiply:

```
122B model, 32K context:

KV Cache (naive):        ~16 GB  → doesn't fit
  + TurboQuant (3.6x):     4.4 GB
  + Token Eviction (2x):   2.2 GB
  + KV Merging (1.5x):     1.5 GB  → fits!
```

| Layer | Technique | Based on | Status |
|-------|-----------|----------|--------|
| **Cache Compression** | TurboQuant (WHT + Lloyd-Max) | [Google, ICLR 2026](https://arxiv.org/abs/2504.19874) | Implemented |
| **Smart Offload** | Dynamic GPU/CPU layer split | [Q-Infer](https://dl.acm.org/doi/full/10.1145/3764589) | Implemented |
| **Token Eviction** | Attention-aware pruning | [PagedEviction, EACL 2026](https://aclanthology.org/2026.findings-eacl.168.pdf) | Planned |
| **Attention Sinks** | Keep first + recent tokens | [StreamingLLM](https://arxiv.org/abs/2309.17453) | Planned |
| **KV Merging** | Merge similar tokens | CaM / D2O | Planned |
| **Multi-Tier Cache** | Important tokens = high precision | [KVC-Q](https://www.sciencedirect.com/science/article/abs/pii/S1383762126000172) | Planned |

### What makes it different

**Ollama / LMStudio**: Load model, hope it fits. No KV optimization, no smart offloading.

**vLLM / SGLang**: Server-focused, needs beefy GPUs, not for your laptop.

**DeepNetz**: One command. Detects your hardware, picks the right optimizations, runs the model. CPU and GPU. Consumer-first.

## Benchmarks

Tested on 9 models from 3B to 122B on RTX 4060 (8GB) + 32GB RAM:

| Model | f16 PPL | turbo4_0 PPL | Delta | Generation |
|-------|---------|-------------|-------|------------|
| Llama-3.2-3B Q4_K_M | 9.77 | 9.82 | **+0.4%** | — |
| Qwen3-4B Q4_K_M | 17.78 | 16.61 | **-6.6%** | — |
| Gemma-3-27B Q2_K | 8.53 | 8.70 | +2.0% | 2.3 tok/s |
| Qwen3.5-35B-A3B Q4_K_XL | 5.91 | 6.07 | +2.7% | 7.4 tok/s |
| Llama-3.3-70B IQ2_M | 4.91 | — | — | 0.7 tok/s |
| Qwen3.5-122B-A10B IQ2_XXS | — | — | — | 1.3 tok/s |

[Full benchmark data + TurboQuant standalone library](https://github.com/Keyvanhardani/turboquant-ggml)

## Architecture

```
deepnetz/
├── __init__.py              # from deepnetz import Model
├── cli.py                   # deepnetz run/serve/info/hardware/download
├── server.py                # OpenAI-compatible FastAPI server
└── engine/
    ├── model.py             # Main Model class
    ├── backend.py           # llama-cpp-python wrapper
    ├── hardware.py          # GPU/CPU/RAM auto-detection
    ├── planner.py           # Budget → optimal inference plan
    ├── gguf_reader.py       # Fast GGUF metadata extraction
    └── downloader.py        # HuggingFace model download
```

## Roadmap

- [x] Hardware auto-detection + budget planner
- [x] GGUF metadata reader
- [x] llama-cpp-python inference backend
- [x] CLI tool (`deepnetz run/info/serve/hardware/download`)
- [x] CPU + GPU + hybrid mode
- [x] Interactive chat + single prompt + streaming
- [x] OpenAI-compatible API server
- [x] Model downloader with auto quant selection
- [x] TurboQuant KV cache compression ([turboquant-ggml](https://github.com/Keyvanhardani/turboquant-ggml))
- [ ] Token eviction (attention sinks + scoring)
- [ ] KV merging (CaM/D2O)
- [ ] Multi-tier adaptive cache
- [ ] Web UI

## Author

**Keyvan Hardani** — [keyvan.ai](https://keyvan.ai) | [deepnetz.com](https://deepnetz.com) | [GitHub](https://github.com/Keyvanhardani) | [LinkedIn](https://linkedin.com/in/keyvanhardani)

## License

MIT
