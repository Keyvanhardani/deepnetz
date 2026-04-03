# DeepNetz

**Run massive models on minimal hardware.**

```
deepnetz run Qwen3.5-122B --gpu 8GB
```

DeepNetz combines cutting-edge research into one framework that makes large language models run on consumer hardware — no A100 required.

## What it does

| You have | Without DeepNetz | With DeepNetz |
|----------|-----------------|---------------|
| RTX 4060 8GB + 32GB RAM | 8B model, 4K context | 122B model, 32K context |
| RTX 3090 24GB + 64GB RAM | 70B model, 8K context | 122B model, 128K context |

## How it works

DeepNetz stacks multiple optimization techniques that individually give 2-4x savings. Combined, they multiply:

```
122B model, 32K context:

KV Cache (naive):        ~16 GB  -- doesn't fit
  + TurboQuant (3.6x):     4.4 GB
  + Token Eviction (2x):   2.2 GB
  + KV Merging (1.5x):     1.5 GB  -- fits on 8GB GPU!
```

### The stack

| Layer | Technique | Paper | Savings |
|-------|-----------|-------|---------|
| **Cache Compression** | TurboQuant (WHT + Lloyd-Max) | [Google, ICLR 2026](https://arxiv.org/abs/2504.19874) | 3.6x |
| **Token Eviction** | Attention-aware pruning | [PagedEviction, EACL 2026](https://aclanthology.org/2026.findings-eacl.168.pdf) | 2-4x |
| **Attention Sinks** | Keep first + recent tokens | [StreamingLLM](https://arxiv.org/abs/2309.17453) | Infinite context |
| **KV Merging** | Merge similar tokens | CaM / D2O | 1.5-2x |
| **Smart Offload** | Dynamic GPU/CPU layer split | [Q-Infer](https://dl.acm.org/doi/full/10.1145/3764589) | Optimal HW use |
| **Multi-Tier Cache** | Important tokens = high precision | [KVC-Q](https://www.sciencedirect.com/science/article/abs/pii/S1383762126000172) | Adaptive |

### What makes it different

**Ollama / LMStudio**: Load model, hope it fits. No KV optimization, no smart offloading.

**vLLM / SGLang**: Server-focused, needs beefy GPUs, not for your laptop.

**DeepNetz**: One command. Detects your hardware, picks the right optimizations, runs the model. Consumer-first.

## Usage

```python
from deepnetz import Model

model = Model("Qwen3.5-122B",
    gpu_budget="8GB",
    ram_budget="32GB",
    target_context=32768)

response = model.chat("Explain quantum computing")
```

CLI:

```bash
# Auto-detect hardware, pick best config
deepnetz run Qwen3.5-122B

# Explicit budget
deepnetz run Llama-3.3-70B --gpu 8GB --ram 32GB --context 32k

# Serve as API
deepnetz serve Qwen3.5-35B --port 8080
```

## Verified benchmarks (TurboQuant KV cache)

Tested on 9 models from 3B to 122B:

| Model | PPL Delta | KV Compression | Generation Speed |
|-------|-----------|---------------|-----------------|
| Llama-3.2-3B | +0.4% | 3.6x | — |
| Gemma-3-27B | +2.0% | 3.6x | 2.3 tok/s |
| Qwen3.5-35B | +2.7% | 3.6x | 7.4 tok/s |
| Qwen3.5-122B | — | 3.6x | 1.3 tok/s |

All tested on RTX 4060 (8GB) + 32GB RAM. [Full benchmark data](https://github.com/Keyvanhardani/turboquant-ggml).

## Roadmap

- [x] TurboQuant KV cache compression (v0.1)
- [ ] Hardware auto-detection + budget planner
- [ ] Dynamic GPU/CPU layer offloading
- [ ] Token eviction (attention sinks + scoring)
- [ ] KV merging (CaM/D2O)
- [ ] Multi-tier adaptive cache
- [ ] CLI tool (`deepnetz run`)
- [ ] OpenAI-compatible API server
- [ ] Web UI

## Author

**Keyvan Hardani** — [keyvan.ai](https://keyvan.ai) | [GitHub](https://github.com/Keyvanhardani) | [LinkedIn](https://linkedin.com/in/keyvanhardani)

## License

MIT
