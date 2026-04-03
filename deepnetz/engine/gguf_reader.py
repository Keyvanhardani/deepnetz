"""
GGUF metadata reader — extract model specs without loading the full model.

Reads architecture, layer count, head dimensions, context length etc.
directly from GGUF file header. Fast (< 1 second for any file size).
"""

import struct
import os
from typing import Optional
from deepnetz.engine.planner import ModelSpec


# GGUF value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 12
GGUF_TYPE_FLOAT64 = 13
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3


def _read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _skip_value(f, vtype):
    """Skip a GGUF value without reading it."""
    sizes = {
        GGUF_TYPE_UINT8: 1, GGUF_TYPE_INT8: 1,
        GGUF_TYPE_UINT16: 2, GGUF_TYPE_INT16: 2,
        GGUF_TYPE_UINT32: 4, GGUF_TYPE_INT32: 4,
        GGUF_TYPE_FLOAT32: 4, GGUF_TYPE_BOOL: 1,
        GGUF_TYPE_UINT64: 8, GGUF_TYPE_INT64: 8,
        GGUF_TYPE_FLOAT64: 8,
    }
    if vtype == GGUF_TYPE_STRING:
        length = struct.unpack("<Q", f.read(8))[0]
        f.read(length)
    elif vtype == GGUF_TYPE_ARRAY:
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        if arr_type == GGUF_TYPE_STRING:
            for _ in range(arr_len):
                _read_string(f)
        elif arr_type in sizes:
            f.read(arr_len * sizes[arr_type])
        else:
            raise ValueError(f"Unknown array type {arr_type}")
    elif vtype in sizes:
        f.read(sizes[vtype])
    else:
        raise ValueError(f"Unknown value type {vtype}")


def _read_value(f, vtype):
    """Read a GGUF value."""
    if vtype == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif vtype == GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif vtype == GGUF_TYPE_BOOL:
        return struct.unpack("<?", f.read(1))[0]
    elif vtype == GGUF_TYPE_STRING:
        return _read_string(f)
    elif vtype == GGUF_TYPE_ARRAY:
        _skip_value(f, vtype)
        return None
    else:
        _skip_value(f, vtype)
        return None


def read_gguf_metadata(path: str) -> dict:
    """Read all metadata key-value pairs from a GGUF file."""
    metadata = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")

        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        metadata["_version"] = version
        metadata["_n_tensors"] = n_tensors

        for _ in range(n_kv):
            key = _read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            value = _read_value(f, vtype)
            if value is not None:
                metadata[key] = value

    return metadata


def gguf_to_model_spec(path: str) -> ModelSpec:
    """Extract ModelSpec from GGUF file metadata."""
    meta = read_gguf_metadata(path)
    file_size_mb = os.path.getsize(path) // (1024 * 1024)

    # Architecture prefix (e.g., "llama", "qwen2", "gemma")
    arch = meta.get("general.architecture", "unknown")

    # Extract key parameters
    n_layers = meta.get(f"{arch}.block_count", 0)
    n_embd = meta.get(f"{arch}.embedding_length", 0)
    n_heads = meta.get(f"{arch}.attention.head_count", 0)
    n_kv_heads = meta.get(f"{arch}.attention.head_count_kv", n_heads)
    context_length = meta.get(f"{arch}.context_length", 4096)

    # Head dimension
    head_dim = n_embd // n_heads if n_heads > 0 else 128

    # Detect MoE
    n_experts = meta.get(f"{arch}.expert_count", 0)
    is_moe = n_experts > 0

    # Estimate params (rough, from file size and quant type)
    file_type = meta.get("general.file_type", 0)
    bpw_estimate = {0: 32, 1: 16, 2: 4, 7: 8, 15: 4.5}.get(file_type, 4)
    n_params_b = (file_size_mb * 8) / (bpw_estimate * 1000)

    # Model name
    name = meta.get("general.name", os.path.basename(path))

    return ModelSpec(
        name=name,
        file_size_mb=file_size_mb,
        n_params_b=round(n_params_b, 1),
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        context_length=context_length,
        is_moe=is_moe,
    )


def print_model_info(spec: ModelSpec, meta: Optional[dict] = None):
    """Pretty-print model info."""
    print(f"\n  Model: {spec.name}")
    print(f"  {'-' * 40}")
    print(f"  Parameters:  ~{spec.n_params_b}B" +
          (" (MoE)" if spec.is_moe else ""))
    print(f"  Layers:      {spec.n_layers}")
    print(f"  Heads:       {spec.n_heads} Q / {spec.n_kv_heads} KV")
    print(f"  Head dim:    {spec.head_dim}")
    print(f"  Context:     {spec.context_length:,}")
    print(f"  File size:   {spec.file_size_mb / 1024:.1f} GB")
    if meta:
        arch = meta.get("general.architecture", "?")
        print(f"  Architecture: {arch}")
    print()
