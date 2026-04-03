"""
DeepNetz Cache — KV cache compression stack.

Implements:
- TurboQuant (WHT + Lloyd-Max quantization)
- Token eviction (attention sinks)
- KV merging (CaM/D2O style)
- Multi-tier adaptive cache
"""
