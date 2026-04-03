# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it via [GitHub Issues](https://github.com/Keyvanhardani/deepnetz/issues) with the label `security`.

Do NOT include exploit details in public issues. Email hardani@hotmail.de for sensitive reports.

## Security Design

- **GGUF models**: Binary format, no code execution risk
- **HuggingFace models**: Only safetensors format accepted, no pickle files
- **Remote endpoints**: TLS validated, no auto-trust
- **Tool execution**: Sandboxed, no shell access
- **API server**: Localhost-only by default (`--host 0.0.0.0` must be explicit)
- **No telemetry**: DeepNetz does not collect or send any data
