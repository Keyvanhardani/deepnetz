"""
DeepNetz API Server — OpenAI-compatible /v1/chat/completions endpoint.

Usage:
    deepnetz serve model.gguf --port 8080

Then use with any OpenAI-compatible client:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
    response = client.chat.completions.create(
        model="deepnetz",
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""

import json
import time
from typing import Optional

from deepnetz.engine.model import Model


def create_app(model_path: str,
               gpu_budget: str = "auto",
               ram_budget: str = "auto",
               target_context: int = 4096,
               cpu_only: bool = False):
    """Create FastAPI app with loaded model."""

    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import StreamingResponse, JSONResponse
        from pydantic import BaseModel as PydanticBaseModel
        from typing import List, Optional as Opt
    except ImportError:
        raise ImportError("pip install fastapi uvicorn")

    app = FastAPI(
        title="DeepNetz API",
        description="OpenAI-compatible API powered by DeepNetz",
        version="0.1.0",
    )

    # Load model at startup
    model = Model(
        model_path,
        gpu_budget=gpu_budget,
        ram_budget=ram_budget,
        target_context=target_context,
        cpu_only=cpu_only,
    )
    model.load()

    class Message(PydanticBaseModel):
        role: str
        content: str

    class ChatRequest(PydanticBaseModel):
        model: str = "deepnetz"
        messages: List[Message]
        max_tokens: int = 512
        temperature: float = 0.7
        stream: bool = False

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": "deepnetz",
                "object": "model",
                "owned_by": "deepnetz",
                "meta": {
                    "name": model.spec.name,
                    "parameters": f"{model.spec.n_params_b}B",
                    "layers": model.spec.n_layers,
                    "context": model.plan.max_context,
                    "kv_cache": model.plan.kv_type_k,
                }
            }]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt = req.messages[-1].content

        if req.stream:
            async def generate():
                for token in model.stream(prompt, max_tokens=req.max_tokens,
                                          temperature=req.temperature):
                    chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "model": "deepnetz",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            response = model.chat(prompt, max_tokens=req.max_tokens,
                                  temperature=req.temperature)
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "model": "deepnetz",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response.split()),
                    "total_tokens": len(prompt.split()) + len(response.split())
                }
            }

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model.spec.name}

    return app
