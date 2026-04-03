"""
DeepNetz API Server — OpenAI-compatible + monitoring + backends.
"""

import json
import time
from typing import List, Optional


def create_app(model_path: str,
               gpu_budget: str = "auto",
               ram_budget: str = "auto",
               target_context: int = 4096,
               cpu_only: bool = False,
               backend: str = "auto"):
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import StreamingResponse, JSONResponse
        from pydantic import BaseModel as PydanticBaseModel
    except ImportError:
        raise ImportError("pip install fastapi uvicorn")

    from deepnetz.engine.model import Model
    from deepnetz.engine.monitor import get_monitor
    from deepnetz.backends.base import GenerationConfig

    app = FastAPI(title="DeepNetz API", version="0.3.0")

    model = Model(
        model_path, gpu_budget=gpu_budget, ram_budget=ram_budget,
        target_context=target_context, cpu_only=cpu_only, backend=backend,
    )
    model.load()
    monitor = get_monitor()

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
        models = []
        for b in model.backends:
            for m in b.list_models():
                models.append({"id": m.name, "backend": m.backend,
                               "size_mb": m.size_mb})
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        config = GenerationConfig(
            max_tokens=req.max_tokens, temperature=req.temperature
        )

        if req.stream:
            async def generate():
                for token in model.backend.stream(messages, config):
                    chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "model": "deepnetz",
                        "choices": [{"index": 0, "delta": {"content": token},
                                     "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            response = model.backend.chat(messages, config)
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "model": "deepnetz",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": response},
                             "finish_reason": "stop"}],
            }

    @app.get("/v1/stats")
    async def system_stats():
        return monitor.get_stats().to_dict()

    @app.get("/v1/backends")
    async def list_backends():
        return [{"name": b.name, **b.detect().__dict__} for b in model.backends]

    class LoadRequest(PydanticBaseModel):
        model: str
        backend: str = "native"

    class DownloadRequest(PydanticBaseModel):
        model: str

    @app.post("/v1/models/load")
    async def load_model_endpoint(req: LoadRequest):
        try:
            model.backend.unload()
            # Resolve model path
            from deepnetz.engine.resolver import resolve_model
            try:
                path = resolve_model(req.model)
            except FileNotFoundError:
                path = req.model
            model.backend.load(path)
            model.model_ref = req.model
            return {"status": "ok", "model": req.model}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @app.post("/v1/models/download")
    async def download_model_endpoint(req: DownloadRequest):
        try:
            from deepnetz.engine.resolver import resolve_model
            path = resolve_model(req.model, output_dir=".")
            return {"status": "ok", "path": path}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": model.backend.name,
                "model": model.model_ref}

    # Mount Web UI
    from deepnetz.ui.routes import mount_ui
    mount_ui(app)

    @app.websocket("/ws/monitor")
    async def ws_monitor(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                stats = monitor.get_stats()
                await ws.send_json(stats.to_dict())
                import asyncio
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass

    return app
