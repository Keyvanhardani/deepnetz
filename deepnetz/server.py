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

    from deepnetz.engine.manager import ModelManager
    from deepnetz.engine.monitor import get_monitor
    from deepnetz.backends.base import GenerationConfig

    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="DeepNetz API", version="0.3.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://deepnetz.com", "https://deepnetz.de",
                        "http://localhost:*", "http://127.0.0.1:*", "*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    manager = ModelManager(
        gpu_budget=gpu_budget,
        ram_budget=ram_budget,
        target_context=target_context,
        cpu_only=cpu_only,
        default_backend=backend,
    )
    app.state.manager = manager

    if model_path:
        manager.load_model(model_path, backend)

    monitor = get_monitor()

    from typing import Any, Union

    class Message(PydanticBaseModel):
        role: str
        content: Any  # str or list (for vision: [{type: "text"}, {type: "image_url"}])

    class ChatRequest(PydanticBaseModel):
        model: str = "deepnetz"
        messages: List[Message]
        max_tokens: int = 512
        temperature: float = 0.7
        stream: bool = False
        session_id: str = ""
        reasoning: bool = False  # Enable reasoning mode

    @app.get("/v1/models")
    async def list_models():
        models = app.state.manager.list_available_models()
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        model = app.state.manager.get_active()
        if model is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "No model loaded. Use /v1/models/load to load a model first.",
                                   "type": "service_unavailable"}},
            )

        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        # Reasoning mode: add reasoning instructions to last user message
        if req.reasoning:
            from deepnetz.engine.features import format_reasoning_prompt
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user" and isinstance(messages[i]["content"], str):
                    messages[i]["content"] = format_reasoning_prompt(messages[i]["content"], True)
                    break

        config = GenerationConfig(
            max_tokens=req.max_tokens, temperature=req.temperature
        )

        # If session_id provided, persist user message and auto-title
        session = None
        if req.session_id:
            session = session_store.get(req.session_id)
            if session:
                # Find the last user message from the request
                user_msg = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else None
                if user_msg:
                    session.messages.append({"role": "user", "content": user_msg})
                    # Auto-generate title from first user message
                    if not session.title and user_msg:
                        session.title = user_msg[:40] + ("..." if len(user_msg) > 40 else "")
                    session_store.save(session)

        if req.stream:
            async def generate():
                full_response = []
                for token in model.backend.stream(messages, config):
                    full_response.append(token)
                    chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "model": "deepnetz",
                        "choices": [{"index": 0, "delta": {"content": token},
                                     "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                # Persist assistant response to session
                if session:
                    assistant_text = "".join(full_response)
                    session_store.add_message(req.session_id, "assistant", assistant_text)
                yield "data: [DONE]\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            response = model.backend.chat(messages, config)
            # Persist assistant response to session
            if session:
                session_store.add_message(req.session_id, "assistant", response)
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "model": "deepnetz",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": response},
                             "finish_reason": "stop"}],
            }

    @app.get("/v1/features")
    async def features():
        """List supported features based on current model."""
        model = app.state.manager.get_active()
        model_name = app.state.manager.model_ref if model else ""
        from deepnetz.engine.features import is_vision_model, is_reasoning_model
        return {
            "vision": is_vision_model(model_name),
            "reasoning": is_reasoning_model(model_name),
            "tool_calling": True,
            "streaming": True,
            "speculative_decoding": False,  # Coming soon
        }

    @app.get("/v1/stats")
    async def system_stats():
        return monitor.get_stats().to_dict()

    @app.get("/v1/backends")
    async def list_backends():
        return [{"name": b.name, **b.detect().__dict__} for b in app.state.manager.backends]

    class LoadRequest(PydanticBaseModel):
        model: str
        backend: str = ""

    class DownloadRequest(PydanticBaseModel):
        model: str

    @app.post("/v1/models/load")
    async def load_model_endpoint(req: LoadRequest):
        import asyncio
        app.state.loading = True
        try:
            # Run in thread so it doesn't block health checks
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, app.state.manager.load_model, req.model, req.backend)
            return {"status": "ok", "model": req.model}
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "error": str(e)},
            )
        finally:
            app.state.loading = False

    @app.post("/v1/models/download")
    async def download_model_endpoint(req: DownloadRequest):
        try:
            from deepnetz.engine.resolver import resolve_model
            path = resolve_model(req.model, output_dir=".")
            return {"status": "ok", "path": path}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    app.state.loading = False

    @app.get("/health")
    async def health():
        mgr = app.state.manager
        model = mgr.get_active()
        backend_name = model.backend.name if model else "none"
        loading = getattr(app.state, 'loading', False)
        return {"status": "loading" if loading else "ok",
                "backend": backend_name,
                "model": mgr.model_ref,
                "loading": loading}

    # ---- Session management ----
    from deepnetz.engine.session import SessionStore
    session_store = SessionStore()

    class SessionCreate(PydanticBaseModel):
        title: str = ""

    class SessionUpdate(PydanticBaseModel):
        title: str = ""

    @app.post("/v1/sessions")
    async def create_session(req: SessionCreate = None):
        title = req.title if req and req.title else ""
        s = session_store.create(title=title)
        return {"id": s.id, "title": s.title, "created_at": s.created_at}

    @app.get("/v1/sessions")
    async def list_sessions():
        sessions = session_store.list_sessions()
        return {"sessions": [{"id": s.id, "title": s.title, "created_at": s.created_at,
                              "updated_at": s.updated_at, "message_count": len(s.messages)}
                             for s in sessions]}

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str):
        s = session_store.get(session_id)
        if not s:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        return {"id": s.id, "title": s.title, "messages": s.messages,
                "config": s.config, "created_at": s.created_at, "updated_at": s.updated_at}

    @app.put("/v1/sessions/{session_id}")
    async def update_session(session_id: str, req: SessionUpdate):
        s = session_store.get(session_id)
        if not s:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        if req.title:
            s.title = req.title
        session_store.save(s)
        return {"id": s.id, "title": s.title}

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str):
        session_store.delete(session_id)
        return {"status": "ok"}

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
