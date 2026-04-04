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
        reasoning: bool = False
        think_mode: bool = False
        tool_call: bool = False
        web_search: bool = False
        images: List[str] = []  # base64 encoded images

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

        # Vision: attach images to last user message
        if req.images:
            from deepnetz.engine.features import prepare_vision_message
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    text = messages[i]["content"] if isinstance(messages[i]["content"], str) else str(messages[i]["content"])
                    messages[i] = prepare_vision_message(text, image_base64=req.images)
                    break

        # Think mode: wrap prompt with <think> instruction
        if req.think_mode:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user" and isinstance(messages[i]["content"], str):
                    messages[i]["content"] = (
                        messages[i]["content"] + "\n\n"
                        "Think carefully before answering. Use <think>...</think> tags "
                        "for your internal reasoning, then give your answer."
                    )
                    break

        # Reasoning mode: add reasoning instructions to last user message
        elif req.reasoning:
            from deepnetz.engine.features import format_reasoning_prompt
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user" and isinstance(messages[i]["content"], str):
                    messages[i]["content"] = format_reasoning_prompt(messages[i]["content"], True)
                    break

        # Tool calling: add tool instruction
        if req.tool_call:
            tool_instruction = {
                "role": "system",
                "content": (
                    "You have access to tools. When you need to use a tool, "
                    "output a JSON block with {\"name\": \"tool_name\", \"arguments\": {...}}. "
                    "Available tools: calculator, web_search, code_interpreter."
                ),
            }
            messages.insert(0, tool_instruction)

        # Web search: add search context instruction
        if req.web_search:
            search_instruction = {
                "role": "system",
                "content": (
                    "The user wants you to search the web for current information. "
                    "Indicate when you would search by writing [SEARCH: query]. "
                    "Then provide the best answer based on your knowledge, noting "
                    "that real-time search is not yet connected."
                ),
            }
            messages.insert(0, search_instruction)

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
            import asyncio, queue, threading

            async def generate():
                q = queue.Queue()
                full_response = []

                def _stream_worker():
                    try:
                        for token in model.backend.stream(messages, config):
                            q.put(token)
                        q.put(None)  # sentinel
                    except Exception as e:
                        q.put(None)

                thread = threading.Thread(target=_stream_worker, daemon=True)
                thread.start()

                while True:
                    try:
                        token = q.get(timeout=0.05)
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                        continue
                    if token is None:
                        break
                    full_response.append(token)
                    chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "model": "deepnetz",
                        "choices": [{"index": 0, "delta": {"content": token},
                                     "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                if session:
                    assistant_text = "".join(full_response)
                    session_store.add_message(req.session_id, "assistant", assistant_text)
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, model.backend.chat, messages, config)
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
        gpu_layers: int = -1
        context_length: int = 0

    class DownloadRequest(PydanticBaseModel):
        model: str
        filename: str = ""  # specific GGUF file in repo

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
            # If filename given, use hf://repo/filename format
            ref = req.model
            if req.filename:
                ref = f"hf://{req.model}/{req.filename}"
            elif "/" in ref and not ref.startswith(("hf://", "http")):
                ref = f"hf://{ref}"
            path = resolve_model(ref, output_dir=".")
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

    # ---- Config / Hardware / Cards / Auth endpoints ----

    @app.get("/v1/config")
    async def get_config():
        mgr = app.state.manager
        model = mgr.get_active()
        return {
            "gpu_budget": mgr.gpu_budget,
            "ram_budget": mgr.ram_budget,
            "target_context": mgr.target_context,
            "cpu_only": mgr.cpu_only,
            "default_backend": mgr.default_backend,
            "model": mgr.model_ref,
            "loaded": mgr.is_loaded,
        }

    @app.post("/v1/models/unload")
    async def unload_model():
        app.state.manager.unload_model()
        return {"status": "ok"}

    @app.get("/v1/cards/search")
    async def search_model_cards(q: str = "", limit: int = 20):
        from deepnetz.engine.cards import load_cards, search_cards
        cards = load_cards()
        if not q:
            # Return all cards sorted by downloads
            all_cards = sorted(cards.values(), key=lambda c: c.downloads, reverse=True)
            return {"cards": [c.to_dict() for c in all_cards[:limit]]}
        results = search_cards(q, cards)
        return {"cards": [c.to_dict() for c in results[:limit]]}

    @app.get("/v1/auth/status")
    async def auth_status():
        import os as _os
        cred_path = _os.path.join(_os.path.expanduser("~"), ".config", "deepnetz", "credentials.json")
        if _os.path.exists(cred_path):
            try:
                with open(cred_path) as f:
                    data = json.load(f)
                if data.get("api_key"):
                    return {
                        "logged_in": True,
                        "username": data.get("username", ""),
                        "api_key_prefix": data["api_key"][:10] + "...",
                    }
            except Exception:
                pass
        return {"logged_in": False}

    @app.get("/v1/hardware")
    async def hardware_info():
        from deepnetz.engine.hardware import detect_hardware
        hw = detect_hardware()
        return {
            "cpu_cores": hw.cpu_cores,
            "ram_mb": hw.ram_mb,
            "gpus": [{"name": g.name, "vram_mb": g.vram_mb, "compute": g.compute_capability} for g in hw.gpus],
            "total_vram_mb": hw.total_vram_mb,
            "has_cuda": hw.has_cuda,
            "os": hw.os,
        }

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
