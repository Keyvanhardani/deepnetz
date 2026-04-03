"""
Web UI routes — mounts dashboard, chat, and model hub pages.
"""

import os
import json

UI_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(UI_DIR, "static")
TEMPLATES_DIR = os.path.join(UI_DIR, "templates")


def mount_ui(app):
    """Mount Web UI routes onto a FastAPI app."""
    try:
        from fastapi import Request, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError:
        return  # FastAPI not available

    # Serve static files
    if os.path.exists(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def _render(template: str, **kwargs) -> str:
        """Simple template rendering (no Jinja2 dependency needed)."""
        path = os.path.join(TEMPLATES_DIR, template)
        if not os.path.exists(path):
            return f"<h1>Template not found: {template}</h1>"
        with open(path) as f:
            html = f.read()
        for key, value in kwargs.items():
            html = html.replace(f"{{{{{key}}}}}", str(value))
        return html

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return _render("dashboard.html")

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page():
        return _render("chat.html")

    @app.get("/models", response_class=HTMLResponse)
    async def models_page():
        return _render("models.html")

    @app.get("/landing", response_class=HTMLResponse)
    async def landing_page():
        return _render("landing.html")
