"""
Web UI routes — serves the DeepNetz SPA and static assets.
"""

import os

UI_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(UI_DIR, "static")
TEMPLATES_DIR = os.path.join(UI_DIR, "templates")


def mount_ui(app):
    """Mount Web UI routes onto a FastAPI app."""
    try:
        from fastapi import Request
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError:
        return  # FastAPI not available

    # Serve static files
    if os.path.exists(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def _render(template: str) -> str:
        path = os.path.join(TEMPLATES_DIR, template)
        if not os.path.exists(path):
            return "<h1>Template not found</h1>"
        with open(path) as f:
            return f.read()

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return _render("index.html")

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page():
        return _render("index.html")

    @app.get("/models", response_class=HTMLResponse)
    async def models_page():
        return _render("index.html")

    @app.get("/monitor", response_class=HTMLResponse)
    async def monitor_page():
        return _render("index.html")

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page():
        return _render("index.html")
