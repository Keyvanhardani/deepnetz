"""
DeepNetz Registry Server — zentraler Punkt für alles.

Deploy auf registry.deepnetz.com.
Clients fragen hier nach: Suche, Modell-Configs, Auth.
Blobs werden direkt von HuggingFace heruntergeladen.

Endpoints:
    POST /v1/auth/register           Registrierung
    POST /v1/auth/login              Login → Token
    GET  /v1/auth/me                 User-Info

    GET  /v1/search?q=Qwen           Modell-Suche (HF-Proxy)
    GET  /v1/models/{name}           Modell-Info (HF-Lookup)
    GET  /v1/models/{name}/files     GGUF-Dateien im Repo

Usage:
    deepnetz registry --port 8090
"""

import json
import os
import time
import hashlib
import secrets
from typing import Optional


def create_registry_app(db_path: str = ""):
    """Create the registry FastAPI app."""
    try:
        from fastapi import FastAPI, HTTPException, Depends
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
        from pydantic import BaseModel as PydanticBaseModel
    except ImportError:
        raise ImportError("pip install fastapi uvicorn")

    import sqlite3

    app = FastAPI(
        title="DeepNetz Registry",
        version="1.0.0",
        description="Model registry for DeepNetz — deepnetz.com",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Database ───────────────────────────────────────────────────

    if not db_path:
        db_path = os.path.join(os.path.expanduser("~"), ".cache",
                               "deepnetz", "registry.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            api_key TEXT UNIQUE,
            created_at REAL,
            last_seen REAL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS pull_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            model TEXT,
            quant TEXT,
            repo TEXT,
            ts REAL
        )
    """)
    db.commit()

    # ── Auth Helpers ───────────────────────────────────────────────

    security = HTTPBearer(auto_error=False)

    def _hash_pw(password: str) -> str:
        salt = secrets.token_hex(16)
        h = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
        return f"{salt}:{h}"

    def _check_pw(password: str, stored: str) -> bool:
        salt, h = stored.split(":", 1)
        return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest() == h

    def _generate_api_key() -> str:
        return f"dn-{secrets.token_hex(24)}"

    def _get_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
        """Optional auth — returns user dict or None."""
        if not credentials:
            return None
        token = credentials.credentials
        row = db.execute(
            "SELECT id, username, api_key FROM users WHERE api_key=?",
            (token,)
        ).fetchone()
        if row:
            db.execute("UPDATE users SET last_seen=? WHERE id=?", (time.time(), row[0]))
            db.commit()
            return {"id": row[0], "username": row[1]}
        return None

    # ── Auth Endpoints ─────────────────────────────────────────────

    class RegisterReq(PydanticBaseModel):
        username: str
        password: str

    class LoginReq(PydanticBaseModel):
        username: str
        password: str

    @app.post("/v1/auth/register")
    async def register(req: RegisterReq):
        if len(req.username) < 3:
            raise HTTPException(400, "Username muss mindestens 3 Zeichen haben")
        if len(req.password) < 6:
            raise HTTPException(400, "Passwort muss mindestens 6 Zeichen haben")

        existing = db.execute(
            "SELECT id FROM users WHERE username=?", (req.username,)
        ).fetchone()
        if existing:
            raise HTTPException(409, "Username bereits vergeben")

        user_id = f"u_{int(time.time() * 1000)}"
        api_key = _generate_api_key()
        db.execute(
            "INSERT INTO users (id, username, password_hash, api_key, created_at, last_seen) VALUES (?,?,?,?,?,?)",
            (user_id, req.username, _hash_pw(req.password), api_key, time.time(), time.time())
        )
        db.commit()
        return {"user_id": user_id, "username": req.username, "api_key": api_key}

    @app.post("/v1/auth/login")
    async def login(req: LoginReq):
        row = db.execute(
            "SELECT id, username, password_hash, api_key FROM users WHERE username=?",
            (req.username,)
        ).fetchone()
        if not row or not _check_pw(req.password, row[2]):
            raise HTTPException(401, "Falscher Username oder Passwort")

        db.execute("UPDATE users SET last_seen=? WHERE id=?", (time.time(), row[0]))
        db.commit()
        return {"user_id": row[0], "username": row[1], "api_key": row[3]}

    @app.get("/v1/auth/me")
    async def me(user=Depends(_get_user)):
        if not user:
            raise HTTPException(401, "Nicht authentifiziert")
        return user

    # ── Search (HF-Proxy über unseren Server) ──────────────────────

    @app.get("/v1/search")
    async def search_models(q: str, limit: int = 15, user=Depends(_get_user)):
        """Modell-Suche. Proxy zu HuggingFace, aber über unseren Server."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            results = list(api.list_models(
                search=f"{q} GGUF",
                sort="downloads",
                direction=-1,
                limit=limit,
            ))
        except Exception as e:
            raise HTTPException(502, f"HuggingFace nicht erreichbar: {e}")

        models = []
        for r in results:
            models.append({
                "repo": r.modelId,
                "downloads": getattr(r, "downloads", 0),
                "likes": getattr(r, "likes", 0),
            })

        return {"query": q, "results": models, "total": len(models)}

    # ── Model Info ─────────────────────────────────────────────────

    @app.get("/v1/models/{repo:path}/files")
    async def model_files(repo: str, user=Depends(_get_user)):
        """Liste GGUF-Dateien in einem HF-Repo."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            all_files = list(api.list_repo_tree(repo, recursive=True))
            gguf_files = []
            for f in all_files:
                if f.rfilename.endswith(".gguf"):
                    gguf_files.append({
                        "name": f.rfilename,
                        "size": getattr(f, "size", 0),
                    })
        except Exception as e:
            raise HTTPException(404, f"Repo nicht gefunden: {repo}")

        return {"repo": repo, "files": gguf_files}

    @app.get("/v1/models/{repo:path}")
    async def model_info(repo: str, user=Depends(_get_user)):
        """Modell-Info von HuggingFace."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.model_info(repo)
            return {
                "repo": repo,
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
                "tags": getattr(info, "tags", []),
                "updated": str(getattr(info, "lastModified", "")),
            }
        except Exception:
            raise HTTPException(404, f"Modell nicht gefunden: {repo}")

    # ── Pull Logging ───────────────────────────────────────────────

    class PullLog(PydanticBaseModel):
        model: str
        quant: str = ""
        repo: str = ""

    @app.post("/v1/pulls")
    async def log_pull(req: PullLog, user=Depends(_get_user)):
        """Client meldet einen Pull — für Analytics."""
        user_id = user["id"] if user else "anonymous"
        db.execute(
            "INSERT INTO pull_log (user_id, model, quant, repo, ts) VALUES (?,?,?,?,?)",
            (user_id, req.model, req.quant, req.repo, time.time())
        )
        db.commit()
        return {"status": "ok"}

    # ── Health ─────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        user_count = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        pull_count = db.execute("SELECT COUNT(*) FROM pull_log").fetchone()[0]
        return {
            "status": "ok",
            "version": "1.0.0",
            "users": user_count,
            "pulls": pull_count,
        }

    return app


# Standalone entry point for uvicorn
# Avoids importing the full deepnetz package
def _create_standalone_app():
    import sys
    # Make sure we don't trigger deepnetz.__init__ imports
    db = os.environ.get("DEEPNETZ_DB_PATH", "")
    return create_registry_app(db_path=db)


app = _create_standalone_app()
