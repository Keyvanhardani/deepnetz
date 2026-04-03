"""
DeepNetz Registry Server — Auth, Search, Analytics.

Endpoints:
    POST /v1/auth/register              Email+Password Registrierung
    POST /v1/auth/login                 Email+Password Login
    GET  /v1/auth/me                    User-Info
    POST /v1/auth/device                Device Flow starten (CLI)
    GET  /v1/auth/device/{code}/poll    CLI pollt bis Login fertig
    GET  /v1/auth/device/{code}/complete  Browser bestätigt Login
    GET  /v1/auth/github                GitHub OAuth Start
    GET  /v1/auth/github/callback       GitHub OAuth Callback
    GET  /v1/auth/google                Google OAuth Start
    GET  /v1/auth/google/callback       Google OAuth Callback

    GET  /v1/search?q=Qwen             Modell-Suche
    GET  /v1/models/{name}             Modell-Info
    POST /v1/pulls                     Pull-Logging

Usage:
    deepnetz registry --port 8090

Env vars:
    DEEPNETZ_DB_PATH          SQLite DB path
    GITHUB_CLIENT_ID           GitHub OAuth App Client ID
    GITHUB_CLIENT_SECRET       GitHub OAuth App Secret
    GOOGLE_CLIENT_ID           Google OAuth Client ID
    GOOGLE_CLIENT_SECRET       Google OAuth Secret
    DEEPNETZ_BASE_URL          Base URL (default: https://registry.deepnetz.com)
"""

import json
import os
import time
import hashlib
import secrets
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional


def create_registry_app(db_path: str = ""):
    try:
        from fastapi import FastAPI, HTTPException, Depends, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
        from fastapi.responses import RedirectResponse, HTMLResponse
        from pydantic import BaseModel as PydanticBaseModel
    except ImportError:
        raise ImportError("pip install fastapi uvicorn")

    import sqlite3

    app = FastAPI(title="DeepNetz Registry", version="1.1.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    BASE_URL = os.environ.get("DEEPNETZ_BASE_URL", "https://registry.deepnetz.com")
    GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
    GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
    GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")

    # ── Database ───────────────────────────────────────────────────

    if not db_path:
        db_path = os.path.join(os.path.expanduser("~"), ".cache", "deepnetz", "registry.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = sqlite3.connect(db_path, check_same_thread=False)

    db.execute("""CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT DEFAULT '',
        password_hash TEXT DEFAULT '',
        oauth_provider TEXT DEFAULT '',
        oauth_id TEXT DEFAULT '',
        api_key TEXT UNIQUE,
        created_at REAL,
        last_seen REAL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS device_codes (
        code TEXT PRIMARY KEY,
        user_code TEXT UNIQUE,
        user_id TEXT DEFAULT '',
        api_key TEXT DEFAULT '',
        status TEXT DEFAULT 'pending',
        created_at REAL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS pull_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT, model TEXT, quant TEXT, repo TEXT, ts REAL
    )""")
    db.commit()

    # Migrate: add email, oauth columns if missing
    for col, default in [("email", "''"), ("oauth_provider", "''"), ("oauth_id", "''")]:
        try:
            db.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT DEFAULT {default}")
            db.commit()
        except sqlite3.OperationalError:
            pass

    # ── Helpers ────────────────────────────────────────────────────

    security = HTTPBearer(auto_error=False)

    def _hash_pw(password: str) -> str:
        salt = secrets.token_hex(16)
        h = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
        return f"{salt}:{h}"

    def _check_pw(password: str, stored: str) -> bool:
        if not stored:
            return False
        salt, h = stored.split(":", 1)
        return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest() == h

    def _generate_api_key() -> str:
        return f"dn-{secrets.token_hex(24)}"

    def _generate_user_code() -> str:
        return secrets.token_hex(4).upper()  # 8 char code like "A1B2C3D4"

    def _find_or_create_oauth_user(provider: str, oauth_id: str, username: str, email: str) -> dict:
        """Find existing OAuth user or create new one."""
        row = db.execute(
            "SELECT id, username, api_key FROM users WHERE oauth_provider=? AND oauth_id=?",
            (provider, oauth_id)
        ).fetchone()
        if row:
            db.execute("UPDATE users SET last_seen=? WHERE id=?", (time.time(), row[0]))
            db.commit()
            return {"id": row[0], "username": row[1], "api_key": row[2]}

        # Check if username taken, append suffix if needed
        base_username = username or f"{provider}_{oauth_id[:8]}"
        final_username = base_username
        suffix = 1
        while db.execute("SELECT id FROM users WHERE username=?", (final_username,)).fetchone():
            final_username = f"{base_username}_{suffix}"
            suffix += 1

        user_id = f"u_{int(time.time() * 1000)}"
        api_key = _generate_api_key()
        db.execute(
            "INSERT INTO users (id, username, email, password_hash, oauth_provider, oauth_id, api_key, created_at, last_seen) VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, final_username, email, "", provider, oauth_id, api_key, time.time(), time.time())
        )
        db.commit()
        return {"id": user_id, "username": final_username, "api_key": api_key}

    def _get_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
        if not credentials:
            return None
        token = credentials.credentials
        row = db.execute("SELECT id, username, api_key FROM users WHERE api_key=?", (token,)).fetchone()
        if row:
            db.execute("UPDATE users SET last_seen=? WHERE id=?", (time.time(), row[0]))
            db.commit()
            return {"id": row[0], "username": row[1]}
        return None

    def _oauth_request(url: str, data: dict = None, headers: dict = None) -> dict:
        """Helper for OAuth HTTP requests."""
        hdrs = {"Accept": "application/json", "User-Agent": "deepnetz/1.0"}
        if headers:
            hdrs.update(headers)
        if data:
            body = urllib.parse.urlencode(data).encode()
            req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
        else:
            req = urllib.request.Request(url, headers=hdrs)
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read().decode())

    # ── Auth: Email + Password ─────────────────────────────────────

    class RegisterReq(PydanticBaseModel):
        username: str
        password: str
        email: str = ""

    class LoginReq(PydanticBaseModel):
        username: str
        password: str

    @app.post("/v1/auth/register")
    async def register(req: RegisterReq):
        if len(req.username) < 3:
            raise HTTPException(400, "Username muss mindestens 3 Zeichen haben")
        if len(req.password) < 6:
            raise HTTPException(400, "Passwort muss mindestens 6 Zeichen haben")
        if db.execute("SELECT id FROM users WHERE username=?", (req.username,)).fetchone():
            raise HTTPException(409, "Username bereits vergeben")

        user_id = f"u_{int(time.time() * 1000)}"
        api_key = _generate_api_key()
        db.execute(
            "INSERT INTO users (id, username, email, password_hash, api_key, created_at, last_seen) VALUES (?,?,?,?,?,?,?)",
            (user_id, req.username, req.email, _hash_pw(req.password), api_key, time.time(), time.time())
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

    # ── Auth: Device Flow (für CLI) ────────────────────────────────
    # 1. CLI ruft POST /v1/auth/device → bekommt device_code + user_code
    # 2. CLI öffnet Browser: {BASE_URL}/auth?code={user_code}
    # 3. User loggt sich ein (Email/GitHub/Google)
    # 4. Browser ruft GET /v1/auth/device/{code}/complete
    # 5. CLI pollt GET /v1/auth/device/{code}/poll → bekommt api_key

    @app.post("/v1/auth/device")
    async def device_start():
        """Start device auth flow — CLI calls this."""
        code = secrets.token_hex(16)
        user_code = _generate_user_code()
        db.execute(
            "INSERT INTO device_codes (code, user_code, status, created_at) VALUES (?,?,?,?)",
            (code, user_code, "pending", time.time())
        )
        db.commit()
        return {
            "device_code": code,
            "user_code": user_code,
            "verification_url": f"{BASE_URL}/auth?code={user_code}",
            "expires_in": 600,
        }

    @app.get("/v1/auth/device/{code}/poll")
    async def device_poll(code: str):
        """CLI polls this until login is complete."""
        row = db.execute(
            "SELECT status, api_key, created_at FROM device_codes WHERE code=?", (code,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Device code not found")
        if time.time() - row[2] > 600:
            raise HTTPException(410, "Device code expired")
        if row[0] == "complete":
            return {"status": "complete", "api_key": row[1]}
        return {"status": "pending"}

    def _complete_device_flow(user_code: str, user: dict):
        """Complete a device flow — link user to device code."""
        row = db.execute(
            "SELECT code, created_at FROM device_codes WHERE user_code=? AND status='pending'",
            (user_code,)
        ).fetchone()
        if row and time.time() - row[1] < 600:
            db.execute(
                "UPDATE device_codes SET status='complete', user_id=?, api_key=? WHERE code=?",
                (user["id"], user["api_key"], row[0])
            )
            db.commit()

    # ── Auth: GitHub OAuth ─────────────────────────────────────────

    @app.get("/v1/auth/github")
    async def github_start(device_code: str = ""):
        if not GITHUB_CLIENT_ID:
            raise HTTPException(501, "GitHub OAuth nicht konfiguriert")
        state = device_code or secrets.token_hex(16)
        params = urllib.parse.urlencode({
            "client_id": GITHUB_CLIENT_ID,
            "redirect_uri": f"{BASE_URL}/v1/auth/github/callback",
            "scope": "read:user user:email",
            "state": state,
        })
        return RedirectResponse(f"https://github.com/login/oauth/authorize?{params}")

    @app.get("/v1/auth/github/callback")
    async def github_callback(code: str, state: str = ""):
        if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
            raise HTTPException(501, "GitHub OAuth nicht konfiguriert")
        # Exchange code for token
        token_data = _oauth_request("https://github.com/login/oauth/access_token", {
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code": code,
        })
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(400, "GitHub auth failed")

        # Get user info
        user_info = _oauth_request("https://api.github.com/user",
                                   headers={"Authorization": f"Bearer {access_token}"})
        gh_id = str(user_info.get("id", ""))
        gh_username = user_info.get("login", "")
        gh_email = user_info.get("email", "")

        user = _find_or_create_oauth_user("github", gh_id, gh_username, gh_email)

        # If this was a device flow, complete it
        if state and len(state) == 32:
            # state might be a device_code
            _complete_device_flow_by_device_code(state, user)

        return _auth_success_page(user)

    # ── Auth: Google OAuth ─────────────────────────────────────────

    @app.get("/v1/auth/google")
    async def google_start(device_code: str = ""):
        if not GOOGLE_CLIENT_ID:
            raise HTTPException(501, "Google OAuth nicht konfiguriert")
        state = device_code or secrets.token_hex(16)
        params = urllib.parse.urlencode({
            "client_id": GOOGLE_CLIENT_ID,
            "redirect_uri": f"{BASE_URL}/v1/auth/google/callback",
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
        })
        return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")

    @app.get("/v1/auth/google/callback")
    async def google_callback(code: str, state: str = ""):
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            raise HTTPException(501, "Google OAuth nicht konfiguriert")
        token_data = _oauth_request("https://oauth2.googleapis.com/token", {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": f"{BASE_URL}/v1/auth/google/callback",
        })
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(400, "Google auth failed")

        user_info = _oauth_request("https://www.googleapis.com/oauth2/v2/userinfo",
                                   headers={"Authorization": f"Bearer {access_token}"})
        g_id = str(user_info.get("id", ""))
        g_name = user_info.get("name", "").replace(" ", "_").lower()
        g_email = user_info.get("email", "")

        user = _find_or_create_oauth_user("google", g_id, g_name, g_email)

        if state and len(state) == 32:
            _complete_device_flow_by_device_code(state, user)

        return _auth_success_page(user)

    # ── Auth: Complete Device via device_code ──────────────────────

    @app.get("/v1/auth/device/{device_code}/complete")
    async def device_complete_via_login(device_code: str, username: str = "", password: str = ""):
        """Browser calls this after email/password login to complete device flow."""
        if not username or not password:
            raise HTTPException(400, "Username und Passwort erforderlich")
        row = db.execute(
            "SELECT id, username, password_hash, api_key FROM users WHERE username=?",
            (username,)
        ).fetchone()
        if not row or not _check_pw(password, row[2]):
            raise HTTPException(401, "Falscher Username oder Passwort")

        user = {"id": row[0], "username": row[1], "api_key": row[3]}
        _complete_device_flow_by_device_code(device_code, user)
        return _auth_success_page(user)

    def _complete_device_flow_by_device_code(device_code: str, user: dict):
        """Complete device flow using the device_code directly."""
        row = db.execute(
            "SELECT code, created_at FROM device_codes WHERE code=? AND status='pending'",
            (device_code,)
        ).fetchone()
        if row and time.time() - row[1] < 600:
            db.execute(
                "UPDATE device_codes SET status='complete', user_id=?, api_key=? WHERE code=?",
                (user["id"], user.get("api_key", ""), row[0])
            )
            db.commit()

    def _auth_success_page(user: dict) -> HTMLResponse:
        """Return a nice success page after login."""
        return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>DeepNetz — Login erfolgreich</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #0d1117; color: #e6edf3;
  display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
.card {{ text-align: center; max-width: 400px; padding: 40px; background: #161b22;
  border: 1px solid #30363d; border-radius: 16px; }}
h2 {{ color: #3fb950; margin-bottom: 8px; }}
p {{ color: #8b949e; font-size: 14px; }}
code {{ background: #21262d; padding: 4px 10px; border-radius: 4px; font-size: 13px; color: #58a6ff; }}
</style></head><body>
<div class="card">
  <h2>Login erfolgreich</h2>
  <p>Willkommen, <strong>{user['username']}</strong>!</p>
  <p style="margin-top:16px;font-size:12px;color:#6e7681;">
    Dein API-Key wurde an das CLI übermittelt.<br>
    Du kannst dieses Fenster schließen.
  </p>
</div>
</body></html>""")

    # ── Search ─────────────────────────────────────────────────────

    @app.get("/v1/search")
    async def search_models(q: str, limit: int = 15, user=Depends(_get_user)):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            results = list(api.list_models(search=f"{q} GGUF", sort="downloads", direction=-1, limit=limit))
        except Exception as e:
            raise HTTPException(502, f"HuggingFace nicht erreichbar: {e}")
        return {"query": q, "results": [
            {"repo": r.modelId, "downloads": getattr(r, "downloads", 0), "likes": getattr(r, "likes", 0)}
            for r in results
        ], "total": len(results)}

    # ── Model Info ─────────────────────────────────────────────────

    @app.get("/v1/models/{repo:path}/files")
    async def model_files(repo: str, user=Depends(_get_user)):
        try:
            from huggingface_hub import HfApi
            all_files = list(HfApi().list_repo_tree(repo, recursive=True))
            return {"repo": repo, "files": [
                {"name": f.rfilename, "size": getattr(f, "size", 0)}
                for f in all_files if f.rfilename.endswith(".gguf")
            ]}
        except Exception:
            raise HTTPException(404, f"Repo nicht gefunden: {repo}")

    @app.get("/v1/models/{repo:path}")
    async def model_info(repo: str, user=Depends(_get_user)):
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(repo)
            return {"repo": repo, "downloads": getattr(info, "downloads", 0),
                    "likes": getattr(info, "likes", 0), "tags": getattr(info, "tags", [])}
        except Exception:
            raise HTTPException(404, f"Modell nicht gefunden: {repo}")

    # ── Pull Logging ───────────────────────────────────────────────

    class PullLog(PydanticBaseModel):
        model: str
        quant: str = ""
        repo: str = ""

    @app.post("/v1/pulls")
    async def log_pull(req: PullLog, user=Depends(_get_user)):
        user_id = user["id"] if user else "anonymous"
        db.execute("INSERT INTO pull_log (user_id, model, quant, repo, ts) VALUES (?,?,?,?,?)",
                   (user_id, req.model, req.quant, req.repo, time.time()))
        db.commit()
        return {"status": "ok"}

    # ── Health ─────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        user_count = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        pull_count = db.execute("SELECT COUNT(*) FROM pull_log").fetchone()[0]
        return {"status": "ok", "version": "1.1.0", "users": user_count, "pulls": pull_count,
                "oauth": {"github": bool(GITHUB_CLIENT_ID), "google": bool(GOOGLE_CLIENT_ID)}}

    return app


def _create_standalone_app():
    db = os.environ.get("DEEPNETZ_DB_PATH", "")
    return create_registry_app(db_path=db)

app = _create_standalone_app()
