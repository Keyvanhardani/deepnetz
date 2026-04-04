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
    db.execute("""CREATE TABLE IF NOT EXISTS config (
        key TEXT PRIMARY KEY,
        value TEXT DEFAULT ''
    )""")
    db.commit()

    # ── Config from DB (overrides env vars) ────────────────────────

    def _get_cfg(key: str, default: str = "") -> str:
        row = db.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
        if row and row[0]:
            return row[0]
        return default

    def _set_cfg(key: str, value: str):
        db.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?,?)", (key, value))
        db.commit()

    # Load OAuth from DB first, then env fallback
    def _github_id(): return _get_cfg("github_client_id", GITHUB_CLIENT_ID)
    def _github_secret(): return _get_cfg("github_client_secret", GITHUB_CLIENT_SECRET)
    def _google_id(): return _get_cfg("google_client_id", GOOGLE_CLIENT_ID)
    def _google_secret(): return _get_cfg("google_client_secret", GOOGLE_CLIENT_SECRET)
    def _base_url(): return _get_cfg("base_url", BASE_URL)

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
            "verification_url": f"{_base_url()}/auth?code={user_code}",
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
        if not _github_id():
            raise HTTPException(501, "GitHub OAuth nicht konfiguriert. Konfiguriere unter /admin")
        state = device_code or secrets.token_hex(16)
        params = urllib.parse.urlencode({
            "client_id": _github_id(),
            "redirect_uri": f"{_base_url()}/v1/auth/github/callback",
            "scope": "read:user user:email",
            "state": state,
        })
        return RedirectResponse(f"https://github.com/login/oauth/authorize?{params}")

    @app.get("/v1/auth/github/callback")
    async def github_callback(code: str, state: str = ""):
        if not _github_id() or not _github_secret():
            raise HTTPException(501, "GitHub OAuth nicht konfiguriert")
        # Exchange code for token
        token_data = _oauth_request("https://github.com/login/oauth/access_token", {
            "client_id": _github_id(),
            "client_secret": _github_secret(),
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
        if not _google_id():
            raise HTTPException(501, "Google OAuth nicht konfiguriert. Konfiguriere unter /admin")
        state = device_code or secrets.token_hex(16)
        params = urllib.parse.urlencode({
            "client_id": _google_id(),
            "redirect_uri": f"{_base_url()}/v1/auth/google/callback",
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
        })
        return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")

    @app.get("/v1/auth/google/callback")
    async def google_callback(code: str, state: str = ""):
        if not _google_id() or not _google_secret():
            raise HTTPException(501, "Google OAuth nicht konfiguriert")
        token_data = _oauth_request("https://oauth2.googleapis.com/token", {
            "client_id": _google_id(),
            "client_secret": _google_secret(),
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": f"{_base_url()}/v1/auth/google/callback",
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
        """Return success page after OAuth login — redirects to deepnetz.com with credentials."""
        import urllib.parse
        params = urllib.parse.urlencode({
            "username": user["username"],
            "apikey": user.get("api_key", ""),
        })
        redirect_url = f"https://deepnetz.com/auth/?{params}"
        return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>DeepNetz — Login erfolgreich</title>
<meta http-equiv="refresh" content="1;url={redirect_url}">
<style>
body {{ font-family: -apple-system, sans-serif; background: #0d1117; color: #e6edf3;
  display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
.card {{ text-align: center; max-width: 400px; padding: 40px; background: #161b22;
  border: 1px solid #30363d; border-radius: 16px; }}
h2 {{ color: #3fb950; margin-bottom: 8px; }}
p {{ color: #8b949e; font-size: 14px; }}
</style></head><body>
<div class="card">
  <h2>Willkommen, {user['username']}!</h2>
  <p>Weiterleitung...</p>
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
                "oauth": {"github": bool(_github_id()), "google": bool(_google_id())}}

    # ── Admin Panel (nur intern/VPN erreichbar + Passwort) ────────

    ADMIN_PASSWORD = _get_cfg("admin_password", os.environ.get("ADMIN_PASSWORD", ""))

    from fastapi import Request

    def _check_admin(request: Request):
        """Check admin auth via header or query param."""
        pw = ADMIN_PASSWORD or _get_cfg("admin_password", "")
        if not pw:
            # Kein Passwort gesetzt → Admin gesperrt bis eins gesetzt wird
            raise HTTPException(403, "Admin-Passwort nicht konfiguriert. Setze ADMIN_PASSWORD als Env-Var im Container.")
        token = request.headers.get("X-Admin-Token", "") or request.query_params.get("token", "")
        if token != pw:
            raise HTTPException(401, "Ungültiges Admin-Token")

    class ConfigUpdate(PydanticBaseModel):
        key: str
        value: str

    @app.get("/admin/api/config")
    async def admin_get_config(request: Request):
        _check_admin(request)
        rows = db.execute("SELECT key, value FROM config").fetchall()
        cfg = {r[0]: r[1] for r in rows}
        return cfg

    @app.post("/admin/api/config")
    async def admin_set_config(req: ConfigUpdate, request: Request):
        _check_admin(request)
        _set_cfg(req.key, req.value)
        return {"status": "ok", "key": req.key}

    @app.get("/admin/api/users")
    async def admin_list_users(request: Request):
        _check_admin(request)
        rows = db.execute(
            "SELECT id, username, email, oauth_provider, created_at, last_seen FROM users ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        return [{"id": r[0], "username": r[1], "email": r[2], "provider": r[3],
                 "created_at": r[4], "last_seen": r[5]} for r in rows]

    @app.delete("/admin/api/users/{user_id}")
    async def admin_delete_user(user_id: str, request: Request):
        _check_admin(request)
        db.execute("DELETE FROM users WHERE id=?", (user_id,))
        db.commit()
        return {"status": "ok"}

    @app.get("/admin/api/pulls")
    async def admin_list_pulls(request: Request):
        _check_admin(request)
        rows = db.execute(
            "SELECT user_id, model, quant, repo, ts FROM pull_log ORDER BY ts DESC LIMIT 100"
        ).fetchall()
        return [{"user_id": r[0], "model": r[1], "quant": r[2], "repo": r[3], "ts": r[4]} for r in rows]

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_page(request: Request):
        _check_admin(request)
        return HTMLResponse(_ADMIN_HTML)

    return app


def _create_standalone_app():
    db = os.environ.get("DEEPNETZ_DB_PATH", "")
    return create_registry_app(db_path=db)

_ADMIN_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DeepNetz Admin</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,sans-serif;background:#0d1117;color:#e6edf3;padding:24px;max-width:900px;margin:0 auto}
h1{font-size:22px;margin-bottom:24px;display:flex;align-items:center;gap:10px}
h1 span{color:#58a6ff}
h2{font-size:16px;margin:28px 0 12px;color:#8b949e;text-transform:uppercase;letter-spacing:0.05em;font-weight:600}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;margin-bottom:16px}
.field{display:flex;gap:10px;margin-bottom:10px;align-items:center}
.field label{width:180px;font-size:13px;color:#8b949e;flex-shrink:0}
.field input{flex:1;background:#21262d;border:1px solid #30363d;border-radius:6px;color:#e6edf3;padding:8px 12px;font-size:13px;outline:none}
.field input:focus{border-color:#58a6ff}
.field button{background:#238636;border:none;border-radius:6px;color:#fff;padding:8px 16px;font-size:13px;cursor:pointer;white-space:nowrap}
.field button:hover{background:#2ea043}
.status{font-size:12px;color:#3fb950;margin-left:8px;display:none}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:#8b949e;font-weight:500;padding:8px;border-bottom:1px solid #30363d}
td{padding:8px;border-bottom:1px solid #21262d;color:#e6edf3}
.del-btn{background:#da3633;border:none;border-radius:4px;color:#fff;padding:3px 10px;font-size:11px;cursor:pointer}
.del-btn:hover{background:#f85149}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:500}
.badge-gh{background:rgba(88,166,255,0.15);color:#58a6ff}
.badge-go{background:rgba(63,185,80,0.15);color:#3fb950}
.badge-pw{background:rgba(139,148,158,0.15);color:#8b949e}
.info{font-size:12px;color:#6e7681;margin-top:6px;line-height:1.5}
.info a{color:#58a6ff}
.tabs{display:flex;gap:4px;margin-bottom:20px}
.tab{padding:8px 16px;background:#21262d;border:1px solid #30363d;border-radius:8px;color:#8b949e;cursor:pointer;font-size:13px}
.tab.active{background:#30363d;color:#e6edf3;border-color:#484f58}
.panel{display:none}.panel.active{display:block}
</style></head><body>
<h1>DN <span>DeepNetz Admin</span></h1>

<div class="tabs">
  <div class="tab active" onclick="showPanel('config')">Config</div>
  <div class="tab" onclick="showPanel('users')">Users</div>
  <div class="tab" onclick="showPanel('pulls')">Pull Log</div>
</div>

<div class="panel active" id="panel-config">
  <h2>OAuth Configuration</h2>
  <div class="card">
    <div class="field"><label>Base URL</label><input id="c-base_url" placeholder="https://registry.deepnetz.com"><button onclick="save('base_url')">Save</button><span class="status" id="s-base_url">Saved</span></div>
    <div class="info">Callback-URLs für OAuth: <code>{base_url}/v1/auth/github/callback</code> und <code>{base_url}/v1/auth/google/callback</code></div>
  </div>

  <div class="card">
    <h2 style="margin-top:0">GitHub OAuth</h2>
    <div class="field"><label>Client ID</label><input id="c-github_client_id" placeholder="Ov23li..."><button onclick="save('github_client_id')">Save</button><span class="status" id="s-github_client_id">Saved</span></div>
    <div class="field"><label>Client Secret</label><input id="c-github_client_secret" type="password" placeholder="secret..."><button onclick="save('github_client_secret')">Save</button><span class="status" id="s-github_client_secret">Saved</span></div>
    <div class="info">
      Erstellen: <a href="https://github.com/settings/developers" target="_blank">github.com/settings/developers</a> &rarr; New OAuth App<br>
      Homepage URL: <code>https://deepnetz.com</code><br>
      Callback URL: <code>https://registry.deepnetz.com/v1/auth/github/callback</code>
    </div>
  </div>

  <div class="card">
    <h2 style="margin-top:0">Google OAuth</h2>
    <div class="field"><label>Client ID</label><input id="c-google_client_id" placeholder="123...apps.googleusercontent.com"><button onclick="save('google_client_id')">Save</button><span class="status" id="s-google_client_id">Saved</span></div>
    <div class="field"><label>Client Secret</label><input id="c-google_client_secret" type="password" placeholder="GOCSPX-..."><button onclick="save('google_client_secret')">Save</button><span class="status" id="s-google_client_secret">Saved</span></div>
    <div class="info">
      Erstellen: <a href="https://console.cloud.google.com/apis/credentials" target="_blank">Google Cloud Console</a> &rarr; Create Credentials &rarr; OAuth Client ID<br>
      Application type: Web application<br>
      Authorized redirect URI: <code>https://registry.deepnetz.com/v1/auth/google/callback</code>
    </div>
  </div>
</div>

<div class="panel" id="panel-users">
  <h2>Registered Users</h2>
  <div class="card"><table><thead><tr><th>Username</th><th>Email</th><th>Provider</th><th>Created</th><th>Last Seen</th><th></th></tr></thead><tbody id="users-table"></tbody></table></div>
</div>

<div class="panel" id="panel-pulls">
  <h2>Pull Log (last 100)</h2>
  <div class="card"><table><thead><tr><th>User</th><th>Model</th><th>Quant</th><th>Repo</th><th>Time</th></tr></thead><tbody id="pulls-table"></tbody></table></div>
</div>

<script>
function showPanel(name){
  document.querySelectorAll('.panel').forEach(function(p){p.classList.remove('active')});
  document.querySelectorAll('.tab').forEach(function(t){t.classList.remove('active')});
  document.getElementById('panel-'+name).classList.add('active');
  event.target.classList.add('active');
  if(name==='users')loadUsers();
  if(name==='pulls')loadPulls();
}
var adminToken=new URLSearchParams(window.location.search).get('token')||'';
function af(url,opts){opts=opts||{};opts.headers=opts.headers||{};opts.headers['X-Admin-Token']=adminToken;return fetch(url,opts);}
async function loadConfig(){
  var r=await af('/admin/api/config');var d=await r.json();
  for(var k in d){var el=document.getElementById('c-'+k);if(el)el.value=d[k];}
}
async function save(key){
  var val=document.getElementById('c-'+key).value;
  await af('/admin/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:key,value:val})});
  var s=document.getElementById('s-'+key);s.style.display='inline';setTimeout(function(){s.style.display='none'},2000);
}
async function loadUsers(){
  var r=await af('/admin/api/users');var d=await r.json();
  var el=document.getElementById('users-table');
  el.innerHTML=d.map(function(u){
    var p=u.provider==='github'?'<span class="badge badge-gh">GitHub</span>':u.provider==='google'?'<span class="badge badge-go">Google</span>':'<span class="badge badge-pw">Email</span>';
    var c=u.created_at?new Date(u.created_at*1000).toLocaleDateString():'—';
    var l=u.last_seen?new Date(u.last_seen*1000).toLocaleDateString():'—';
    return '<tr><td>'+u.username+'</td><td>'+(u.email||'—')+'</td><td>'+p+'</td><td>'+c+'</td><td>'+l+'</td><td><button class="del-btn" onclick="delUser(\\''+u.id+'\\')">Delete</button></td></tr>';
  }).join('');
}
async function delUser(id){
  if(!confirm('User löschen?'))return;
  await af('/admin/api/users/'+id,{method:'DELETE'});loadUsers();
}
async function loadPulls(){
  var r=await af('/admin/api/pulls');var d=await r.json();
  var el=document.getElementById('pulls-table');
  el.innerHTML=d.map(function(p){
    var t=p.ts?new Date(p.ts*1000).toLocaleString():'—';
    return '<tr><td>'+(p.user_id||'anon')+'</td><td>'+(p.model||'—')+'</td><td>'+(p.quant||'—')+'</td><td>'+(p.repo||'—')+'</td><td>'+t+'</td></tr>';
  }).join('');
}
loadConfig();
</script></body></html>"""

app = _create_standalone_app()
