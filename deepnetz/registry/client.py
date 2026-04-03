"""
Registry Client — alle Anfragen gehen über registry.deepnetz.com.

Flow:
    1. User registriert sich:  deepnetz register
    2. User loggt sich ein:    deepnetz login
    3. API-Key wird lokal gespeichert
    4. Alle Anfragen nutzen den Key:
       - deepnetz search Qwen      → GET /v1/search?q=Qwen
       - deepnetz pull Qwen3.5-35B → GET /v1/search → Download von HF
"""

import json
import os
import urllib.request
import urllib.error
from typing import Optional, List, Dict

# Produktiv: https://registry.deepnetz.com
# Lokal:     http://localhost:8090
DEFAULT_REGISTRY = "https://registry.deepnetz.com"


class RegistryClient:
    """Client für den DeepNetz Registry Server."""

    def __init__(self, registry_url: str = ""):
        self._url = (
            registry_url
            or os.environ.get("DEEPNETZ_REGISTRY", "")
            or DEFAULT_REGISTRY
        ).rstrip("/")
        self._timeout = 10
        self._api_key = self._load_api_key()

    # ── Auth ───────────────────────────────────────────────────────

    def register(self, username: str, password: str) -> Dict:
        """Registrierung auf dem Registry Server."""
        resp = self._request("/v1/auth/register", method="POST", data={
            "username": username,
            "password": password,
        })
        if resp and "api_key" in resp:
            self._save_api_key(resp["api_key"])
            self._api_key = resp["api_key"]
        return resp or {}

    def login(self, username: str, password: str) -> Dict:
        """Login → API-Key wird lokal gespeichert."""
        resp = self._request("/v1/auth/login", method="POST", data={
            "username": username,
            "password": password,
        })
        if resp and "api_key" in resp:
            self._save_api_key(resp["api_key"])
            self._api_key = resp["api_key"]
        return resp or {}

    def me(self) -> Optional[Dict]:
        """Aktuellen User abfragen."""
        return self._request("/v1/auth/me")

    @property
    def is_authenticated(self) -> bool:
        return bool(self._api_key)

    # ── Search ─────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 15) -> List[Dict]:
        """Modell-Suche über den Registry Server."""
        resp = self._request(f"/v1/search?q={query}&limit={limit}")
        if resp:
            return resp.get("results", [])
        return []

    # ── Model Info ─────────────────────────────────────────────────

    def model_info(self, repo: str) -> Optional[Dict]:
        return self._request(f"/v1/models/{repo}")

    def model_files(self, repo: str) -> List[Dict]:
        resp = self._request(f"/v1/models/{repo}/files")
        if resp:
            return resp.get("files", [])
        return []

    # ── Pull Logging ───────────────────────────────────────────────

    def log_pull(self, model: str, quant: str = "", repo: str = ""):
        """Pull an den Server melden (Analytics)."""
        self._request("/v1/pulls", method="POST", data={
            "model": model, "quant": quant, "repo": repo,
        })

    # ── Health ─────────────────────────────────────────────────────

    def is_available(self) -> bool:
        resp = self._request("/health")
        return resp is not None and resp.get("status") == "ok"

    # ── Internal ───────────────────────────────────────────────────

    def _request(self, endpoint: str, method: str = "GET",
                 data: dict = None) -> Optional[dict]:
        url = f"{self._url}{endpoint}"
        headers = {
            "User-Agent": "deepnetz/1.0",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            if data:
                body = json.dumps(data).encode()
                headers["Content-Type"] = "application/json"
                req = urllib.request.Request(url, data=body, headers=headers, method=method)
            else:
                req = urllib.request.Request(url, headers=headers, method=method)

            resp = urllib.request.urlopen(req, timeout=self._timeout)
            return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode())
                detail = body.get("detail", str(e))
            except Exception:
                detail = str(e)
            raise RuntimeError(detail)
        except (urllib.error.URLError, OSError, TimeoutError):
            return None

    @staticmethod
    def _credentials_path() -> str:
        path = os.path.join(os.path.expanduser("~"), ".config", "deepnetz")
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, "credentials.json")

    def _load_api_key(self) -> str:
        path = self._credentials_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                return data.get("api_key", "")
            except (json.JSONDecodeError, OSError):
                pass
        return ""

    def _save_api_key(self, api_key: str):
        path = self._credentials_path()
        with open(path, "w") as f:
            json.dump({"api_key": api_key}, f)
        os.chmod(path, 0o600)  # Nur User lesbar
