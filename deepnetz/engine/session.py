"""
Session Management — persistent conversation state.
"""

import json
import os
import time
import sqlite3
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Session:
    id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    title: str = ""
    config: Dict = field(default_factory=lambda: {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 512,
        "repeat_penalty": 1.1,
    })


class SessionStore:
    """SQLite-backed session storage."""

    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = os.path.join(os.path.expanduser("~"), ".cache",
                                   "deepnetz", "sessions.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                messages TEXT,
                config TEXT,
                created_at REAL,
                updated_at REAL,
                title TEXT DEFAULT ''
            )
        """)
        self._db.commit()
        # Migrate existing databases: add title column if missing
        try:
            self._db.execute("ALTER TABLE sessions ADD COLUMN title TEXT DEFAULT ''")
            self._db.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

    def create(self, session_id: str = "", title: str = "") -> Session:
        if not session_id:
            session_id = f"s_{int(time.time()*1000)}"
        s = Session(id=session_id, created_at=time.time(), updated_at=time.time(), title=title)
        self._db.execute(
            "INSERT OR REPLACE INTO sessions (id, messages, config, created_at, updated_at, title) VALUES (?,?,?,?,?,?)",
            (s.id, json.dumps(s.messages), json.dumps(s.config), s.created_at, s.updated_at, s.title)
        )
        self._db.commit()
        return s

    def get(self, session_id: str) -> Optional[Session]:
        row = self._db.execute(
            "SELECT id, messages, config, created_at, updated_at, title FROM sessions WHERE id=?",
            (session_id,)
        ).fetchone()
        if not row:
            return None
        return Session(
            id=row[0], messages=json.loads(row[1]),
            config=json.loads(row[2]),
            created_at=row[3], updated_at=row[4],
            title=row[5] or ""
        )

    def save(self, session: Session):
        session.updated_at = time.time()
        self._db.execute(
            "UPDATE sessions SET messages=?, config=?, updated_at=?, title=? WHERE id=?",
            (json.dumps(session.messages), json.dumps(session.config),
             session.updated_at, session.title, session.id)
        )
        self._db.commit()

    def list_sessions(self, limit: int = 50) -> List[Session]:
        """Return sessions without full message history (lightweight for sidebar)."""
        rows = self._db.execute(
            "SELECT id, title, created_at, updated_at, messages FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        results = []
        for r in rows:
            msg_count = len(json.loads(r[4])) if r[4] else 0
            results.append(Session(
                id=r[0], title=r[1] or "", created_at=r[2], updated_at=r[3],
                messages=[{}] * msg_count  # placeholder list for count only
            ))
        return results

    def add_message(self, session_id: str, role: str, content: str) -> Optional[Session]:
        """Append a message to a session and save."""
        s = self.get(session_id)
        if not s:
            return None
        s.messages.append({"role": role, "content": content})
        self.save(s)
        return s

    def update_title(self, session_id: str, title: str) -> Optional[Session]:
        """Update session title."""
        s = self.get(session_id)
        if not s:
            return None
        s.title = title
        self.save(s)
        return s

    def delete(self, session_id: str):
        self._db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        self._db.commit()

    def export_session(self, session_id: str, fmt: str = "json") -> str:
        s = self.get(session_id)
        if not s:
            return ""
        if fmt == "json":
            return json.dumps({"id": s.id, "messages": s.messages}, indent=2)
        elif fmt == "markdown":
            lines = [f"# Chat Session {s.id}\n"]
            for m in s.messages:
                role = "**User**" if m["role"] == "user" else "**Assistant**"
                lines.append(f"{role}: {m['content']}\n")
            return "\n".join(lines)
        return ""
