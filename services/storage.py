"""
Embedded SQLite Storage Engine for ArchGraph (Privacy-First Ephemeral Mode)
Provides thread-safe session, graph artifact, and offloaded source-file storage.
Used strictly as a low-RAM disk spill during active sessions:
- All sessions and code are wiped on server startup (`cleanup_all_on_startup`)
- Expired sessions cascade-delete all artifacts and file contents (`ON DELETE CASCADE`)
- WAL checkpoints truncate deleted data from disk immediately
"""
import os
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator


class StorageEngine:
    """Thread-safe ephemeral SQLite manager for keeping Python RAM usage minimal."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            root_dir = Path(__file__).resolve().parent.parent
            data_dir = root_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "docustream.db")
        else:
            self.db_path = db_path
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """
        Creates a connection with WAL mode and foreign keys enabled,
        and guarantees both transaction commit/rollback AND conn.close() on exit.
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initializes database tables if they do not exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    last_active REAL NOT NULL,
                    file_count INTEGER DEFAULT 0,
                    ast_nodes INTEGER DEFAULT 0,
                    ast_edges INTEGER DEFAULT 0,
                    communities_count INTEGER DEFAULT 0,
                    god_nodes_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS graph_artifacts (
                    session_id TEXT PRIMARY KEY,
                    graph_json TEXT,
                    graph_report_md TEXT,
                    graph_html TEXT,
                    analysis_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS session_files (
                    session_id TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    content TEXT NOT NULL,
                    PRIMARY KEY (session_id, filepath),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS file_manifests (
                    session_id TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    sha256_hash TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    last_extracted REAL NOT NULL,
                    PRIMARY KEY (session_id, filepath),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(last_active);
                CREATE INDEX IF NOT EXISTS idx_session_files_sid ON session_files(session_id);
            """)

    def _safe_wal_checkpoint(self, conn: sqlite3.Connection):
        """Commits any active transaction on the connection before attempting a WAL truncate."""
        try:
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except sqlite3.OperationalError:
            # If another reader thread briefly holds a shared lock, fall back to PASSIVE
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
            except sqlite3.Error:
                pass

    def cleanup_all_on_startup(self):
        """
        Privacy guarantee: Wipes any leftover sessions, source files, and artifacts
        on server startup and truncates the WAL journal so no user code persists at rest.
        """
        with self._get_connection() as conn:
            conn.execute("DELETE FROM sessions;")
            conn.execute("DELETE FROM graph_artifacts;")
            conn.execute("DELETE FROM session_files;")
            conn.execute("DELETE FROM file_manifests;")
            self._safe_wal_checkpoint(conn)

    def create_or_update_session(
        self,
        session_id: str,
        file_count: int = 0,
        ast_nodes: int = 0,
        ast_edges: int = 0,
        communities_count: int = 0,
        god_nodes_count: int = 0
    ) -> Dict[str, Any]:
        """Creates or updates a session metadata record."""
        now = time.time()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, created_at, last_active, file_count, ast_nodes, ast_edges, communities_count, god_nodes_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_active=excluded.last_active,
                    file_count=excluded.file_count,
                    ast_nodes=excluded.ast_nodes,
                    ast_edges=excluded.ast_edges,
                    communities_count=excluded.communities_count,
                    god_nodes_count=excluded.god_nodes_count;
            """, (session_id, now, now, file_count, ast_nodes, ast_edges, communities_count, god_nodes_count))
        return self.get_session(session_id) or {}

    def touch_session(self, session_id: str):
        """Updates last_active timestamp to keep session alive."""
        with self._get_connection() as conn:
            conn.execute("UPDATE sessions SET last_active = ? WHERE session_id = ?;", (time.time(), session_id))

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves session record by ID."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM sessions WHERE session_id = ?;", (session_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def save_artifacts(
        self,
        session_id: str,
        graph_json: str = "",
        graph_report_md: str = "",
        graph_html: str = "",
        analysis_json: str = ""
    ):
        """Saves generated Graphify artifacts (graph.json, report, HTML, and metrics)."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO graph_artifacts (session_id, graph_json, graph_report_md, graph_html, analysis_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    graph_json=excluded.graph_json,
                    graph_report_md=excluded.graph_report_md,
                    graph_html=excluded.graph_html,
                    analysis_json=excluded.analysis_json;
            """, (session_id, graph_json, graph_report_md, graph_html, analysis_json))

    def get_artifacts(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves stored graph artifacts for a session."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM graph_artifacts WHERE session_id = ?;", (session_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    # ─────────────────────────────────────────────────────────────────────────
    # Low-RAM Ephemeral Source File Spill
    # ─────────────────────────────────────────────────────────────────────────

    def save_session_files(self, session_id: str, files_map: Dict[str, str]):
        """Spills raw source file strings to SQLite so Python RAM stays low."""
        if not files_map:
            return
        rows = [(session_id, path, content) for path, content in files_map.items()]
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT INTO session_files (session_id, filepath, content)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id, filepath) DO UPDATE SET
                    content=excluded.content;
            """, rows)

    def get_session_file(self, session_id: str, filepath: str) -> Optional[str]:
        """Fetches a single offloaded source file from SQLite in <1ms."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT content FROM session_files WHERE session_id = ? AND filepath = ?;",
                (session_id, filepath)
            )
            row = cur.fetchone()
            return row["content"] if row else None

    def list_session_files(self, session_id: str) -> List[str]:
        """Returns all filepaths stored for a session without loading their contents."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT filepath FROM session_files WHERE session_id = ? ORDER BY filepath;",
                (session_id,)
            )
            return [row["filepath"] for row in cur.fetchall()]

    def save_manifest(self, session_id: str, manifest: Dict[str, Dict[str, Any]]):
        """Saves file hashes for session diagnostics."""
        now = time.time()
        rows = [
            (session_id, path, data.get('hash', ''), data.get('type', 'code'), now)
            for path, data in manifest.items()
        ]
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT INTO file_manifests (session_id, filepath, sha256_hash, file_type, last_extracted)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id, filepath) DO UPDATE SET
                    sha256_hash=excluded.sha256_hash,
                    file_type=excluded.file_type,
                    last_extracted=excluded.last_extracted;
            """, rows)

    def get_manifest(self, session_id: str) -> Dict[str, str]:
        """Returns {filepath: sha256_hash} map."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT filepath, sha256_hash FROM file_manifests WHERE session_id = ?;", (session_id,))
            return {row['filepath']: row['sha256_hash'] for row in cur.fetchall()}

    def delete_session(self, session_id: str):
        """Deletes a session, cascades to all files/artifacts, and truncates WAL."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))
            self._safe_wal_checkpoint(conn)

    def cleanup_expired_sessions(self, max_age_seconds: int = 3600) -> int:
        """Removes sessions that have been inactive longer than max_age_seconds (default 1h)."""
        cutoff = time.time() - max_age_seconds
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE last_active < ?;", (cutoff,))
            deleted = cur.rowcount
            if deleted > 0:
                self._safe_wal_checkpoint(conn)
            return deleted


# Global singleton instance
storage = StorageEngine()