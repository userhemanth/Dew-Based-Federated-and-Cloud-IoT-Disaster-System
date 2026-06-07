# db_manager.py
"""
Cloud Database Manager for Dew-Based Federated Learning.

Manages two SQLite tables:
  - offline_buffer  : per-client local alert queue (edge/dew layer)
  - cloud_alerts    : synchronized records visible to all clients (cloud layer)

Usage:
  from db_manager import init_db, buffer_local_alert, sync_to_cloud, get_cloud_alerts
"""

import sqlite3
import os
import time
from pathlib import Path

DB_PATH = Path("data/cloud_database.sqlite")


# ─────────────────────────────────────────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the database and tables if they don't already exist."""
    os.makedirs("data", exist_ok=True)
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS offline_buffer (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id    TEXT    NOT NULL,
                device_name  TEXT,
                timestamp    TEXT    NOT NULL,
                label        TEXT    NOT NULL,
                confidence   REAL    NOT NULL,
                latitude     REAL,
                longitude    REAL,
                gps_accuracy REAL,
                synced       INTEGER NOT NULL DEFAULT 0   -- 0=pending, 1=synced
            );

            CREATE TABLE IF NOT EXISTS cloud_alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id    TEXT    NOT NULL,
                device_name  TEXT,
                timestamp    TEXT    NOT NULL,
                label        TEXT    NOT NULL,
                confidence   REAL    NOT NULL,
                latitude     REAL,
                longitude    REAL,
                gps_accuracy REAL,
                synced_at    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS round_sync (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                round_number    INTEGER NOT NULL,
                avg_accuracy    REAL,
                num_clients     INTEGER,
                synced_at       TEXT NOT NULL
            );
        """)
        
        # Add image_path column if it doesn't exist (for backward compatibility)
        try:
            conn.execute("ALTER TABLE offline_buffer ADD COLUMN image_path TEXT;")
        except sqlite3.OperationalError:
            pass # Column already exists



# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE BUFFER — Edge / Dew Layer
# ─────────────────────────────────────────────────────────────────────────────

def buffer_local_alert(
    device_id: str, 
    label: str, 
    confidence: float, 
    device_name: str = None,
    latitude: float = None,
    longitude: float = None,
    gps_accuracy: float = None,
    captured_at: str = None,
    image_path: str = None
) -> int:
    """
    Store a new alert in the local offline buffer.
    Returns the new row ID.
    """
    ts = captured_at if captured_at else time.strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO offline_buffer 
               (device_id, device_name, timestamp, label, confidence, latitude, longitude, gps_accuracy, image_path) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (device_id, device_name, ts, label, round(confidence, 6), latitude, longitude, gps_accuracy, image_path),
        )
        return cur.lastrowid


def get_pending_alerts(device_id: str = None) -> list[dict]:
    """
    Return all un-synced alerts.
    If device_id is given, filter to that device only.
    """
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        if device_id is not None:
            rows = conn.execute(
                "SELECT * FROM offline_buffer WHERE synced=0 AND device_id=? ORDER BY id",
                (device_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM offline_buffer WHERE synced=0 ORDER BY id"
            ).fetchall()
    return [dict(r) for r in rows]


def get_all_buffered(device_id: str = None) -> list[dict]:
    """Return all buffered alerts (synced or not)."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        if device_id is not None:
            rows = conn.execute(
                "SELECT * FROM offline_buffer WHERE device_id=? ORDER BY id DESC",
                (device_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM offline_buffer ORDER BY id DESC"
            ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# CLOUD SYNC — Cloud Layer
# ─────────────────────────────────────────────────────────────────────────────

def sync_to_cloud() -> int:
    """
    Push all pending offline_buffer rows into cloud_alerts and mark them synced.
    Returns the number of rows synced.
    """
    pending = get_pending_alerts()
    if not pending:
        return 0

    synced_at = time.strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        for row in pending:
            conn.execute(
                """INSERT INTO cloud_alerts
                   (device_id, device_name, timestamp, label, confidence, latitude, longitude, gps_accuracy, synced_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (row["device_id"], row["device_name"], row["timestamp"], row["label"],
                 row["confidence"], row["latitude"], row["longitude"], row["gps_accuracy"], synced_at),
            )
            conn.execute(
                "UPDATE offline_buffer SET synced=1 WHERE id=?",
                (row["id"],),
            )
    return len(pending)


def get_cloud_alerts(limit: int = 200) -> list[dict]:
    """Return the most recent cloud-synced alerts."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM cloud_alerts ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def sync_round_to_cloud(round_number: int, avg_accuracy: float | None, num_clients: int) -> None:
    """Store a FL round summary in the cloud database."""
    synced_at = time.strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.execute(
            """INSERT INTO round_sync (round_number, avg_accuracy, num_clients, synced_at)
               VALUES (?, ?, ?, ?)""",
            (round_number, avg_accuracy, num_clients, synced_at),
        )


def get_cloud_rounds(limit: int = 50) -> list[dict]:
    """Return federated round summaries from the cloud database."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM round_sync ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_alert_synced_to_aws(alert_id: int, image_url: str) -> None:
    """
    Mark a single alert as synced after successful AWS upload.
    Copies it to cloud_alerts and sets synced=1 in offline_buffer.
    """
    synced_at = time.strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM offline_buffer WHERE id=?", (alert_id,)).fetchone()
        if row:
            conn.execute(
                """INSERT INTO cloud_alerts
                   (device_id, device_name, timestamp, label, confidence, latitude, longitude, gps_accuracy, synced_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (row["device_id"], row["device_name"], row["timestamp"], row["label"],
                 row["confidence"], row["latitude"], row["longitude"], row["gps_accuracy"], synced_at)
            )
            conn.execute("UPDATE offline_buffer SET synced=1 WHERE id=?", (alert_id,))


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def cloud_stats() -> dict:
    """Return aggregate statistics from the cloud database."""
    with _connect() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM cloud_alerts").fetchone()[0]
        pending  = conn.execute("SELECT COUNT(*) FROM offline_buffer WHERE synced=0").fetchone()[0]
        buffered = conn.execute("SELECT COUNT(*) FROM offline_buffer").fetchone()[0]
        top_label = conn.execute(
            "SELECT label, COUNT(*) as cnt FROM cloud_alerts GROUP BY label ORDER BY cnt DESC LIMIT 1"
        ).fetchone()
    return {
        "total_cloud_alerts": total,
        "pending_sync":       pending,
        "total_buffered":     buffered,
        "top_label":          top_label[0] if top_label else "N/A",
        "top_label_count":    top_label[1] if top_label else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL
# ─────────────────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")   # safe for multi-reader use
    return conn


# Auto-initialise when imported
init_db()
