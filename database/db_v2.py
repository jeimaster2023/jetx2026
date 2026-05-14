"""
database/db_v2.py
═══════════════════════════════════════════════════════════════════════════════
Extensión de la DB para el sistema dual engine v3.

Tablas adicionales:
  - entropy_snapshots       → historial de entropía
  - binary_sequence_events  → eventos de secuencia binaria raros
  - clustering_snapshots    → estado de clustering cada N rondas
  - dual_engine_signals     → señales del motor dual con metadata completa
  - fibonacci_events        → coincidencias Fibonacci detectadas
  - mode_effectiveness      → efectividad por modo y por target
  - spike_distance_history  → distancias acumuladas para Fibonacci

Todo acceso asíncrono vía aiosqlite.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiosqlite

from utils.logger import get_logger, log_error

log = get_logger("db_v2")

_DDL_V2 = """
-- Snapshots de entropía
CREATE TABLE IF NOT EXISTS entropy_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    entropy_score   REAL    NOT NULL,
    label           TEXT    NOT NULL,
    compression_ratio REAL  NOT NULL DEFAULT 0,
    run_length_score  REAL  NOT NULL DEFAULT 0,
    is_dead_market  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_es_ts ON entropy_snapshots(ts);

-- Eventos de secuencia binaria relevantes
CREATE TABLE IF NOT EXISTS binary_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    rareness_score  REAL    NOT NULL,
    longest_run_0   INTEGER NOT NULL,
    longest_run_1   INTEGER NOT NULL,
    density_0       REAL    NOT NULL,
    density_1       REAL    NOT NULL,
    micro_cycles    TEXT    NOT NULL DEFAULT '[]',
    sequence_30     TEXT    NOT NULL DEFAULT '[]'  -- JSON últimas 30 rondas binarias
);
CREATE INDEX IF NOT EXISTS idx_be_ts ON binary_events(ts);

-- Snapshots de clustering
CREATE TABLE IF NOT EXISTS clustering_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    context_label   TEXT    NOT NULL,
    context_score   REAL    NOT NULL,
    anomaly_score   REAL    NOT NULL,
    is_anomaly      INTEGER NOT NULL DEFAULT 0,
    cluster_id      INTEGER NOT NULL DEFAULT 0,
    similar_contexts INTEGER NOT NULL DEFAULT 0,
    buffer_size     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cs_ts ON clustering_snapshots(ts);
CREATE INDEX IF NOT EXISTS idx_cs_label ON clustering_snapshots(context_label);

-- Señales del motor dual con metadata completa
CREATE TABLE IF NOT EXISTS dual_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    mode            TEXT    NOT NULL,   -- safe / hunter / moon
    signal_type     TEXT    NOT NULL,   -- safe / high_risk / moon
    target          REAL    NOT NULL,
    safe_score      INTEGER NOT NULL DEFAULT 0,
    hunter_score    INTEGER NOT NULL DEFAULT 0,
    moon_score      INTEGER NOT NULL DEFAULT 0,
    entropy_score   REAL    NOT NULL DEFAULT 0,
    tension_score   REAL    NOT NULL DEFAULT 0,
    compression_score REAL  NOT NULL DEFAULT 0,
    confidence      INTEGER NOT NULL DEFAULT 0,
    binary_rareness REAL    NOT NULL DEFAULT 0,
    anomaly_score   REAL    NOT NULL DEFAULT 0,
    fibonacci_note  TEXT    NOT NULL DEFAULT '',
    context_logs    TEXT    NOT NULL DEFAULT '[]',
    -- Resolución
    result          REAL,
    won             INTEGER,
    rounds_passed   INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ds_ts      ON dual_signals(ts);
CREATE INDEX IF NOT EXISTS idx_ds_mode    ON dual_signals(mode);
CREATE INDEX IF NOT EXISTS idx_ds_target  ON dual_signals(target);

-- Coincidencias Fibonacci relevantes
CREATE TABLE IF NOT EXISTS fibonacci_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    coincidence_ratio REAL  NOT NULL,
    golden_pairs    INTEGER NOT NULL DEFAULT 0,
    expansion       INTEGER NOT NULL DEFAULT 0,
    compression     INTEGER NOT NULL DEFAULT 1,
    note            TEXT    NOT NULL DEFAULT ''
);

-- Efectividad acumulada por modo y target
CREATE TABLE IF NOT EXISTS mode_effectiveness (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mode            TEXT    NOT NULL,
    target          REAL    NOT NULL,
    total           INTEGER NOT NULL DEFAULT 0,
    wins            INTEGER NOT NULL DEFAULT 0,
    winrate         REAL    NOT NULL DEFAULT 0,
    avg_score       REAL    NOT NULL DEFAULT 0,
    last_updated    TEXT    NOT NULL DEFAULT '',
    UNIQUE(mode, target)
);

-- Historial de distancias de spikes para Fibonacci (extendido)
CREATE TABLE IF NOT EXISTS spike_distance_history (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT    NOT NULL,
    level   REAL    NOT NULL,
    distance INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sdh_level ON spike_distance_history(level);
"""


async def init_db_v2(conn: aiosqlite.Connection) -> None:
    """Aplica el DDL v2 sobre la conexión existente."""
    try:
        await conn.executescript(_DDL_V2)
        await conn.commit()
        log.info("Tablas DB v2 inicializadas correctamente.")
    except Exception as e:
        log_error("init_db_v2 falló", e)
        raise


# ── Entropía ──────────────────────────────────────────────────────────────────

async def insert_entropy_snapshot(
    conn: aiosqlite.Connection,
    ts: datetime,
    entropy_score: float,
    label: str,
    compression_ratio: float,
    run_length_score: float,
    is_dead_market: bool,
) -> None:
    try:
        await conn.execute(
            """INSERT INTO entropy_snapshots
               (ts, entropy_score, label, compression_ratio, run_length_score, is_dead_market)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ts.isoformat(), entropy_score, label, compression_ratio,
             run_length_score, 1 if is_dead_market else 0),
        )
        await conn.commit()
    except Exception as e:
        log_error("insert_entropy_snapshot falló", e)


# ── Binary events ──────────────────────────────────────────────────────────────

async def insert_binary_event(
    conn: aiosqlite.Connection,
    ts: datetime,
    rareness_score: float,
    longest_run_0: int,
    longest_run_1: int,
    density_0: float,
    density_1: float,
    micro_cycles: list[str],
    sequence_30: list[int],
) -> None:
    try:
        await conn.execute(
            """INSERT INTO binary_events
               (ts, rareness_score, longest_run_0, longest_run_1,
                density_0, density_1, micro_cycles, sequence_30)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts.isoformat(), rareness_score, longest_run_0, longest_run_1,
             density_0, density_1,
             json.dumps(micro_cycles), json.dumps(sequence_30)),
        )
        await conn.commit()
    except Exception as e:
        log_error("insert_binary_event falló", e)


# ── Clustering ────────────────────────────────────────────────────────────────

async def insert_clustering_snapshot(
    conn: aiosqlite.Connection,
    ts: datetime,
    context_label: str,
    context_score: float,
    anomaly_score: float,
    is_anomaly: bool,
    cluster_id: int,
    similar_contexts: int,
    buffer_size: int,
) -> None:
    try:
        await conn.execute(
            """INSERT INTO clustering_snapshots
               (ts, context_label, context_score, anomaly_score,
                is_anomaly, cluster_id, similar_contexts, buffer_size)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts.isoformat(), context_label, context_score, anomaly_score,
             1 if is_anomaly else 0, cluster_id, similar_contexts, buffer_size),
        )
        await conn.commit()
    except Exception as e:
        log_error("insert_clustering_snapshot falló", e)


# ── Dual signals ──────────────────────────────────────────────────────────────

async def insert_dual_signal(
    conn: aiosqlite.Connection,
    ts: datetime,
    mode: str,
    signal_type: str,
    target: float,
    safe_score: int,
    hunter_score: int,
    moon_score: int,
    entropy_score: float,
    tension_score: float,
    compression_score: float,
    confidence: int,
    binary_rareness: float,
    anomaly_score: float,
    fibonacci_note: str,
    context_logs: list[str],
) -> int:
    try:
        async with conn.execute(
            """INSERT INTO dual_signals
               (ts, mode, signal_type, target, safe_score, hunter_score, moon_score,
                entropy_score, tension_score, compression_score, confidence,
                binary_rareness, anomaly_score, fibonacci_note, context_logs)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts.isoformat(), mode, signal_type, target,
                safe_score, hunter_score, moon_score,
                entropy_score, tension_score, compression_score, confidence,
                binary_rareness, anomaly_score, fibonacci_note,
                json.dumps(context_logs[:10]),
            ),
        ) as cur:
            row_id = cur.lastrowid
        await conn.commit()
        log.info("[DUAL SIGNAL] id=%d mode=%s target=%.2fx conf=%d%%",
                 row_id, mode, target, confidence)
        return row_id
    except Exception as e:
        log_error("insert_dual_signal falló", e)
        return -1


async def resolve_dual_signal(
    conn: aiosqlite.Connection,
    signal_id: int,
    result_val: float,
    won: bool,
    rounds_passed: int,
) -> None:
    try:
        await conn.execute(
            "UPDATE dual_signals SET result=?, won=?, rounds_passed=? WHERE id=?",
            (result_val, 1 if won else 0, rounds_passed, signal_id),
        )
        await conn.commit()
    except Exception as e:
        log_error("resolve_dual_signal falló", e)


async def get_pending_dual_signals(conn: aiosqlite.Connection) -> list[dict]:
    try:
        async with conn.execute(
            "SELECT id, target, mode, rounds_passed FROM dual_signals WHERE result IS NULL"
        ) as cur:
            rows = await cur.fetchall()
        return [{"id": r[0], "target": r[1], "mode": r[2], "rounds_passed": r[3]}
                for r in rows]
    except Exception as e:
        log_error("get_pending_dual_signals falló", e)
        return []


# ── Fibonacci events ──────────────────────────────────────────────────────────

async def insert_fibonacci_event(
    conn: aiosqlite.Connection,
    ts: datetime,
    coincidence_ratio: float,
    golden_pairs: int,
    expansion: bool,
    compression: bool,
    note: str,
) -> None:
    try:
        await conn.execute(
            """INSERT INTO fibonacci_events
               (ts, coincidence_ratio, golden_pairs, expansion, compression, note)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ts.isoformat(), coincidence_ratio, golden_pairs,
             1 if expansion else 0, 1 if compression else 0, note),
        )
        await conn.commit()
    except Exception as e:
        log_error("insert_fibonacci_event falló", e)


# ── Mode effectiveness ────────────────────────────────────────────────────────

async def update_mode_effectiveness(
    conn: aiosqlite.Connection,
    mode: str,
    target: float,
    won: bool,
    score: int,
) -> None:
    """Actualiza la tabla de efectividad por modo/target."""
    try:
        # Leer estado actual
        async with conn.execute(
            "SELECT total, wins, avg_score FROM mode_effectiveness WHERE mode=? AND target=?",
            (mode, target),
        ) as cur:
            row = await cur.fetchone()

        ts_now = datetime.now().isoformat()
        if row:
            total = row[0] + 1
            wins  = row[1] + (1 if won else 0)
            winrate = round(wins / total * 100, 1)
            avg_sc = round((row[2] * row[0] + score) / total, 1)
            await conn.execute(
                """UPDATE mode_effectiveness
                   SET total=?, wins=?, winrate=?, avg_score=?, last_updated=?
                   WHERE mode=? AND target=?""",
                (total, wins, winrate, avg_sc, ts_now, mode, target),
            )
        else:
            await conn.execute(
                """INSERT INTO mode_effectiveness
                   (mode, target, total, wins, winrate, avg_score, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (mode, target, 1, 1 if won else 0, 100.0 if won else 0.0, float(score), ts_now),
            )
        await conn.commit()
    except Exception as e:
        log_error("update_mode_effectiveness falló", e)


@dataclass
class ModeEffectivenessStats:
    mode: str
    target: float
    total: int
    wins: int
    winrate: float
    avg_score: float


async def get_mode_effectiveness(
    conn: aiosqlite.Connection
) -> list[ModeEffectivenessStats]:
    """Lee la efectividad acumulada por modo y target."""
    try:
        async with conn.execute(
            "SELECT mode, target, total, wins, winrate, avg_score FROM mode_effectiveness ORDER BY mode, target"
        ) as cur:
            rows = await cur.fetchall()
        return [ModeEffectivenessStats(*r) for r in rows]
    except Exception as e:
        log_error("get_mode_effectiveness falló", e)
        return []


# ── Spike distances para Fibonacci ───────────────────────────────────────────

async def append_spike_distance(
    conn: aiosqlite.Connection,
    ts: datetime,
    level: float,
    distance: int,
) -> None:
    """Guarda distancia en la tabla de historial para Fibonacci."""
    try:
        await conn.execute(
            "INSERT INTO spike_distance_history (ts, level, distance) VALUES (?, ?, ?)",
            (ts.isoformat(), level, distance),
        )
        await conn.commit()
    except Exception as e:
        log_error("append_spike_distance falló", e)


async def get_spike_distances_for_fibonacci(
    conn: aiosqlite.Connection,
    levels: list[float] = None,
    limit: int = 30,
) -> dict[float, list[int]]:
    """
    Lee las últimas `limit` distancias por nivel para el análisis Fibonacci.
    Retorna dict nivel → lista de distancias.
    """
    if levels is None:
        levels = [5.0, 10.0, 20.0, 50.0]

    result: dict[float, list[int]] = {}
    for level in levels:
        try:
            async with conn.execute(
                """SELECT distance FROM spike_distance_history
                   WHERE level=? ORDER BY ts DESC LIMIT ?""",
                (level, limit),
            ) as cur:
                rows = await cur.fetchall()
            result[level] = [r[0] for r in rows]
        except Exception as e:
            log_error(f"get_spike_distances_for_fibonacci(level={level}) falló", e)
            result[level] = []

    return result


# ── Métricas globales ─────────────────────────────────────────────────────────

async def get_dual_engine_stats(conn: aiosqlite.Connection) -> dict:
    """Estadísticas completas del motor dual."""
    try:
        stats: dict = {}
        for mode in ("safe", "hunter", "moon"):
            async with conn.execute(
                """SELECT COUNT(*), SUM(won), AVG(confidence),
                          AVG(entropy_score), AVG(compression_score), AVG(tension_score)
                   FROM dual_signals
                   WHERE mode=? AND result IS NOT NULL""",
                (mode,),
            ) as cur:
                row = await cur.fetchone()

            total = row[0] or 0
            wins  = row[1] or 0
            stats[mode] = {
                "total": total,
                "wins": wins,
                "winrate": round(wins / total * 100, 1) if total > 0 else 0.0,
                "avg_confidence": round(row[2] or 0, 1),
                "avg_entropy": round(row[3] or 0, 4),
                "avg_compression": round(row[4] or 0, 4),
                "avg_tension": round(row[5] or 0, 1),
            }

        # Estadísticas globales
        async with conn.execute(
            "SELECT COUNT(*), SUM(won) FROM dual_signals WHERE result IS NOT NULL"
        ) as cur:
            row = await cur.fetchone()
        total_global = row[0] or 0
        wins_global  = row[1] or 0
        stats["global"] = {
            "total": total_global,
            "wins": wins_global,
            "winrate": round(wins_global / total_global * 100, 1) if total_global > 0 else 0.0,
        }

        return stats
    except Exception as e:
        log_error("get_dual_engine_stats falló", e)
        return {}
