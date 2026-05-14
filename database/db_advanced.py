"""
database/db_advanced.py
═══════════════════════════════════════════════════════════════════════════════
Extensión de la capa de base de datos para el motor estadístico avanzado.

Tablas nuevas (SQLite):
  - spike_distances     → distancia entre spikes por nivel
  - tension_history     → historial del índice de tensión
  - compression_history → historial de compresión detectada
  - pre_spike_contexts  → contextos guardados antes de spikes confirmados
  - advanced_signals    → señales del motor avanzado con metadata completa

Funciones:
  - init_advanced_db()           → crea las tablas si no existen
  - insert_spike_distance()      → registra cada spike y su distancia al anterior
  - insert_tension_snapshot()    → guarda snapshot de tensión por ronda
  - insert_compression_event()   → guarda evento de compresión detectada
  - save_pre_spike_context()     → guarda el contexto previo a un spike confirmado
  - insert_advanced_signal()     → guarda señal avanzada completa
  - get_spike_distance_stats()   → estadísticas reales de distancia por nivel
  - get_tension_history()        → historial de tensión reciente
  - get_hourly_spike_frequency() → frecuencia de spikes por hora
  - get_recent_advanced_signals()→ señales avanzadas recientes
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiosqlite

from utils.logger import get_logger, log_error

log = get_logger("db_advanced")

_DDL_ADVANCED = """
-- Registro de cada spike y la distancia al anterior del mismo nivel
CREATE TABLE IF NOT EXISTS spike_distances (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    level       REAL    NOT NULL,   -- nivel del spike (5, 10, 20, etc.)
    val         REAL    NOT NULL,   -- valor exacto del spike
    distance    INTEGER NOT NULL,   -- rondas desde el spike anterior del mismo nivel
    round_id    INTEGER             -- referencia a rounds.id
);
CREATE INDEX IF NOT EXISTS idx_sd_level ON spike_distances(level);
CREATE INDEX IF NOT EXISTS idx_sd_ts    ON spike_distances(ts);

-- Snapshot del índice de tensión por cada N rondas
CREATE TABLE IF NOT EXISTS tension_history (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                      TEXT    NOT NULL,
    tension_value           REAL    NOT NULL,
    compression_contrib     REAL    NOT NULL DEFAULT 0,
    temporal_contrib        REAL    NOT NULL DEFAULT 0,
    volatility_contrib      REAL    NOT NULL DEFAULT 0,
    tension_label           TEXT    NOT NULL DEFAULT '',
    compression_score       REAL    NOT NULL DEFAULT 0,
    overdue_levels          TEXT    NOT NULL DEFAULT '[]',  -- JSON array
    context_tags            TEXT    NOT NULL DEFAULT '[]'   -- JSON array
);
CREATE INDEX IF NOT EXISTS idx_th_ts ON tension_history(ts);

-- Eventos de compresión extrema detectados
CREATE TABLE IF NOT EXISTS compression_events (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                      TEXT    NOT NULL,
    compression_score       REAL    NOT NULL,
    ultra_low_count         INTEGER NOT NULL,
    low_count               INTEGER NOT NULL,
    consecutive_ultra_low   INTEGER NOT NULL,
    label                   TEXT    NOT NULL,
    -- ¿Cuántas rondas después ocurrió el próximo spike? (se actualiza retroactivamente)
    rounds_to_next_spike5   INTEGER,
    rounds_to_next_spike10  INTEGER,
    rounds_to_next_spike20  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_ce_ts ON compression_events(ts);

-- Contextos estadísticos guardados justo antes de que ocurriera un spike confirmado
-- Para alimentar el análisis pre-spike con datos reales
CREATE TABLE IF NOT EXISTS pre_spike_contexts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    spike_ts            TEXT    NOT NULL,
    spike_val           REAL    NOT NULL,
    spike_level         REAL    NOT NULL,
    -- Estadísticas de las 15 rondas previas
    low_density_pre     REAL    NOT NULL DEFAULT 0,
    std_pre             REAL    NOT NULL DEFAULT 0,
    compression_pre     REAL    NOT NULL DEFAULT 0,
    tension_pre         REAL    NOT NULL DEFAULT 0,
    -- Distancia desde el spike previo del mismo nivel
    distance_from_prev  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_psc_level ON pre_spike_contexts(spike_level);

-- Señales del motor avanzado con metadata completa
CREATE TABLE IF NOT EXISTS advanced_signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT    NOT NULL,
    signal_type         TEXT    NOT NULL,   -- 'safe' / 'high_risk' / 'moon'
    target              REAL    NOT NULL,
    score               INTEGER NOT NULL,
    confidence          INTEGER NOT NULL,
    tension_index       REAL    NOT NULL DEFAULT 0,
    compression_score   REAL    NOT NULL DEFAULT 0,
    overdue_levels      TEXT    NOT NULL DEFAULT '[]',
    context_tags        TEXT    NOT NULL DEFAULT '[]',
    result              REAL,               -- valor real de la siguiente ronda
    won                 INTEGER,            -- 1 = ganó / 0 = perdió
    rounds_passed       INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_as_ts   ON advanced_signals(ts);
CREATE INDEX IF NOT EXISTS idx_as_type ON advanced_signals(signal_type);
"""


# ── Init ──────────────────────────────────────────────────────────────────────

async def init_advanced_db(conn: aiosqlite.Connection) -> None:
    """Aplica el DDL avanzado sobre la conexión existente."""
    try:
        await conn.executescript(_DDL_ADVANCED)
        await conn.commit()
        log.info("Tablas avanzadas inicializadas correctamente.")
    except Exception as e:
        log_error("init_advanced_db falló", e)
        raise


# ── Spike distances ───────────────────────────────────────────────────────────

async def insert_spike_distance(
    conn: aiosqlite.Connection,
    ts: datetime,
    level: float,
    val: float,
    distance: int,
    round_id: Optional[int] = None,
) -> None:
    """Registra un spike y la distancia al anterior del mismo nivel."""
    try:
        await conn.execute(
            """INSERT INTO spike_distances (ts, level, val, distance, round_id)
               VALUES (?, ?, ?, ?, ?)""",
            (ts.isoformat(), level, val, distance, round_id),
        )
        await conn.commit()
        log.debug("spike_distances: nivel=%.0fx dist=%d val=%.2f", level, distance, val)
    except Exception as e:
        log_error("insert_spike_distance falló", e)


@dataclass
class SpikeDistanceDBStats:
    level: float
    n: int
    avg: float
    median: float
    std: float
    p25: float
    p75: float
    min_dist: int
    max_dist: int


async def get_spike_distance_stats(
    conn: aiosqlite.Connection, level: float, limit: int = 500
) -> Optional[SpikeDistanceDBStats]:
    """Calcula estadísticas reales de distancia entre spikes de un nivel desde la DB."""
    try:
        async with conn.execute(
            "SELECT distance FROM spike_distances WHERE level = ? ORDER BY ts DESC LIMIT ?",
            (level, limit),
        ) as cur:
            rows = await cur.fetchall()

        if len(rows) < 5:
            return None

        import statistics as st

        dists = sorted([r[0] for r in rows])
        n     = len(dists)
        avg   = st.mean(dists)
        med   = st.median(dists)
        std   = st.stdev(dists) if n >= 2 else 0.0
        p25   = dists[max(0, int(n * 0.25))]
        p75   = dists[min(n - 1, int(n * 0.75))]

        return SpikeDistanceDBStats(
            level=level,
            n=n,
            avg=round(avg, 1),
            median=round(med, 1),
            std=round(std, 1),
            p25=float(p25),
            p75=float(p75),
            min_dist=int(min(dists)),
            max_dist=int(max(dists)),
        )
    except Exception as e:
        log_error(f"get_spike_distance_stats(level={level}) falló", e)
        return None


# ── Tension history ───────────────────────────────────────────────────────────

async def insert_tension_snapshot(
    conn: aiosqlite.Connection,
    ts: datetime,
    tension_value: float,
    compression_contrib: float,
    temporal_contrib: float,
    volatility_contrib: float,
    tension_label: str,
    compression_score: float,
    overdue_levels: list[float],
    context_tags: list[str],
) -> None:
    try:
        await conn.execute(
            """INSERT INTO tension_history
               (ts, tension_value, compression_contrib, temporal_contrib,
                volatility_contrib, tension_label, compression_score,
                overdue_levels, context_tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts.isoformat(), tension_value, compression_contrib,
                temporal_contrib, volatility_contrib, tension_label,
                compression_score,
                json.dumps(overdue_levels), json.dumps(context_tags),
            ),
        )
        await conn.commit()
    except Exception as e:
        log_error("insert_tension_snapshot falló", e)


async def get_tension_history(
    conn: aiosqlite.Connection, limit: int = 50
) -> list[dict]:
    """Retorna el historial reciente del índice de tensión."""
    try:
        async with conn.execute(
            """SELECT ts, tension_value, tension_label, compression_score, overdue_levels
               FROM tension_history ORDER BY ts DESC LIMIT ?""",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "ts": r[0],
                "tension": r[1],
                "label": r[2],
                "compression": r[3],
                "overdue_levels": json.loads(r[4]),
            }
            for r in rows
        ]
    except Exception as e:
        log_error("get_tension_history falló", e)
        return []


# ── Compression events ────────────────────────────────────────────────────────

async def insert_compression_event(
    conn: aiosqlite.Connection,
    ts: datetime,
    compression_score: float,
    ultra_low_count: int,
    low_count: int,
    consecutive_ultra_low: int,
    label: str,
) -> int:
    """Registra un evento de compresión. Retorna el ID del registro."""
    try:
        async with conn.execute(
            """INSERT INTO compression_events
               (ts, compression_score, ultra_low_count, low_count,
                consecutive_ultra_low, label)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ts.isoformat(), compression_score, ultra_low_count,
             low_count, consecutive_ultra_low, label),
        ) as cur:
            row_id = cur.lastrowid
        await conn.commit()
        log.info("[COMPRESIÓN GUARDADA] id=%d score=%.2f label=%s",
                 row_id, compression_score, label)
        return row_id
    except Exception as e:
        log_error("insert_compression_event falló", e)
        return -1


async def update_compression_event_spike(
    conn: aiosqlite.Connection,
    event_id: int,
    rounds_to_spike5: Optional[int] = None,
    rounds_to_spike10: Optional[int] = None,
    rounds_to_spike20: Optional[int] = None,
) -> None:
    """Actualiza retroactivamente el evento con cuántas rondas tardó en aparecer el spike."""
    try:
        await conn.execute(
            """UPDATE compression_events
               SET rounds_to_next_spike5=?, rounds_to_next_spike10=?, rounds_to_next_spike20=?
               WHERE id=?""",
            (rounds_to_spike5, rounds_to_spike10, rounds_to_spike20, event_id),
        )
        await conn.commit()
    except Exception as e:
        log_error("update_compression_event_spike falló", e)


# ── Pre-spike contexts ────────────────────────────────────────────────────────

async def save_pre_spike_context(
    conn: aiosqlite.Connection,
    spike_ts: datetime,
    spike_val: float,
    spike_level: float,
    low_density_pre: float,
    std_pre: float,
    compression_pre: float,
    tension_pre: float,
    distance_from_prev: int,
) -> None:
    """Guarda el contexto estadístico que precedió a un spike confirmado."""
    try:
        await conn.execute(
            """INSERT INTO pre_spike_contexts
               (spike_ts, spike_val, spike_level, low_density_pre, std_pre,
                compression_pre, tension_pre, distance_from_prev)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (spike_ts.isoformat(), spike_val, spike_level,
             low_density_pre, std_pre, compression_pre, tension_pre,
             distance_from_prev),
        )
        await conn.commit()
        log.debug("pre_spike_context guardado: %.2fx (nivel=%.0fx)", spike_val, spike_level)
    except Exception as e:
        log_error("save_pre_spike_context falló", e)


@dataclass
class PreSpikeDBStats:
    """Estadísticas de contextos previos a spikes, calculadas desde la DB."""
    level: float
    n: int
    avg_low_density: float
    avg_std: float
    avg_compression: float
    avg_tension: float


async def get_pre_spike_stats(
    conn: aiosqlite.Connection, level: float, limit: int = 200
) -> Optional[PreSpikeDBStats]:
    """Lee estadísticas reales de contextos previos a spikes desde la DB."""
    try:
        async with conn.execute(
            """SELECT low_density_pre, std_pre, compression_pre, tension_pre
               FROM pre_spike_contexts
               WHERE spike_level = ? ORDER BY spike_ts DESC LIMIT ?""",
            (level, limit),
        ) as cur:
            rows = await cur.fetchall()

        if len(rows) < 3:
            return None

        import statistics as st

        return PreSpikeDBStats(
            level=level,
            n=len(rows),
            avg_low_density=round(st.mean(r[0] for r in rows), 3),
            avg_std=round(st.mean(r[1] for r in rows), 3),
            avg_compression=round(st.mean(r[2] for r in rows), 3),
            avg_tension=round(st.mean(r[3] for r in rows), 3),
        )
    except Exception as e:
        log_error(f"get_pre_spike_stats(level={level}) falló", e)
        return None


# ── Advanced signals ──────────────────────────────────────────────────────────

async def insert_advanced_signal(
    conn: aiosqlite.Connection,
    ts: datetime,
    signal_type: str,
    target: float,
    score: int,
    confidence: int,
    tension_index: float,
    compression_score: float,
    overdue_levels: list[float],
    context_tags: list[str],
) -> int:
    try:
        async with conn.execute(
            """INSERT INTO advanced_signals
               (ts, signal_type, target, score, confidence, tension_index,
                compression_score, overdue_levels, context_tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts.isoformat(), signal_type, target, score, confidence,
                tension_index, compression_score,
                json.dumps(overdue_levels), json.dumps(context_tags),
            ),
        ) as cur:
            row_id = cur.lastrowid
        await conn.commit()
        return row_id
    except Exception as e:
        log_error("insert_advanced_signal falló", e)
        return -1


async def resolve_advanced_signal(
    conn: aiosqlite.Connection,
    signal_id: int,
    result_val: float,
    won: bool,
    rounds_passed: int,
) -> None:
    try:
        await conn.execute(
            """UPDATE advanced_signals
               SET result=?, won=?, rounds_passed=? WHERE id=?""",
            (result_val, 1 if won else 0, rounds_passed, signal_id),
        )
        await conn.commit()
    except Exception as e:
        log_error("resolve_advanced_signal falló", e)


async def get_advanced_signal_stats(conn: aiosqlite.Connection) -> dict:
    """Estadísticas de rendimiento de señales avanzadas."""
    try:
        stats: dict = {}
        for sig_type in ("safe", "high_risk", "moon"):
            async with conn.execute(
                """SELECT COUNT(*), SUM(won), AVG(tension_index), AVG(compression_score)
                   FROM advanced_signals
                   WHERE signal_type=? AND result IS NOT NULL""",
                (sig_type,),
            ) as cur:
                row = await cur.fetchone()
            total = row[0] or 0
            wins  = row[1] or 0
            stats[sig_type] = {
                "total": total,
                "wins": wins,
                "winrate": round(wins / total * 100, 1) if total > 0 else 0.0,
                "avg_tension": round(row[2] or 0, 1),
                "avg_compression": round(row[3] or 0, 3),
            }
        return stats
    except Exception as e:
        log_error("get_advanced_signal_stats falló", e)
        return {}


# ── Hourly spike frequency ────────────────────────────────────────────────────

async def get_hourly_spike_frequency(
    conn: aiosqlite.Connection, level: float, hour: Optional[str] = None
) -> Optional[dict]:
    """
    Frecuencia de spikes de un nivel por hora del día.
    Si se pasa `hour` (formato '00'–'23'), filtra solo esa hora.
    """
    try:
        if hour:
            async with conn.execute(
                """SELECT COUNT(*) FROM spike_distances
                   WHERE level=? AND strftime('%H', ts)=?""",
                (level, hour),
            ) as cur:
                count_hour = (await cur.fetchone())[0]

            async with conn.execute(
                "SELECT COUNT(*) FROM spike_distances WHERE level=?", (level,)
            ) as cur:
                count_total = (await cur.fetchone())[0]

            return {
                "level": level,
                "hour": hour,
                "count_this_hour": count_hour,
                "count_total": count_total,
                "pct_this_hour": round(count_hour / count_total * 100, 1) if count_total else 0,
            }
        else:
            async with conn.execute(
                """SELECT strftime('%H', ts) as hr, COUNT(*) as cnt
                   FROM spike_distances WHERE level=?
                   GROUP BY hr ORDER BY hr""",
                (level,),
            ) as cur:
                rows = await cur.fetchall()
            return {r[0]: r[1] for r in rows}
    except Exception as e:
        log_error(f"get_hourly_spike_frequency(level={level}) falló", e)
        return None