"""
database/db.py
Capa de acceso a datos con aiosqlite.
resolve_pending_signals ahora retorna lista de señales resueltas
para que main.py las notifique por Telegram.
"""

from __future__ import annotations

import aiosqlite
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from core.config import DB_PATH
from utils.logger import get_logger, log_error

log = get_logger("database")

_DDL = """
CREATE TABLE IF NOT EXISTS rounds (
    id  INTEGER PRIMARY KEY AUTOINCREMENT,
    val REAL    NOT NULL,
    ts  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rounds_ts ON rounds(ts);

CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    val_target    REAL    NOT NULL,
    signal_type   TEXT    NOT NULL DEFAULT 'legacy',
    score         INTEGER NOT NULL DEFAULT 0,
    confidence    INTEGER NOT NULL DEFAULT 0,
    result        REAL,
    won           INTEGER,
    rounds_passed INTEGER DEFAULT 0
);
"""


@dataclass
class ResolvedSignal:
    """Señal que acaba de resolverse (ganada o perdida)."""
    signal_id: int
    target: float
    result: float
    won: bool
    rounds_passed: int


async def init_db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    await conn.executescript(_DDL)
    await conn.commit()
    log.info("Base de datos inicializada: %s", DB_PATH)
    return conn


async def insert_round(conn: aiosqlite.Connection, val: float, ts: datetime) -> None:
    try:
        await conn.execute("INSERT INTO rounds (val, ts) VALUES (?, ?)",
                           (val, ts.isoformat()))
        await conn.commit()
    except Exception as e:
        log_error("insert_round falló", e)
        raise


async def insert_signal(conn: aiosqlite.Connection, ts: datetime, val_target: float,
                        signal_type: str, score: int, confidence: int) -> None:
    try:
        await conn.execute(
            "INSERT INTO signals (ts, val_target, signal_type, score, confidence) VALUES (?,?,?,?,?)",
            (ts.isoformat(), val_target, signal_type, score, confidence))
        await conn.commit()
    except Exception as e:
        log_error("insert_signal falló", e)
        raise


async def resolve_pending_signals(
    conn: aiosqlite.Connection, latest_val: float
) -> list[ResolvedSignal]:
    """
    Evalúa señales pendientes. ESTRICTAMENTE 1 RONDA (Disparo único).
    Si la siguiente ronda falla, se cuenta como perdida inmediatamente.
    """
    resolved: list[ResolvedSignal] = []
    try:
        async with conn.execute(
            "SELECT id, val_target, rounds_passed FROM signals WHERE result IS NULL"
        ) as cursor:
            pendientes = await cursor.fetchall()

        for row in pendientes:
            sig_id, target, rp = row["id"], row["val_target"], row["rounds_passed"]
            rp += 1

            if latest_val >= target:
                # La ganó en el disparo exacto (Ronda 1)
                await conn.execute(
                    "UPDATE signals SET result=?, won=1, rounds_passed=? WHERE id=?",
                    (latest_val, rp, sig_id))
                resolved.append(ResolvedSignal(sig_id, target, latest_val, True, rp))
                log.info("✅ Señal #%d ganada (Precisa): %.2fx >= %.2fx", sig_id, latest_val, target)

            else:
                # La perdió en el disparo exacto. Cero tolerancia, se marca como perdida.
                await conn.execute(
                    "UPDATE signals SET result=?, won=0, rounds_passed=? WHERE id=?",
                    (latest_val, rp, sig_id))
                resolved.append(ResolvedSignal(sig_id, target, latest_val, False, rp))
                log.info("❌ Señal #%d perdida (Fallo): %.2fx < %.2fx", sig_id, latest_val, target)

        await conn.commit()
    except Exception as e:
        log_error("resolve_pending_signals falló", e)

    return resolved


async def get_recent_rounds(conn: aiosqlite.Connection, limit: int = 200) -> list[float]:
    try:
        async with conn.execute(
            "SELECT val FROM rounds ORDER BY ts DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
        return [r["val"] for r in rows]
    except Exception as e:
        log_error("get_recent_rounds falló", e)
        return []


@dataclass
class Metrics:
    total: int
    wins: int
    winrate: float
    profit_factor: float
    max_drawdown: int
    avg_multiplier: float


async def get_metrics(conn: aiosqlite.Connection) -> Metrics:
    try:
        async with conn.execute(
            "SELECT val_target, result, won FROM signals WHERE result IS NOT NULL ORDER BY ts ASC"
        ) as cursor:
            rows = await cursor.fetchall()

        if not rows:
            return Metrics(0, 0, 0.0, 0.0, 0, 0.0)

        total = len(rows)
        wins  = sum(1 for r in rows if r["won"] == 1)
        winrate = (wins / total) * 100.0

        gross_profit = sum((r["val_target"] - 1.0) for r in rows if r["won"] == 1)
        gross_loss   = sum(1.0 for r in rows if r["won"] == 0)
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else round(gross_profit, 2)

        max_dd, cur_dd = 0, 0
        for r in rows:
            if r["won"] == 0:
                cur_dd += 1; max_dd = max(max_dd, cur_dd)
            else:
                cur_dd = 0

        winning = [r["result"] for r in rows if r["won"] == 1 and r["result"]]
        avg_mult = round(sum(winning) / len(winning), 2) if winning else 0.0

        return Metrics(total, wins, round(winrate, 1), profit_factor, max_dd, avg_mult)
    except Exception as e:
        log_error("get_metrics falló", e)
        return Metrics(0, 0, 0.0, 0.0, 0, 0.0)


async def get_hourly_distribution(conn: aiosqlite.Connection, hour: str) -> Optional[dict]:
    try:
        async with conn.execute(
            "SELECT val FROM rounds WHERE strftime('%H', ts) = ?", (hour,)
        ) as cursor:
            rows = await cursor.fetchall()

        vals = [r["val"] for r in rows]
        if len(vals) < 30:
            return None

        n = len(vals)
        return {
            "n": n,
            "crashes_low_pct": round(sum(1 for v in vals if v < 1.50) / n * 100, 1),
            "spikes_high_pct": round(sum(1 for v in vals if v >= 5.0) / n * 100, 1),
            "mid_range_pct":   round(sum(1 for v in vals if 1.50 <= v < 5.0) / n * 100, 1),
        }
    except Exception as e:
        log_error("get_hourly_distribution falló", e)
        return None

























