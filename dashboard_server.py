"""
dashboard_server.py — Servidor FastAPI para el dashboard JetX v3.
════════════════════════════════════════════════════════════════════
Expone la DB SQLite de jetx_v3 a través de:
  - WebSocket /ws        → stream de rondas en tiempo real (polling DB)
  - GET /api/history     → últimas N rondas
  - GET /api/stats       → estadísticas completas
  - GET /api/signals     → señales del motor dual
  - GET /api/entropy     → snapshots de entropía
  - WebSocket /ws/live   → broadcast a todos los clientes conectados

Corre en paralelo con main_v3.py — solo lee la DB, no escribe.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import statistics
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = "jetx_data.db"
POLL_INTERVAL = 1.5          # segundos entre lecturas de DB
MAX_HISTORY   = 200

# Archivo compartido entre procesos para el multiplicador en vuelo.
# main_v3.py lo escribe cada 250ms; dashboard_server lo lee aqui.
LIVE_TICK_FILE = "live_tick.json"

def get_live_mult() -> "float | None":
    """Lee el multiplicador en vuelo desde el archivo compartido con main_v3."""
    try:
        with open(LIVE_TICK_FILE, "r") as f:
            data = json.load(f)
            return data.get("val")   # None si entre rondas
    except Exception:
        return None


# ── Gestor de clientes WebSocket ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data, ensure_ascii=False)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()
last_round_id: int = 0


# ── Helpers DB (síncrono — lectura rápida) ────────────────────────────────────

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def get_rounds(limit: int = MAX_HISTORY) -> list[dict]:
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT id, val, ts FROM rounds ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"id": r["id"], "val": r["val"], "ts": r["ts"]} for r in reversed(rows)]
    except Exception:
        return []


def get_latest_round_id() -> int:
    try:
        with _db() as conn:
            row = conn.execute("SELECT MAX(id) as mid FROM rounds").fetchone()
        return row["mid"] or 0
    except Exception:
        return 0


def get_dual_signals(limit: int = 10) -> list[dict]:
    try:
        with _db() as conn:
            # Intentar tabla dual_engine_signals primero
            try:
                rows = conn.execute("""
                    SELECT id, ts, mode, signal_type, target, safe_score,
                           hunter_score, moon_score, entropy_score,
                           tension_score, compression_score, confidence,
                           result, won, rounds_passed
                    FROM dual_engine_signals
                    ORDER BY id DESC LIMIT ?
                """, (limit,)).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                # Fallback a signals básica
                rows = conn.execute("""
                    SELECT id, ts, val_target as target, signal_type,
                           score, confidence, result, won, rounds_passed
                    FROM signals
                    ORDER BY id DESC LIMIT ?
                """, (limit,)).fetchall()
                return [dict(r) for r in rows]
    except Exception:
        return []


def get_entropy_snapshots(limit: int = 20) -> list[dict]:
    try:
        with _db() as conn:
            rows = conn.execute("""
                SELECT ts, entropy_score, label, is_dead_market
                FROM entropy_snapshots
                ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def compute_stats(rounds: list[dict]) -> dict:
    """Calcula estadísticas completas a partir del historial."""
    if not rounds:
        return {}

    vals = [r["val"] for r in rounds]
    n    = len(vals)

    # Básicas
    mean_val  = statistics.mean(vals)
    median_val = statistics.median(vals)
    std_val   = statistics.stdev(vals) if n > 1 else 0.0
    min_val   = min(vals)
    max_val   = max(vals)

    # Densidades por rango
    def pct(lo, hi):
        return round(sum(1 for v in vals if lo <= v < hi) / n * 100, 1)

    ranges = {
        "1x_2x":    pct(1.0,  2.0),
        "2x_5x":    pct(2.0,  5.0),
        "5x_10x":   pct(5.0,  10.0),
        "10x_25x":  pct(10.0, 25.0),
        "25x_100x": pct(25.0, 100.0),
        "100x_plus": round(sum(1 for v in vals if v >= 100.0) / n * 100, 1),
    }

    crash_density = round(sum(1 for v in vals if v < 2.0) / n, 4)

    # Rachas
    hot_streak  = 0
    cold_streak = 0
    _hs = 0
    _cs = 0
    for v in reversed(vals):
        if v >= 2.0:
            _hs += 1
            _cs = 0
        else:
            _cs += 1
            _hs = 0
        hot_streak  = max(hot_streak,  _hs)
        cold_streak = max(cold_streak, _cs)

    # Rondas desde última aparición de multiplicadores clave
    def rounds_since(threshold: float) -> int:
        for i, v in enumerate(reversed(vals)):
            if v >= threshold:
                return i
        return n

    rs_10  = rounds_since(10.0)
    rs_25  = rounds_since(25.0)
    rs_50  = rounds_since(50.0)
    rs_100 = rounds_since(100.0)

    # Hot/Cold top 5 (redondeados a 0.5x buckets)
    freq: dict[float, int] = {}
    for v in vals:
        bucket = round(v * 2) / 2
        freq[bucket] = freq.get(bucket, 0) + 1
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    hot_numbers  = [{"val": k, "count": v} for k, v in sorted_freq[:5]]
    cold_numbers = [{"val": k, "count": v} for k, v in sorted_freq[-5:]]

    # Tendencia (últimas 20 vs previas 20)
    recent_20 = vals[:20] if n >= 20 else vals
    prev_20   = vals[20:40] if n >= 40 else vals[min(n//2, 20):]
    trend_dir = "up" if (statistics.mean(recent_20) > statistics.mean(prev_20) if prev_20 else False) else "down"

    # Volatilidad (coef. variación)
    volatility = round(std_val / mean_val, 4) if mean_val > 0 else 0.0

    # Probabilidad crash temprano (val < 2)
    crash_prob = round(crash_density * 100, 1)

    # Detección de comportamiento anormal (z-score del último valor)
    last_val = vals[-1] if vals else 1.0
    anomaly = abs(last_val - mean_val) > 2.5 * std_val if std_val > 0 else False

    # Índice de estabilidad (inverso de volatilidad normalizado)
    stability = round(max(0, min(100, 100 - volatility * 50)), 1)

    # Nivel de riesgo
    if crash_density > 0.50:
        risk_level = "EXTREMO"
    elif crash_density > 0.35:
        risk_level = "ALTO"
    elif crash_density > 0.20:
        risk_level = "MEDIO"
    else:
        risk_level = "BAJO"

    # Momentum (slope lineal de últimos 30)
    recent_30 = vals[:30] if n >= 30 else vals
    if len(recent_30) >= 4:
        x_mean = (len(recent_30) - 1) / 2
        slope_num = sum((i - x_mean) * (v - mean_val) for i, v in enumerate(recent_30))
        slope_den = sum((i - x_mean)**2 for i in range(len(recent_30)))
        momentum  = round(slope_num / slope_den, 4) if slope_den != 0 else 0.0
    else:
        momentum = 0.0

    # Heatmap de frecuencias (bins de 0.5x hasta 20x, luego grandes)
    bins = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0,
            12.0, 15.0, 20.0, 30.0, 50.0, 100.0]
    heatmap = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        cnt = sum(1 for v in vals if lo <= v < hi)
        heatmap.append({"lo": lo, "hi": hi, "count": cnt, "pct": round(cnt/n*100, 1)})
    cnt_100 = sum(1 for v in vals if v >= 100.0)
    heatmap.append({"lo": 100.0, "hi": 9999.0, "count": cnt_100, "pct": round(cnt_100/n*100, 1)})

    return {
        "n": n,
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "crash_density": crash_density,
        "crash_prob": crash_prob,
        "ranges": ranges,
        "hot_streak": hot_streak,
        "cold_streak": cold_streak,
        "rounds_since_10x": rs_10,
        "rounds_since_25x": rs_25,
        "rounds_since_50x": rs_50,
        "rounds_since_100x": rs_100,
        "hot_numbers": hot_numbers,
        "cold_numbers": cold_numbers,
        "trend": trend_dir,
        "volatility": volatility,
        "stability": stability,
        "risk_level": risk_level,
        "momentum": momentum,
        "anomaly_detected": anomaly,
        "last_val": round(last_val, 2),
        "heatmap": heatmap,
    }


# ── Background poller ─────────────────────────────────────────────────────────

async def poll_db():
    """Polling loop: detecta nuevas rondas y hace broadcast por WS."""
    global last_round_id
    last_round_id = get_latest_round_id()

    while True:
        await asyncio.sleep(POLL_INTERVAL)
        try:
            current_id = get_latest_round_id()
            if current_id > last_round_id:
                # Nuevas rondas detectadas
                rounds = get_rounds(MAX_HISTORY)
                stats  = compute_stats(rounds)
                signals = get_dual_signals(10)
                entropy = get_entropy_snapshots(10)

                payload = {
                    "type":    "update",
                    "rounds":  rounds[-50:],   # solo las últimas 50 para el stream
                    "stats":   stats,
                    "signals": signals,
                    "entropy": entropy,
                    "ts":      datetime.now().isoformat(),
                }
                await manager.broadcast(payload)
                last_round_id = current_id
        except Exception as e:
            print(f"[poll_db error] {e}")


async def poll_live_ticks():
    """
    Polling loop de 250ms: transmite el multiplicador en vuelo en tiempo real.
    El frontend recibe {'type': 'tick', 'val': 2.34} y mueve el avión en vivo.
    Cuando val=null, el frontend sabe que BetPlay está en fase de espera.
    """
    last_sent: float | None = -1.0   # sentinel diferente de None y cualquier float

    while True:
        await asyncio.sleep(0.25)
        try:
            val = get_live_mult()
            # Solo broadcast si cambia el valor (evitar spam)
            if val != last_sent:
                await manager.broadcast({
                    "type": "tick",
                    "val":  val,
                    "ts":   datetime.now().isoformat(),
                })
                last_sent = val
        except Exception as e:
            print(f"[poll_live_ticks error] {e}")


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    task1 = asyncio.create_task(poll_db())
    task2 = asyncio.create_task(poll_live_ticks())
    yield
    task1.cancel()
    task2.cancel()


app = FastAPI(title="JetX Dashboard API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Sirve el dashboard directamente en localhost:8765"""
    # Buscar index.html relativo al script
    base = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(base, "dashboard", "index.html"),
        os.path.join(base, "index.html"),
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>No se encontró dashboard/index.html</h1>"
                        "<p>Asegúrate de que existe jetx_v3/dashboard/index.html</p>", status_code=404)


@app.get("/api/history")
async def api_history(limit: int = 200):
    rounds = get_rounds(limit)
    return {"rounds": rounds, "count": len(rounds)}


@app.get("/api/stats")
async def api_stats():
    rounds = get_rounds(MAX_HISTORY)
    return compute_stats(rounds)


@app.get("/api/signals")
async def api_signals(limit: int = 20):
    return {"signals": get_dual_signals(limit)}


@app.get("/api/full")
async def api_full():
    """Payload completo para inicializar el dashboard."""
    rounds  = get_rounds(MAX_HISTORY)
    stats   = compute_stats(rounds)
    signals = get_dual_signals(10)
    entropy = get_entropy_snapshots(10)
    return {
        "rounds":  rounds,
        "stats":   stats,
        "signals": signals,
        "entropy": entropy,
        "ts":      datetime.now().isoformat(),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Enviar estado inicial completo
        rounds  = get_rounds(MAX_HISTORY)
        stats   = compute_stats(rounds)
        signals = get_dual_signals(10)
        entropy = get_entropy_snapshots(10)
        await ws.send_text(json.dumps({
            "type":    "init",
            "rounds":  rounds,
            "stats":   stats,
            "signals": signals,
            "entropy": entropy,
            "ts":      datetime.now().isoformat(),
        }))

        # Mantener conexión viva
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "ping"}))

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
