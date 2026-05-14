"""
main_v3.py — Orquestador principal del bot JetX v3.
═══════════════════════════════════════════════════════════════════════════════
Sistema de doble motor estadístico:
  SAFE MODE   → señales frecuentes, conservadoras
  HUNTER MODE → detección de contextos para cuotas altas
  MOON MODE   → detección de contextos para cuotas extremas (10x+)

Integra:
  - Entropía / Dead Market Mode
  - Secuencias binarias
  - Fibonacci experimental
  - Machine Learning descriptivo (clustering + anomalías)
  - Scores completamente separados por modo
  - DB avanzada v2 con todas las métricas
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime
from typing import Optional

import aiosqlite

from core.config import (
    COOLDOWN_ROUNDS, COOLDOWN_MINUTES, MIN_HISTORY, DB_PATH,
)
from analyzer.stats import SpikeBlocker, analyze
from analyzer.advanced_stats import MultiLevelSpikeBlocker, analyze_advanced, SPIKE_LEVELS
from analyzer.spike_tracker import SpikeTracker
from analyzer.dual_engine import DualEngineResult, run_dual_engine
from analyzer.ml_clustering import ContextMemory
from analyzer.ai_analyzer import get_ai_vote

from database.db import (
    init_db, insert_round, get_recent_rounds,
    get_metrics, get_hourly_distribution,
)
from database.db_advanced import (
    init_advanced_db,
    insert_tension_snapshot,
    insert_compression_event,
    get_advanced_signal_stats,
)
from database.db_v2 import (
    init_db_v2,
    insert_dual_signal,
    resolve_dual_signal,
    get_pending_dual_signals,
    insert_entropy_snapshot,
    insert_binary_event,
    insert_clustering_snapshot,
    insert_fibonacci_event,
    update_mode_effectiveness,
    get_mode_effectiveness,
    get_dual_engine_stats,
    append_spike_distance,
    get_spike_distances_for_fibonacci,
)
from browser.scraper import JetXScraper
from tg_bot.notifier_v2 import (
    send_dual_signal,
    send_result_v2,
    send_status_v2,
)
from utils.logger import get_logger, log_error

# Escritura de ticks en vuelo al archivo compartido con dashboard_server
import json as _json
_LIVE_TICK_FILE = "live_tick.json"

def _set_live_mult(val):
    """Escribe el multiplicador en vuelo al archivo compartido con dashboard_server."""
    try:
        with open(_LIVE_TICK_FILE, "w") as _f:
            _json.dump({"val": val}, _f)
    except Exception:
        pass

log = get_logger("main_v3")

# ── Intervalos ────────────────────────────────────────────────────────────────
TENSION_SNAPSHOT_INTERVAL  = 10   # cada N rondas
ENTROPY_SNAPSHOT_INTERVAL  = 15
CLUSTERING_SNAPSHOT_INTERVAL = 20
STATUS_REPORT_INTERVAL     = 500  # cada N rondas
COMPRESSION_THRESHOLD      = 0.55
BINARY_RARENESS_THRESHOLD  = 0.45
FIB_SAVE_THRESHOLD         = 0.35


class CooldownTracker:
    def __init__(self):
        self._last_ts: Optional[datetime] = None
        self._rounds_since: int = 0

    def tick(self):
        self._rounds_since += 1

    def can_emit(self, mode: str = "safe") -> bool:
        min_rounds = COOLDOWN_ROUNDS if mode == "safe" else max(COOLDOWN_ROUNDS, 4)
        if self._last_ts is None:
            return self._rounds_since >= min_rounds

        elapsed = (datetime.now() - self._last_ts).total_seconds() / 60.0
        min_minutes = COOLDOWN_MINUTES if mode == "safe" else max(COOLDOWN_MINUTES, 2.0)
        return self._rounds_since >= min_rounds and elapsed >= min_minutes

    def register_emission(self):
        self._last_ts = datetime.now()
        self._rounds_since = 0


async def _resolve_pending(conn: aiosqlite.Connection, latest_val: float) -> None:
    """
    Resuelve señales pendientes con VENTANA DE 3 RONDAS.
    Una señal gana si el target se alcanza en cualquiera de las 3 rondas.
    Después de 3 rondas sin alcanzar el target → perdida.
    (Igual que JetX Pro — esto sube el win rate considerablemente)
    """
    pending = await get_pending_dual_signals(conn)
    for sig in pending:
        sig_id  = sig["id"]
        target  = sig["target"]
        mode    = sig["mode"]
        rp      = sig["rounds_passed"] + 1

        won = latest_val >= target

        if won:
            # Ganada en esta ronda
            await resolve_dual_signal(conn, sig_id, latest_val, True, rp)
            await update_mode_effectiveness(conn, mode, target, True, score=0)
            await send_result_v2(sig_id, mode, target, latest_val, True, rp)
            log.info("✅ Dual señal #%d (%s %.1fx) → %.2fx en %d ronda(s)", sig_id, mode, target, latest_val, rp)

        elif rp >= 3:
            # Ventana de 3 rondas agotada sin ganar → perdida
            await resolve_dual_signal(conn, sig_id, latest_val, False, rp)
            await update_mode_effectiveness(conn, mode, target, False, score=0)
            await send_result_v2(sig_id, mode, target, latest_val, False, rp)
            log.info("❌ Dual señal #%d (%s %.1fx) → %.2fx — perdida tras %d rondas", sig_id, mode, target, latest_val, rp)

        else:
            # Sigue esperando (rp < 3 y no ganó aún)
            # Solo actualizamos rounds_passed, sin cerrar la señal
            import aiosqlite as _aio
            await conn.execute(
                "UPDATE dual_signals SET rounds_passed = ? WHERE id = ?",
                (rp, sig_id)
            )
            await conn.commit()
            log.debug("⏳ Señal #%d (%s %.1fx) esperando ronda %d/3 — actual: %.2fx", sig_id, mode, target, rp, latest_val))


async def run() -> None:
    log.info("═══════════════════════════════════════════════════════")
    log.info("  JetX Bot v3 — Motor Dual Estadístico")
    log.info("  SAFE MODE + HUNTER MODE + MOON MODE")
    log.info("  Entropía | Fibonacci | ML Clustering | Dead Market")
    log.info("═══════════════════════════════════════════════════════")

    # ── Inicializar DB ─────────────────────────────────────────────────────────
    conn: aiosqlite.Connection = await init_db()
    await init_advanced_db(conn)
    await init_db_v2(conn)

    # ── Componentes del sistema ────────────────────────────────────────────────
    spike_blocker   = SpikeBlocker()
    ml_blocker      = MultiLevelSpikeBlocker()
    spike_tracker   = SpikeTracker()
    context_memory  = ContextMemory(max_size=500)
    cooldown        = CooldownTracker()

    round_counter   = 0
    last_dead_market_notified = False

    async with JetXScraper() as scraper:
        # ── Task paralela: multiplicador en vuelo → dashboard ──────────────────
        async def _live_tick_task():
            async for tick_val in scraper.live_tick_stream():
                _set_live_mult(tick_val)

        asyncio.create_task(_live_tick_task())
        # ── Loop principal: rondas completadas ─────────────────────────────────
        async for val, ts in scraper.round_stream():
            round_counter += 1
            cooldown.tick()

            try:
                # ── Persistir ronda ───────────────────────────────────────────
                await insert_round(conn, val, ts)
                await _resolve_pending(conn, val)

                # ── Actualizar bloqueadores y tracker ─────────────────────────
                spike_blocker.register_round(val)
                ml_blocker.register_round(val)
                await spike_tracker.register_round(val, ts, conn)

                # Guardar distancia para Fibonacci
                for level in SPIKE_LEVELS:
                    dist = spike_tracker.get_rounds_since(level)
                    if val >= level:  # acaba de ser un spike
                        await append_spike_distance(conn, ts, level, dist)

                # ── Cargar historial ──────────────────────────────────────────
                history = await get_recent_rounds(conn, limit=200)
                if len(history) < MIN_HISTORY:
                    log.debug("Calibrando… %d/%d rondas", len(history), MIN_HISTORY)
                    continue

                # ── Análisis base ─────────────────────────────────────────────
                hourly_dist = await get_hourly_distribution(conn, ts.strftime("%H"))
                base_result = analyze(history, spike_blocker, hourly_dist)

                # ── Análisis avanzado ─────────────────────────────────────────
                adv_result = analyze_advanced(
                    history=history,
                    blocker=ml_blocker,
                    base_prob_200=base_result.prob_200,
                    base_prob_300=base_result.prob_300,
                    base_prob_500=base_result.prob_500,
                    base_prob_800=base_result.prob_800,
                    base_prob_1500=base_result.prob_1500,
                    crash_density=base_result.crash_density,
                    base_score=base_result.score,
                )

                # Actualizar tensión en tracker
                if adv_result.tension:
                    spike_tracker.set_last_tension(adv_result.tension.value)

                # ── Cargar distancias de spikes para Fibonacci ─────────────────
                spike_distances = await get_spike_distances_for_fibonacci(conn)

                # ── Motor dual ────────────────────────────────────────────────
                dual_result = run_dual_engine(
                    history=history,
                    adv_result=adv_result,
                    context_memory=context_memory,
                    spike_distances=spike_distances,
                    prob_190=base_result.prob_190,
                    prob_200=base_result.prob_200,
                    prob_250=base_result.prob_250,
                    prob_300=base_result.prob_300,
                    prob_500=base_result.prob_500,
                    prob_800=base_result.prob_800,
                    prob_1500=base_result.prob_1500,
                    crash_density=base_result.crash_density,
                    base_score=base_result.score,
                )

                # ── Log de ronda ──────────────────────────────────────────────
                log.info(
                    "[%s] %.2fx | mode=%s | safe=%d | hunter=%d | moon=%d | "
                    "entropy=%.3f | comp=%.3f | tension=%.0f",
                    ts.strftime("%H:%M:%S"), val,
                    dual_result.mode,
                    dual_result.safe_score,
                    dual_result.hunter_score,
                    dual_result.moon_score,
                    dual_result.entropy_score_val,
                    dual_result.compression_score,
                    dual_result.tension_score,
                )

                # Log contextos especiales
                for ctx in dual_result.context_logs:
                    log.info(ctx)

                # ── Persistir snapshots periódicos ────────────────────────────

                # Entropía
                if round_counter % ENTROPY_SNAPSHOT_INTERVAL == 0:
                    e = dual_result.entropy
                    await insert_entropy_snapshot(
                        conn, ts=ts,
                        entropy_score=e.entropy_score,
                        label=e.label,
                        compression_ratio=e.compression_ratio,
                        run_length_score=e.run_length_score,
                        is_dead_market=e.is_dead_market,
                    )

                # Tensión (del motor avanzado)
                if round_counter % TENSION_SNAPSHOT_INTERVAL == 0 and adv_result.tension:
                    t = adv_result.tension
                    tmap = adv_result.temporal_map
                    overdue = tmap.overdue_levels() if tmap else []
                    await insert_tension_snapshot(
                        conn, ts=ts,
                        tension_value=t.value,
                        compression_contrib=t.compression_contribution,
                        temporal_contrib=t.temporal_contribution,
                        volatility_contrib=t.volatility_contribution,
                        tension_label=t.label,
                        compression_score=adv_result.compression.compression_score if adv_result.compression else 0.0,
                        overdue_levels=overdue,
                        context_tags=t.context_tags,
                    )

                # Clustering
                if round_counter % CLUSTERING_SNAPSHOT_INTERVAL == 0:
                    cl = dual_result.clustering
                    await insert_clustering_snapshot(
                        conn, ts=ts,
                        context_label=cl.context_label,
                        context_score=cl.context_score,
                        anomaly_score=cl.anomaly_score,
                        is_anomaly=cl.is_anomaly,
                        cluster_id=cl.cluster_id,
                        similar_contexts=cl.similar_contexts,
                        buffer_size=context_memory.buffer_size,
                    )

                # Eventos binarios relevantes
                if dual_result.binary.rareness_score >= BINARY_RARENESS_THRESHOLD:
                    b = dual_result.binary
                    await insert_binary_event(
                        conn, ts=ts,
                        rareness_score=b.rareness_score,
                        longest_run_0=b.longest_run_0,
                        longest_run_1=b.longest_run_1,
                        density_0=b.density_0,
                        density_1=b.density_1,
                        micro_cycles=b.micro_cycles,
                        sequence_30=b.sequence[:30],
                    )

                # Eventos Fibonacci relevantes
                fib = dual_result.fibonacci
                if fib and fib.coincidence_ratio >= FIB_SAVE_THRESHOLD:
                    await insert_fibonacci_event(
                        conn, ts=ts,
                        coincidence_ratio=fib.coincidence_ratio,
                        golden_pairs=len(fib.golden_ratio_pairs),
                        expansion=fib.expansion_detected,
                        compression=fib.compression_detected,
                        note=fib.note,
                    )

                # Eventos de compresión del motor avanzado
                if (adv_result.compression and
                        adv_result.compression.compression_score >= COMPRESSION_THRESHOLD):
                    comp = adv_result.compression
                    event_id = await insert_compression_event(
                        conn, ts=ts,
                        compression_score=comp.compression_score,
                        ultra_low_count=comp.ultra_low_count,
                        low_count=comp.low_count,
                        consecutive_ultra_low=comp.consecutive_ultra_low,
                        label=comp.label,
                    )
                    spike_tracker.register_compression_event(event_id)

                # ── Dead Market: notificar una sola vez ───────────────────────
                if dual_result.mode == "dead" and not last_dead_market_notified:
                    await send_dual_signal(dual_result, await get_metrics(conn), ts,
                                           send_dead_market=True)
                    last_dead_market_notified = True
                elif dual_result.mode != "dead":
                    last_dead_market_notified = False

                # ── Emitir señal ──────────────────────────────────────────────
                if dual_result.emit_signal and cooldown.can_emit(dual_result.mode):
                    metrics = await get_metrics(conn)

                    raw_history = [r[0] if isinstance(r,
                    ai_vote = await get_ai_vote(dual_result, raw_history, timeout=8.0)

                    if not ai_vote.vote:
                        log.info(
                            "🚫 IA bloqueó señal | reason='%s' tier=%s mode=%s target=%.1fx | block='%s'",
                            ai_vote.reason, ai_vote.tier, dual_result.mode,
                            dual_result.final_target, ai_vote.block_reason,
                        )
                    else:
                        # La IA puede ajustar tanto el target como el tier
                        if ai_vote.target > 0 and ai_vote.target != dual_result.final_target:
                            log.info(
                                "[IA] target ajustado: %.1fx → %.1fx (tier=%s)",
                                dual_result.final_target, ai_vote.target, ai_vote.tier,
                            )
                            dual_result.final_target = ai_vote.target

                        # Si la IA propone un tier diferente al del motor, respetarlo
                        if ai_vote.tier in ("safe", "hunter", "moon") and ai_vote.tier != dual_result.mode:
                            log.info("[IA] modo ajustado: %s → %s", dual_result.mode, ai_vote.tier)
                            dual_result.mode = ai_vote.tier
                            dual_result.final_signal_type = ai_vote.tier if ai_vote.tier == "safe" else (
                                "moon" if ai_vote.tier == "moon" else "high_risk"
                            )

                        # Ajustar confianza
                        dual_result.final_confidence = max(
                            1, min(99,
                            dual_result.final_confidence + ai_vote.confidence_adj)
                        )

                        # Guardar señal en DB
                        sig_id = await insert_dual_signal(
                            conn, ts=ts,
                            mode=dual_result.mode,
                            signal_type=dual_result.final_signal_type,
                            target=dual_result.final_target,
                            safe_score=dual_result.safe_score,
                            hunter_score=dual_result.hunter_score,
                            moon_score=dual_result.moon_score,
                            entropy_score=dual_result.entropy_score_val,
                            tension_score=dual_result.tension_score,
                            compression_score=dual_result.compression_score,
                            confidence=dual_result.final_confidence,
                            binary_rareness=dual_result.binary.rareness_score,
                            anomaly_score=dual_result.clustering.anomaly_score,
                            fibonacci_note=dual_result.fibonacci.note if dual_result.fibonacci else "",
                            context_logs=dual_result.context_logs,
                        )

                        sent = await send_dual_signal(dual_result, metrics, ts, last_val=val)

                        if sent:
                            cooldown.register_emission()
                            log.info(
                                "✅ Señal #%d emitida | mode=%s target=%.1fx conf=%d%% "
                                "safe=%d hunter=%d moon=%d | IA: %s",
                                sig_id, dual_result.mode, dual_result.final_target,
                                dual_result.final_confidence,
                                dual_result.safe_score, dual_result.hunter_score,
                                dual_result.moon_score, ai_vote.reason,
                            )

                            
                # ── Reporte de estado periódico ───────────────────────────────
                if round_counter % STATUS_REPORT_INTERVAL == 0:
                    stats = await get_dual_engine_stats(conn)
                    effectiveness = await get_mode_effectiveness(conn)
                    await send_status_v2(stats, effectiveness, ts)
                    log.info("[STATUS] Rondas=%d | %s", round_counter, str(stats.get("global", {})))

            except Exception as e:
                log_error(f"Error procesando ronda val={val}", e)

    await conn.close()
    log.info("Bot detenido limpiamente.")


def _handle_exit(signum, frame):
    log.info("Señal de sistema %d — deteniendo bot…", signum)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)
    try:
        asyncio.run(run())
    except SystemExit:
        log.info("Cierre limpio completado.")
    except Exception as e:
        log_error("Error fatal en el loop principal", e)
        sys.exit(1)
