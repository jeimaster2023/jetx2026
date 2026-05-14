"""
tg_bot/notifier_v2.py
═══════════════════════════════════════════════════════════════════════════════
Notificador Telegram para el sistema dual engine v3.

Formatos de mensaje:
  - SAFE MODE:   mensaje limpio y directo
  - HUNTER MODE: mensaje con contexto estadístico detallado
  - MOON MODE:   mensaje con análisis completo
  - DEAD MARKET: aviso de bloqueo
  - RESULTADO:   ganada / perdida

Todos los mensajes incluyen:
  - Score por modo
  - Entropía / Compresión / Tensión (cuando aplica)
  - Sin narrativa inventada — solo datos matemáticos
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import aiohttp

from core.config import TELEGRAM_TOKEN, CHAT_ID
from analyzer.dual_engine import DualEngineResult
from database.db import Metrics, ResolvedSignal
from utils.logger import get_logger, log_signal, log_error

log = get_logger("notifier_v2")

_TG_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# Íconos por modo
_MODE_ICON = {
    "safe":   "🔵",
    "hunter": "🔴",
    "moon":   "🌕",
    "dead":   "⛔",
    "none":   "⚪",
}

_RISK_LABEL = {
    "safe":     "CONSERVADOR",
    "high_risk": "AGRESIVO",
    "moon":     "MOON",
}


async def _send_raw(text: str) -> bool:
    """Envía mensaje Telegram HTML. Retorna True si fue exitoso."""
    try:
        # ssl=False necesario cuando hay proxy/antivirus con certificado autofirmado
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                _TG_URL,
                json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    return True
                body = await resp.text()
                log_error(f"Telegram HTTP {resp.status}: {body[:200]}", None)
                return False
    except asyncio.TimeoutError:
        log_error("Telegram timeout", None)
        return False
    except Exception as e:
        log_error("Telegram error", e)
        return False


def _bar(value: float, max_val: float = 100.0, length: int = 10) -> str:
    """Barra de progreso ASCII."""
    filled = int(round(value / max_val * length))
    filled = max(0, min(length, filled))
    return "█" * filled + "░" * (length - filled)


def build_safe_message(result: DualEngineResult, metrics: Metrics, ts: datetime, last_val: float = 0.0) -> str:
    safe = result.safe
    target = safe.target

    if target <= 2.00:
        riesgo = "🟢 Bajo"
    elif target <= 3.00:
        riesgo = "🟡 Medio"
    else:
        riesgo = "🔴 Alto"

    ultimo = f"después del <b>{last_val:.2f}x</b>\n" if last_val > 0 else ""

    msg = (
        f"¡PREPARA TU ENTRADA! ✈️\n"
        f"👉 Ingresa en la siguiente ronda.\n"
        f"{ultimo}"
        f"\n🏁 Retirar en: <b>{target:.2f}x</b>\n"
        f"⚠️ Riesgo: {riesgo}\n\n"
        f"📊 Score: {safe.safe_score}/100 | Conf: {safe.confidence}%\n"
        f"⏰ {ts.strftime('%H:%M:%S')}"
    )

    log_signal(
        f"type=safe target={target}x score={safe.safe_score} "
        f"confidence={safe.confidence}% last_val={last_val:.2f}"
    )
    return msg


def build_hunter_message(result: DualEngineResult, metrics: Metrics, ts: datetime, last_val: float = 0.0) -> str:
    hunter = result.hunter
    target = hunter.target
    is_moon = hunter.mode == "moon"

    if target >= 50:
        riesgo = "🔴 Muy Alto"
    elif target >= 20:
        riesgo = "🔴 Alto"
    elif target >= 10:
        riesgo = "🟠 Alto"
    else:
        riesgo = "🟡 Medio-Alto"

    score = hunter.moon_score if is_moon else hunter.hunter_score
    ultimo = f"después del <b>{last_val:.2f}x</b>\n" if last_val > 0 else ""

    msg = (
        f"¡PREPARA TU ENTRADA! ✈️\n"
        f"👉 Ingresa en la siguiente ronda.\n"
        f"{ultimo}"
        f"\n🏁 Retirar en: <b>{target:.2f}x</b>\n"
        f"⚠️ Riesgo: {riesgo}\n\n"
        f"📊 Score: {score}/100 | Conf: {hunter.confidence}%\n"
        f"⏰ {ts.strftime('%H:%M:%S')}"
    )

    log_signal(
        f"type={hunter.mode} target={target}x "
        f"hunter_score={hunter.hunter_score} moon_score={hunter.moon_score} "
        f"confidence={hunter.confidence}% last_val={last_val:.2f}"
    )
    return msg


def build_dead_market_message(result: DualEngineResult, ts: datetime) -> str:
    dead = result.dead_market
    lines = [
        f"⛔ <b>DEAD MARKET — Señales bloqueadas</b>",
        "",
        f"Causa: {dead.reason}",
        f"🔀 Entropía: {dead.entropy_score:.3f}",
        f"☠️  Toxicidad: {dead.toxicity_score:.3f}",
        f"⏰ {ts.strftime('%H:%M:%S')}",
    ]
    return "\n".join(lines)


async def send_dual_signal(
    result: DualEngineResult,
    metrics: Metrics,
    ts: datetime,
    send_dead_market: bool = False,
    last_val: float = 0.0,
) -> bool:
    if result.mode == "dead" and send_dead_market:
        msg = build_dead_market_message(result, ts)
        return await _send_raw(msg)

    if not result.emit_signal:
        return False

    if result.mode == "safe":
        msg = build_safe_message(result, metrics, ts, last_val=last_val)
    elif result.mode in ("hunter", "moon"):
        msg = build_hunter_message(result, metrics, ts, last_val=last_val)
    else:
        return False

    return await _send_raw(msg)


def build_result_message(
    signal_id: int,
    mode: str,
    target: float,
    result_val: float,
    won: bool,
    rounds_passed: int,
) -> str:
    if won:
        return (
            f"✅ <b>GANADA</b> — {target:.2f}x\n"
            f"Resultado: {result_val:.2f}x"
        )
    else:
        return (
            f"❌ <b>PERDIDA</b> — {target:.2f}x\n"
            f"Resultado: {result_val:.2f}x"
        )


async def send_result_v2(
    signal_id: int,
    mode: str,
    target: float,
    result_val: float,
    won: bool,
    rounds_passed: int,
) -> bool:
    msg = build_result_message(signal_id, mode, target, result_val, won, rounds_passed)
    return await _send_raw(msg)


async def send_status_v2(
    stats: dict,
    mode_effectiveness: list,
    ts: datetime,
) -> bool:
    """Envía resumen de estado del sistema."""
    g = stats.get("global", {})
    lines = [
        "📊 <b>Estado del sistema JetX v3</b>",
        "",
        f"Global: {g.get('wins', 0)}W/{g.get('total', 0) - g.get('wins', 0)}L "
        f"({g.get('winrate', 0):.1f}%)",
        "",
        "── Por modo ──",
    ]

    for mode in ("safe", "hunter", "moon"):
        s = stats.get(mode, {})
        if s.get("total", 0) > 0:
            icon = _MODE_ICON.get(mode, "⚪")
            lines.append(
                f"{icon} {mode.upper()}: {s['wins']}W/{s['total']-s['wins']}L "
                f"({s['winrate']:.1f}%) | conf≈{s.get('avg_confidence', 0):.0f}%"
            )

    if mode_effectiveness:
        lines.append("")
        lines.append("── Por target ──")
        for eff in mode_effectiveness[:8]:
            if eff.total > 0:
                lines.append(
                    f"{eff.mode} {eff.target:.1f}x: "
                    f"{eff.wins}/{eff.total} ({eff.winrate:.0f}%)"
                )

    lines.append(f"\n⏰ {ts.strftime('%Y-%m-%d %H:%M:%S')}")
    return await _send_raw("\n".join(lines))