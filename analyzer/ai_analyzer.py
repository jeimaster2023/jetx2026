"""
analyzer/ai_analyzer.py
═══════════════════════════════════════════════════════════════════════════════
IA como FILTRO REAL y ANALISTA DE PATRONES.

La IA recibe:
  - Historial de las últimas 50 rondas (valores reales)
  - Probabilidades ponderadas 50/30/20 (como la versión Pro)
  - Scores del motor dual
  - Análisis de patrones: falso impulso, recurrencias, momentum, densidad
  - Contexto estadístico completo

Y devuelve JSON con decisión de 3 niveles:
  {
    "vote": "YES" | "NO",
    "tier": "safe" | "hunter" | "moon",   ← nivel de señal sugerido
    "target": 2.0,                         ← target exacto sugerido
    "reason": "...",                       ← máximo 15 palabras
    "confidence_adj": 5,                   ← ajuste -20..+20
    "block_reason": ""                     ← por qué bloquea si vote=NO
  }

Criterios estrictos por tier:
  SAFE  (2x–3x): crash_density < 0.35, prob_200 >= 50%, score >= 50
  HUNTER (5x–8x): compression >= 0.40, tension >= 35, prob_500 >= 15%
  MOON  (10x+):   moon_score >= 55, overdue en 10x+, compression >= 0.50

Si la IA dice NO → señal bloqueada completamente.
Si Groq falla → fallback local con criterios más estrictos que antes.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from typing import Optional

from core.config import GROQ_KEY, GROQ_MODEL, GROQ_TEMPERATURE
from analyzer.dual_engine import DualEngineResult
from utils.logger import get_logger, log_error

log = get_logger("ai_analyzer")

_MODEL = "llama-3.3-70b-versatile"
_MAX_TOKENS = 220


# ══════════════════════════════════════════════════════════════════════════════
# Análisis de patrones locales (alimentan tanto a Groq como al fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _weighted_prob(history: list[float], target: float) -> float:
    """Probabilidad ponderada 50/30/20 — igual que en JetX Pro."""
    h1 = history[:10]
    h2 = history[10:40]
    h3 = history[40:100]

    def hit(w):
        return sum(1 for v in w if v >= target) / len(w) if w else 0.0

    wa = (0.50 if h1 else 0) + (0.30 if h2 else 0) + (0.20 if h3 else 0)
    if wa == 0:
        return 0.0
    raw = (hit(h1) * (0.50 if h1 else 0)
           + hit(h2) * (0.30 if h2 else 0)
           + hit(h3) * (0.20 if h3 else 0)) / wa
    return round(raw * 100, 2)


def _detect_false_impulse(history: list[float]) -> bool:
    """Ronda anterior >= 5x seguida de 3 crashes < 1.5x = trampa."""
    if len(history) < 4:
        return False
    return history[0] >= 5.0 and all(v < 1.50 for v in history[1:4])


def _detect_recurrence_pattern(history: list[float]) -> dict:
    """
    Detecta patrones de recurrencia en el historial.
    Busca si el mercado tiene ciclos de baja → spike identificables.
    """
    if len(history) < 20:
        return {"found": False, "avg_cycle": 0, "confidence": 0}

    # Encontrar posiciones de spikes >= 3x
    spike_positions = [i for i, v in enumerate(history[:50]) if v >= 3.0]
    if len(spike_positions) < 2:
        return {"found": False, "avg_cycle": 0, "confidence": 0}

    # Calcular distancias entre spikes
    distances = [spike_positions[i] - spike_positions[i+1]
                 for i in range(len(spike_positions)-1)]
    if not distances:
        return {"found": False, "avg_cycle": 0, "confidence": 0}

    avg_cycle = round(statistics.mean(distances), 1)
    # Si el ciclo es consistente (baja desviación), hay patrón
    if len(distances) >= 2:
        std = statistics.stdev(distances)
        consistency = max(0, 1 - std / max(avg_cycle, 1))
    else:
        consistency = 0.5

    last_spike_ago = spike_positions[0] if spike_positions else 999

    return {
        "found": consistency >= 0.4,
        "avg_cycle": avg_cycle,
        "last_spike_ago": last_spike_ago,
        "consistency": round(consistency, 2),
        "due_soon": last_spike_ago >= avg_cycle * 0.8 if avg_cycle > 0 else False,
    }


def _analyze_momentum(history: list[float]) -> str:
    """EMA corta vs larga para momentum — mismo método que JetX Pro."""
    if len(history) < 20:
        return "insuficiente"
    datos = list(reversed(history[:30]))

    def ema(data, n):
        k = 2 / (n + 1)
        e = sum(data[:n]) / n
        for v in data[n:]:
            e = v * k + e * (1 - k)
        return e

    ema5  = ema(datos, 5)
    ema20 = ema(datos, min(20, len(datos)))

    if ema5 > ema20 * 1.15:   return "alcista_fuerte"
    elif ema5 > ema20:         return "alcista"
    elif ema5 < ema20 * 0.85:  return "bajista_fuerte"
    else:                      return "neutral"


def _build_pattern_context(history: list[float], result: DualEngineResult) -> dict:
    """Construye el contexto completo de patrones para la IA."""
    h = history[:100]

    prob_190  = _weighted_prob(h, 1.90)
    prob_200  = _weighted_prob(h, 2.00)
    prob_300  = _weighted_prob(h, 3.00)
    prob_500  = _weighted_prob(h, 5.00)
    prob_1000 = _weighted_prob(h, 10.00)

    crash_density_15 = sum(1 for v in h[:15] if v < 1.50) / max(len(h[:15]), 1)
    crash_density_30 = sum(1 for v in h[:30] if v < 1.50) / max(len(h[:30]), 1)
    spike_density    = sum(1 for v in h[:30] if v >= 5.0) / max(len(h[:30]), 1)

    false_impulse  = _detect_false_impulse(h)
    recurrence     = _detect_recurrence_pattern(h)
    momentum       = _analyze_momentum(h)

    # Racha actual (cuántas consecutivas < 2x)
    racha_baja = 0
    for v in h:
        if v < 2.0:
            racha_baja += 1
        else:
            break

    # Últimas 5 rondas
    ultimas5 = [round(v, 2) for v in h[:5]]

    safe  = result.safe
    hunter = result.hunter

    return {
        "mode_proposed": result.mode,
        "target_proposed": result.final_target,
        # Scores del motor
        "safe_score": result.safe_score,
        "hunter_score": result.hunter_score,
        "moon_score": result.moon_score,
        "entropy": round(result.entropy_score_val, 3),
        "compression": round(result.compression_score, 3),
        "tension": round(result.tension_score, 1),
        "anomaly_score": round(result.clustering.anomaly_score, 3),
        "context_cluster": result.clustering.context_label,
        "overdue_levels": [int(l) for l in (hunter.overdue_levels if hunter else [])],
        # Probabilidades ponderadas 50/30/20
        "prob_190_pct": prob_190,
        "prob_200_pct": prob_200,
        "prob_300_pct": prob_300,
        "prob_500_pct": prob_500,
        "prob_1000_pct": prob_1000,
        # Densidades
        "crash_density_15r": round(crash_density_15, 3),
        "crash_density_30r": round(crash_density_30, 3),
        "spike_density_30r": round(spike_density, 3),
        # Patrones
        "false_impulse_detected": false_impulse,
        "momentum": momentum,
        "racha_baja_consecutiva": racha_baja,
        "recurrence_pattern": recurrence,
        "last_10_rounds": [round(v, 2) for v in h[:10]],
        "ultimas_5": ultimas5,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Prompt para Groq
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
Eres un filtro estadístico estricto para JetX (juego crash). Tu única función es decidir si el contexto actual justifica emitir una señal, y en qué nivel.

RESPONDE SOLO con JSON válido. Sin texto extra, sin markdown, sin explicaciones fuera del JSON.

Formato exacto:
{"vote":"YES","tier":"safe","target":2.0,"reason":"prob 200x alta con crash density baja","confidence_adj":8,"block_reason":""}

Campos:
- "vote": "YES" o "NO"
- "tier": "safe" | "hunter" | "moon"
- "target": número float exacto (1.9, 2.0, 3.0, 5.0, 8.0, 10.0)
- "reason": máximo 12 palabras, solo datos matemáticos
- "confidence_adj": entero -20 a +20
- "block_reason": si vote=NO, razón en máximo 8 palabras

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITERIOS ESTRICTOS — debes cumplir TODOS los de un tier para votar YES:

TIER SAFE (target 2x o 3x):
  ✓ crash_density_15r < 0.35
  ✓ prob_200_pct >= 50%
  ✓ safe_score >= 50
  ✓ momentum NO es "bajista_fuerte"
  ✓ false_impulse_detected = false
  → target: 3.0 si prob_300_pct >= 35% y safe_score >= 60, sino 2.0

TIER HUNTER (target 5x u 8x):
  ✓ compression >= 0.40
  ✓ tension >= 35
  ✓ prob_500_pct >= 12%
  ✓ hunter_score >= 50
  ✓ crash_density_30r < 0.45
  → target: 8.0 si overdue_levels contiene 8, sino 5.0

TIER MOON (target 10x):
  ✓ moon_score >= 55
  ✓ overdue_levels contiene 10 o más
  ✓ compression >= 0.50
  ✓ crash_density_15r < 0.40
  ✓ entropy < 0.85
  NO votar moon si recurrence_pattern.due_soon = false y moon_score < 65

BLOQUES AUTOMÁTICOS (vota NO sin importar scores):
  ✗ false_impulse_detected = true
  ✗ crash_density_15r >= 0.60
  ✗ momentum = "bajista_fuerte" para safe/hunter
  ✗ racha_baja_consecutiva >= 8
  ✗ entropy >= 0.90
  ✗ las últimas 3 rondas son todas > 5x (post-spike FOMO)

PRIORIDAD: si safe Y hunter se activan → preferir safe (más win rate).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def _build_prompt(ctx: dict) -> str:
    return (
        f"Contexto actual: {json.dumps(ctx, ensure_ascii=False)}\n"
        "¿Se justifica emitir señal? Responde SOLO con el JSON."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Dataclass de resultado
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AIVote:
    vote: bool
    tier: str          # "safe" | "hunter" | "moon"
    target: float
    reason: str
    confidence_adj: int
    block_reason: str
    source: str        # "groq" | "local" | "parse_error"


# ══════════════════════════════════════════════════════════════════════════════
# Parser de respuesta Groq
# ══════════════════════════════════════════════════════════════════════════════

def _parse_ai_response(raw: str, fallback_target: float, fallback_tier: str) -> AIVote:
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        # Buscar JSON dentro del texto si hay basura alrededor
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start >= 0 and end > start:
            clean = clean[start:end]
        obj = json.loads(clean)

        vote   = str(obj.get("vote", "NO")).upper() == "YES"
        tier   = str(obj.get("tier", fallback_tier)).lower()
        if tier not in ("safe", "hunter", "moon"):
            tier = fallback_tier
        target = float(obj.get("target", fallback_target))
        reason = str(obj.get("reason", ""))[:100]
        adj    = max(-20, min(20, int(obj.get("confidence_adj", 0))))
        block  = str(obj.get("block_reason", ""))[:80]

        return AIVote(vote=vote, tier=tier, target=target, reason=reason,
                      confidence_adj=adj, block_reason=block, source="groq")

    except Exception as e:
        log.debug("Error parseando respuesta IA: %s | raw=%s", e, raw[:120])
        return AIVote(vote=True, target=fallback_target, tier=fallback_tier,
                      reason="parse error — fallback aprobado",
                      confidence_adj=0, block_reason="", source="parse_error")


# ══════════════════════════════════════════════════════════════════════════════
# Fallback local — criterios estrictos
# ══════════════════════════════════════════════════════════════════════════════

def _local_vote(result: DualEngineResult, history: list[float]) -> AIVote:
    """
    Voto local sin API. Replica los criterios del system prompt
    con lógica Python pura. Más estricto que el anterior fallback.
    """
    ctx = _build_pattern_context(history, result)

    # ── Bloques automáticos ────────────────────────────────────────────────────
    last3_spike = all(v >= 5.0 for v in history[:3]) if len(history) >= 3 else False

    if ctx["false_impulse_detected"]:
        return AIVote(vote=False, tier="none", target=0.0,
                      reason="falso impulso detectado — trampa", confidence_adj=-15,
                      block_reason="false impulse activo", source="local")

    if ctx["crash_density_15r"] >= 0.60:
        return AIVote(vote=False, tier="none", target=0.0,
                      reason="crash density >= 60% — mercado tóxico", confidence_adj=-15,
                      block_reason="crash density crítica", source="local")

    if ctx["racha_baja_consecutiva"] >= 8:
        return AIVote(vote=False, tier="none", target=0.0,
                      reason="racha de 8+ crashes consecutivos", confidence_adj=-10,
                      block_reason="racha baja extrema", source="local")

    if last3_spike:
        return AIVote(vote=False, tier="none", target=0.0,
                      reason="3 spikes seguidos — riesgo post-FOMO", confidence_adj=-10,
                      block_reason="post-spike FOMO", source="local")

    if ctx["entropy"] >= 0.90:
        return AIVote(vote=False, tier="none", target=0.0,
                      reason="entropía extrema — mercado caótico", confidence_adj=-10,
                      block_reason="entropia >= 0.90", source="local")

    mode = result.mode

    # ── SAFE ──────────────────────────────────────────────────────────────────
    if mode == "safe":
        ok = (
            ctx["crash_density_15r"] < 0.35
            and ctx["prob_200_pct"] >= 50.0
            and result.safe_score >= 50
            and ctx["momentum"] != "bajista_fuerte"
        )
        if ok:
            target = 3.0 if (ctx["prob_300_pct"] >= 35.0 and result.safe_score >= 60) else 2.0
            adj = +8 if result.safe_score >= 65 else +4
            return AIVote(vote=True, tier="safe", target=target,
                          reason=f"prob200={ctx['prob_200_pct']:.0f}% crash={ctx['crash_density_15r']:.2f}",
                          confidence_adj=adj, block_reason="", source="local")
        else:
            missing = []
            if ctx["crash_density_15r"] >= 0.35: missing.append(f"crash={ctx['crash_density_15r']:.2f}>=0.35")
            if ctx["prob_200_pct"] < 50.0:       missing.append(f"prob200={ctx['prob_200_pct']:.0f}%<50%")
            if result.safe_score < 50:            missing.append(f"score={result.safe_score}<50")
            if ctx["momentum"] == "bajista_fuerte": missing.append("momentum bajista")
            return AIVote(vote=False, tier="safe", target=0.0,
                          reason=f"condiciones safe no cumplidas ({len(missing)} fallas)",
                          confidence_adj=-5, block_reason=", ".join(missing[:2]), source="local")

    # ── HUNTER ────────────────────────────────────────────────────────────────
    if mode == "hunter":
        factores = sum([
            ctx["compression"] >= 0.40,
            ctx["tension"] >= 35,
            ctx["prob_500_pct"] >= 12.0,
            result.hunter_score >= 50,
            ctx["crash_density_30r"] < 0.45,
        ])
        ok = factores >= 4  # todos o casi todos
        if ok:
            target = 8.0 if (8 in ctx["overdue_levels"]) else 5.0
            adj = min(15, factores * 3)
            return AIVote(vote=True, tier="hunter", target=target,
                          reason=f"{factores}/5 factores hunter — compression={ctx['compression']:.2f}",
                          confidence_adj=adj, block_reason="", source="local")
        else:
            return AIVote(vote=False, tier="hunter", target=0.0,
                          reason=f"solo {factores}/5 factores hunter activos",
                          confidence_adj=-5, block_reason=f"{factores}/5 factores insuficientes",
                          source="local")

    # ── MOON ──────────────────────────────────────────────────────────────────
    if mode == "moon":
        overdue_high = any(l >= 10 for l in ctx["overdue_levels"])
        recurrence_ok = ctx["recurrence_pattern"].get("due_soon", False) or result.moon_score >= 65
        ok = (
            result.moon_score >= 55
            and overdue_high
            and ctx["compression"] >= 0.50
            and ctx["crash_density_15r"] < 0.40
            and ctx["entropy"] < 0.85
            and recurrence_ok
        )
        if ok:
            return AIVote(vote=True, tier="moon", target=10.0,
                          reason=f"moon_score={result.moon_score} overdue={ctx['overdue_levels']} comp={ctx['compression']:.2f}",
                          confidence_adj=+10, block_reason="", source="local")
        else:
            missing = []
            if result.moon_score < 55:               missing.append(f"moon_score={result.moon_score}<55")
            if not overdue_high:                      missing.append("sin overdue en 10x+")
            if ctx["compression"] < 0.50:            missing.append(f"comp={ctx['compression']:.2f}<0.50")
            if not recurrence_ok:                    missing.append("recurrencia no confirma")
            return AIVote(vote=False, tier="moon", target=0.0,
                          reason=f"condiciones moon insuficientes",
                          confidence_adj=-8, block_reason=", ".join(missing[:2]), source="local")

    # ── Otros modos ────────────────────────────────────────────────────────────
    return AIVote(vote=False, tier="none", target=0.0,
                  reason="modo no votable", confidence_adj=0,
                  block_reason="modo inactivo", source="local")


# ══════════════════════════════════════════════════════════════════════════════
# Función principal
# ══════════════════════════════════════════════════════════════════════════════

async def get_ai_vote(
    result: DualEngineResult,
    history: list[float],
    timeout: float = 8.0,
) -> AIVote:
    """
    Consulta Groq con análisis completo de patrones.
    Si Groq falla → fallback local estricto.
    """
    raw_history = [r[0] if isinstance(r, (list, tuple)) else r for r in history]
    ctx = _build_pattern_context(raw_history, result)

    fallback_target = result.final_target
    fallback_tier   = result.mode if result.mode in ("safe", "hunter", "moon") else "safe"

    if not GROQ_KEY or GROQ_KEY in ("", "YOUR_GROQ_KEY"):
        log.debug("Sin GROQ_KEY — usando fallback local estricto")
        vote = _local_vote(result, raw_history)
        log.info(
            "[IA-LOCAL] vote=%s tier=%s target=%.1fx reason='%s'",
            "✅YES" if vote.vote else "❌NO",
            vote.tier, vote.target, vote.reason,
        )
        return vote

    try:
        import asyncio
        from groq import AsyncGroq

        client = AsyncGroq(api_key=GROQ_KEY)
        prompt = _build_prompt(ctx)

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.05,
                max_tokens=_MAX_TOKENS,
            ),
            timeout=timeout,
        )

        raw = response.choices[0].message.content.strip()
        ai_vote = _parse_ai_response(raw, fallback_target, fallback_tier)

        log.info(
            "[IA-GROQ] vote=%s tier=%s target=%.1fx reason='%s' adj=%+d",
            "✅YES" if ai_vote.vote else "❌NO",
            ai_vote.tier, ai_vote.target, ai_vote.reason, ai_vote.confidence_adj,
        )
        return ai_vote

    except ImportError:
        log.warning("groq no instalado — fallback local. Instala: pip install groq")
        return _local_vote(result, raw_history)
    except Exception as e:
        log_error("IA Groq falló — usando fallback local", e)
        return _local_vote(result, raw_history)


# ── Compatibilidad ────────────────────────────────────────────────────────────
async def get_ai_interpretation(result: DualEngineResult) -> str:
    vote = await get_ai_vote(result, [])
    return f"vote={vote.vote} tier={vote.tier} reason='{vote.reason}' adj={vote.confidence_adj:+d}"
