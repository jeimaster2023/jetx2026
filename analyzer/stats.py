"""
analyzer/stats.py
Motor estadístico puro. Detecta el target más ambicioso estadísticamente
justificado: desde 1.90x hasta 15x según el contexto real.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional

from core.config import (
    SCORE_CONSERVATIVE,
    SCORE_INTERMEDIATE,
    SCORE_AGGRESSIVE,
    MIN_HISTORY,
    SPIKE_THRESHOLD,
    POST_SPIKE_BLOCK_ROUNDS,
)
from utils.logger import get_logger

log = get_logger("analyzer")

SIGNAL_NONE         = "none"
SIGNAL_CONSERVATIVE = "conservative"   # 1.90–2.00x
SIGNAL_INTERMEDIATE = "intermediate"   # 2.50–3.00x
SIGNAL_HIGH         = "high"           # 5.00–8.00x
SIGNAL_MOON         = "moon"           # 10.00–15.00x


@dataclass
class AnalysisResult:
    operable: bool
    block_reason: str = ""

    prob_190: float = 0.0
    prob_200: float = 0.0
    prob_250: float = 0.0
    prob_300: float = 0.0
    prob_500: float = 0.0
    prob_800: float = 0.0
    prob_1500: float = 0.0

    score: int = 0
    confidence: int = 0
    std_dev: float = 0.0
    crash_density: float = 0.0
    spike_density: float = 0.0
    cluster_label: str = ""

    signal_type: str = SIGNAL_NONE
    target: float = 0.0
    risk_level: str = "extreme"

    spike_block_active: bool = False
    rounds_until_unblock: int = 0


def _weighted_prob(history: list[float], target: float) -> float:
    w1, w2, w3 = 0.50, 0.30, 0.20
    h1 = history[:10]
    h2 = history[10:40]
    h3 = history[40:100]

    def hit(w: list[float]) -> float:
        return sum(1 for v in w if v >= target) / len(w) if w else 0.0

    wa = (w1 if h1 else 0) + (w2 if h2 else 0) + (w3 if h3 else 0)
    if wa == 0:
        return 0.0
    raw = (hit(h1)*(w1 if h1 else 0) + hit(h2)*(w2 if h2 else 0) + hit(h3)*(w3 if h3 else 0)) / wa
    return round(raw * 100, 2)


def _compute_score(prob_190, crash_density, spike_density, std_dev, history_recent):
    score_a = min(35, max(0, (prob_190 - 30) * (35 / 40)))
    score_b = max(0, 30 - crash_density * 50)
    if std_dev < 0.5:      score_c = 5
    elif std_dev <= 3.5:   score_c = 20
    elif std_dev <= 6.0:   score_c = 12
    else:                  score_c = 4
    consistent = sum(1 for v in history_recent[:10] if v >= 1.90) if history_recent else 0
    score_d = min(15, consistent * 1.5)
    return max(0, min(100, int(score_a + score_b + score_c + score_d)))


def _best_target(prob_190, prob_200, prob_250, prob_300,
                 prob_500, prob_800, prob_1500, crash_density, score):
    """
    Elige el target más alto estadísticamente justificado.
    Retorna (target, signal_type, risk_level).
    """
    # MOON 15x
    if score >= SCORE_AGGRESSIVE and crash_density < 0.20 and prob_1500 >= 8:
        return 15.0, SIGNAL_MOON, "very_high"

    # MOON 10x
    if score >= SCORE_AGGRESSIVE and crash_density < 0.20 and prob_800 >= 10:
        return 10.0, SIGNAL_MOON, "very_high"

    # HIGH 8x
    if score >= SCORE_INTERMEDIATE and crash_density < 0.28 and prob_800 >= 8:
        return 8.0, SIGNAL_HIGH, "high"

    # HIGH 5x
    if score >= SCORE_INTERMEDIATE and crash_density < 0.30 and prob_500 >= 12:
        return 5.0, SIGNAL_HIGH, "high"

    # INTERMEDIATE 3x
    if score >= SCORE_INTERMEDIATE and crash_density < 0.38 and prob_300 >= 22:
        return 3.0, SIGNAL_INTERMEDIATE, "medium"

    # INTERMEDIATE 2.5x
    if score >= SCORE_INTERMEDIATE and crash_density < 0.38 and prob_250 >= 28:
        return 2.5, SIGNAL_INTERMEDIATE, "medium"

    # CONSERVATIVE 2x
    if score >= SCORE_CONSERVATIVE and crash_density < 0.48 and prob_190 >= 38:
        if prob_200 >= 38:
            return 2.0, SIGNAL_CONSERVATIVE, "low"
        return 1.9, SIGNAL_CONSERVATIVE, "low"

    return 0.0, SIGNAL_NONE, "extreme"


def _describe_cluster(crash_density, spike_density, std_dev):
    if crash_density > 0.55:  return "Alta densidad de crashes"
    if spike_density > 0.20 and crash_density < 0.25: return "Sesión volátil con spikes"
    if crash_density < 0.20 and std_dev < 3.5:        return "Contexto estable"
    if crash_density < 0.35 and spike_density < 0.10: return "Contexto neutro"
    return "Contexto irregular"


class SpikeBlocker:
    def __init__(self):
        self._rounds_blocked = 0

    def register_round(self, val: float):
        if val >= SPIKE_THRESHOLD:
            self._rounds_blocked = POST_SPIKE_BLOCK_ROUNDS
            log.info("Anti-FOMO: spike %.2fx → bloqueando %d rondas.", val, POST_SPIKE_BLOCK_ROUNDS)
        elif self._rounds_blocked > 0:
            self._rounds_blocked -= 1

    @property
    def is_blocked(self): return self._rounds_blocked > 0

    @property
    def rounds_remaining(self): return self._rounds_blocked


def analyze(history: list[float], spike_blocker: SpikeBlocker,
            hourly_dist: Optional[dict]) -> AnalysisResult:

    if len(history) < MIN_HISTORY:
        return AnalysisResult(operable=False,
            block_reason=f"Calibrando — {len(history)}/{MIN_HISTORY} rondas")

    if spike_blocker.is_blocked:
        return AnalysisResult(operable=False,
            block_reason=f"Bloqueo post-spike ({spike_blocker.rounds_remaining} rondas)",
            spike_block_active=True,
            rounds_until_unblock=spike_blocker.rounds_remaining)

    work     = history[:100]
    recent15 = history[:15]
    recent30 = history[:30]

    prob_190  = _weighted_prob(work, 1.90)
    prob_200  = _weighted_prob(work, 2.00)
    prob_250  = _weighted_prob(work, 2.50)
    prob_300  = _weighted_prob(work, 3.00)
    prob_500  = _weighted_prob(work, 5.00)
    prob_800  = _weighted_prob(work, 8.00)
    prob_1500 = _weighted_prob(work, 15.00)

    crash_density = sum(1 for v in recent15 if v < 1.50) / len(recent15)
    spike_density = sum(1 for v in recent30 if v >= 5.0)  / len(recent30)
    std_dev       = statistics.stdev(recent30) if len(recent30) >= 2 else 0.0
    cluster_label = _describe_cluster(crash_density, spike_density, std_dev)

    score      = _compute_score(prob_190, crash_density, spike_density, std_dev, history)
    raw_conf   = score * 0.65 + prob_190 * 0.35
    confidence = int(max(0, min(100, raw_conf * (1 - crash_density * 0.4))))

    if hourly_dist and hourly_dist["crashes_low_pct"] > 55:
        score      = int(score * 0.75)
        confidence = int(confidence * 0.75)

    target, signal_type, risk_level = _best_target(
        prob_190, prob_200, prob_250, prob_300,
        prob_500, prob_800, prob_1500, crash_density, score)

    return AnalysisResult(
        operable      = signal_type != SIGNAL_NONE,
        block_reason  = "" if signal_type != SIGNAL_NONE else f"Score {score} insuficiente",
        prob_190=prob_190, prob_200=prob_200, prob_250=prob_250,
        prob_300=prob_300, prob_500=prob_500, prob_800=prob_800, prob_1500=prob_1500,
        score=score, confidence=confidence,
        std_dev=round(std_dev, 2),
        crash_density=round(crash_density, 3),
        spike_density=round(spike_density, 3),
        cluster_label=cluster_label,
        signal_type=signal_type, target=target, risk_level=risk_level,
        spike_block_active=spike_blocker.is_blocked,
        rounds_until_unblock=spike_blocker.rounds_remaining,
    )