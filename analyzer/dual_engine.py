"""
analyzer/dual_engine.py  — JetX v3 CORREGIDO
═══════════════════════════════════════════════════════════════════════════════
Cambios respecto al original:

1. Umbrales subidos a niveles que generan ganancia real:
   SAFE_MIN_SCORE    22 → 50
   HUNTER_MIN_SCORE  22 → 45
   MOON_MIN_SCORE    18 → 55

2. Targets realistas:
   MOON_TARGETS ahora solo va hasta 10x (no 50x/100x/1000x)
   HUNTER_TARGETS simplificado a 5x y 8x

3. Detección de falso impulso integrada (de JetX Pro):
   Si ronda[0] >= 5x y rondas[1:4] < 1.5x → bloqueo automático

4. Ventana de resolución de señales: 3 rondas (ver main_v3.py)

Todos los demás módulos (entropy, advanced_stats, ml_clustering) sin cambios.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

from analyzer.advanced_stats import (
    AdvancedAnalysisResult,
    CompressionAnalysis,
    TensionIndex,
    TemporalMap,
    PreSpikePattern,
    SPIKE_LEVELS,
)
from analyzer.entropy import (
    EntropyResult,
    BinarySequenceResult,
    FibonacciResult,
    DeadMarketResult,
    analyze_entropy,
    analyze_binary_sequences,
    analyze_fibonacci,
    evaluate_dead_market,
)
from analyzer.ml_clustering import (
    ClusteringResult,
    ContextMemory,
    classify_context,
    CONTEXT_SAFE,
    CONTEXT_HUNTER,
    CONTEXT_CHAOTIC,
)
from utils.logger import get_logger

log = get_logger("dual_engine")

# ── Thresholds SAFE MODE — subidos para filtrar señales malas ─────────────────
SAFE_MIN_SCORE          = 50   # era 22 → subido a 50
SAFE_MAX_CRASH_DENSITY  = 0.40 # era 0.55 → más estricto
SAFE_MAX_ENTROPY        = 0.85
SAFE_MIN_PROB_190       = 45.0 # era 25.0 → más estricto

# ── Thresholds HUNTER MODE ────────────────────────────────────────────────────
HUNTER_MIN_SCORE        = 45   # era 22 → subido a 45
HUNTER_MAX_CRASH_DENSITY = 0.45
HUNTER_MIN_TENSION      = 30.0 # era 10.0
HUNTER_MIN_COMPRESSION  = 0.35 # era 0.15

# ── Thresholds MOON — solo cuando el contexto es muy claro ───────────────────
MOON_MIN_SCORE          = 55   # era 18 → subido a 55
MOON_MIN_TENSION        = 30.0
MOON_MAX_CRASH_DENSITY  = 0.40 # era 0.60

# ── Targets realistas ─────────────────────────────────────────────────────────
SAFE_TARGETS   = [1.90, 2.00, 3.00]
HUNTER_TARGETS = [5.0, 8.0]
MOON_TARGETS   = [10.0]          # era hasta 1000x — ahora solo 10x


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses de resultado (sin cambios estructurales)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SafeModeResult:
    active: bool
    target: float
    safe_score: int
    confidence: int
    reason_on: str
    reason_off: str
    prob_190: float
    prob_200: float
    prob_250: float
    prob_300: float
    crash_density: float
    std_dev: float


@dataclass
class HunterModeResult:
    active: bool
    mode: str
    target: float
    hunter_score: int
    moon_score: int
    confidence: int
    reason_on: str
    reason_off: str
    compression_contrib: float
    tension_contrib: float
    temporal_contrib: float
    overdue_levels: list[float]
    fibonacci_note: str
    binary_rareness: float
    anomaly_score: float


@dataclass
class DualEngineResult:
    mode: str
    safe: SafeModeResult
    hunter: HunterModeResult
    entropy: EntropyResult
    binary: BinarySequenceResult
    fibonacci: FibonacciResult
    dead_market: DeadMarketResult
    clustering: ClusteringResult
    safe_score: int
    hunter_score: int
    moon_score: int
    entropy_score_val: float
    tension_score: float
    compression_score: float
    emit_signal: bool
    final_target: float
    final_confidence: int
    final_signal_type: str
    context_logs: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Detector de falso impulso (importado de JetX Pro)
# ══════════════════════════════════════════════════════════════════════════════

def _detect_false_impulse(history: list[float]) -> bool:
    """
    Ronda[0] >= 5x seguida de 3 rondas < 1.5x = trampa post-spike.
    Bloquea señales automáticamente.
    """
    if len(history) < 4:
        return False
    return history[0] >= 5.0 and all(v < 1.50 for v in history[1:4])


# ══════════════════════════════════════════════════════════════════════════════
# SAFE MODE engine
# ══════════════════════════════════════════════════════════════════════════════

def _compute_safe_score(
    prob_190: float,
    prob_200: float,
    prob_250: float,
    prob_300: float,
    crash_density: float,
    std_dev: float,
    entropy_score: float,
    history_recent: list[float],
    recent_spikes_window5: bool,
) -> int:
    score = 0

    # A. Probabilidad 1.90x (0–30 pts)
    score += int(min(30, max(0, (prob_190 - 30) * (30 / 40))))

    # B. Densidad baja de crashes (0–25 pts) — más estricto
    score += int(max(0, 25 - crash_density * 70))

    # C. Volatilidad controlada (0–20 pts)
    if std_dev < 0.5:      score += 5
    elif std_dev <= 2.5:   score += 20
    elif std_dev <= 4.0:   score += 12
    elif std_dev <= 7.0:   score += 6
    else:                  score += 2

    # D. Consistencia reciente (0–15 pts)
    recent10 = history_recent[:10]
    consistent = sum(1 for v in recent10 if v >= 1.90)
    score += min(15, consistent * 1.5)

    # E. Penalizar entropía alta
    if entropy_score >= 0.80:   score -= 10
    elif entropy_score >= 0.70: score -= 5

    # F. Penalizar spikes recientes (anti-FOMO)
    if recent_spikes_window5:
        score -= 12  # era -8, más fuerte

    return max(0, min(100, score))


def _best_safe_target(
    prob_190: float,
    prob_200: float,
    prob_250: float,
    prob_300: float,
    crash_density: float,
    safe_score: int,
) -> float:
    # 3.00x — solo con score alto y prob validada
    if safe_score >= 60 and crash_density < 0.30 and prob_300 >= 35:
        return 3.00
    # 2.00x — estándar
    if safe_score >= SAFE_MIN_SCORE and crash_density < SAFE_MAX_CRASH_DENSITY and prob_200 >= 50:
        return 2.00
    # 1.90x — mínimo
    if safe_score >= SAFE_MIN_SCORE and prob_190 >= SAFE_MIN_PROB_190:
        return 1.90
    return 0.0


def evaluate_safe_mode(
    history: list[float],
    prob_190: float,
    prob_200: float,
    prob_250: float,
    prob_300: float,
    crash_density: float,
    entropy: EntropyResult,
    adv_result: Optional[AdvancedAnalysisResult] = None,
) -> SafeModeResult:

    if len(history) < 15:
        return SafeModeResult(
            active=False, target=0.0, safe_score=0, confidence=0,
            reason_on="", reason_off="Datos insuficientes",
            prob_190=prob_190, prob_200=prob_200, prob_250=prob_250,
            prob_300=prob_300, crash_density=crash_density, std_dev=0.0,
        )

    try:
        std_dev = statistics.stdev(history[:15])
    except statistics.StatisticsError:
        std_dev = 0.0

    recent_spike5 = any(v >= 5.0 for v in history[:5])

    # Bloqueo por falso impulso
    if _detect_false_impulse(history):
        return SafeModeResult(
            active=False, target=0.0, safe_score=0, confidence=0,
            reason_on="", reason_off="Falso impulso detectado — trampa post-spike",
            prob_190=prob_190, prob_200=prob_200, prob_250=prob_250,
            prob_300=prob_300, crash_density=crash_density, std_dev=round(std_dev, 2),
        )

    score = _compute_safe_score(
        prob_190, prob_200, prob_250, prob_300,
        crash_density, std_dev, entropy.entropy_score,
        history, recent_spike5,
    )

    reasons_off = []
    if entropy.is_dead_market:
        reasons_off.append("Dead Market activo")
    if crash_density >= SAFE_MAX_CRASH_DENSITY:
        reasons_off.append(f"crash_density={crash_density:.2%} ≥ {SAFE_MAX_CRASH_DENSITY:.2%}")
    if entropy.entropy_score >= SAFE_MAX_ENTROPY:
        reasons_off.append(f"entropía={entropy.entropy_score:.3f} demasiado alta")

    if reasons_off:
        return SafeModeResult(
            active=False, target=0.0, safe_score=score, confidence=0,
            reason_on="", reason_off=" | ".join(reasons_off),
            prob_190=prob_190, prob_200=prob_200, prob_250=prob_250,
            prob_300=prob_300, crash_density=crash_density, std_dev=round(std_dev, 2),
        )

    target = _best_safe_target(prob_190, prob_200, prob_250, prob_300, crash_density, score)
    if target == 0.0:
        return SafeModeResult(
            active=False, target=0.0, safe_score=score, confidence=0,
            reason_on="", reason_off=f"Score insuficiente ({score}/100 — mínimo {SAFE_MIN_SCORE})",
            prob_190=prob_190, prob_200=prob_200, prob_250=prob_250,
            prob_300=prob_300, crash_density=crash_density, std_dev=round(std_dev, 2),
        )

    conf = int(score * 0.65 + prob_190 * 0.35)
    conf = int(conf * (1 - crash_density * 0.4))
    conf = max(0, min(100, conf))

    reason_on = (f"score={score} | crash={crash_density:.2%} | "
                 f"P(≥{target}x)={prob_200:.1f}% | std={std_dev:.2f}")

    return SafeModeResult(
        active=True, target=target, safe_score=score, confidence=conf,
        reason_on=reason_on, reason_off="",
        prob_190=prob_190, prob_200=prob_200, prob_250=prob_250,
        prob_300=prob_300, crash_density=crash_density, std_dev=round(std_dev, 2),
    )


# ══════════════════════════════════════════════════════════════════════════════
# HUNTER MODE engine
# ══════════════════════════════════════════════════════════════════════════════

def _compute_hunter_score(
    compression: Optional[CompressionAnalysis],
    tension: Optional[TensionIndex],
    temporal_map: Optional[TemporalMap],
    pre_patterns: list[PreSpikePattern],
    binary: BinarySequenceResult,
    fibonacci: FibonacciResult,
    clustering: ClusteringResult,
    base_prob_500: float,
    base_prob_800: float,
    crash_density: float,
) -> int:
    score = 0

    # A. Compresión (0–25 pts)
    if compression:
        score += int(compression.compression_score * 25)
        if compression.consecutive_ultra_low >= 3:
            score += min(10, compression.consecutive_ultra_low * 3)

    # B. Tensión (0–20 pts)
    if tension:
        score += int(tension.value * 0.20)

    # C. Distancia temporal (0–25 pts)
    if temporal_map:
        overdue = temporal_map.overdue_levels()
        very_overdue = temporal_map.very_overdue_levels()
        score += min(15, len(overdue) * 5)
        score += min(10, len(very_overdue) * 5)

    # D. Patrones pre-spike (0–15 pts)
    for pp in pre_patterns:
        if pp.level in [5.0, 8.0] and pp.current_matches:
            score += 10
        elif pp.level in [10.0, 20.0] and pp.current_matches:
            score += 15

    # E. Rareza binaria (0–10 pts)
    score += int(binary.rareness_score * 10)

    # F. Fibonacci (0–5 pts)
    if fibonacci.compression_detected or fibonacci.expansion_detected:
        score += 5

    # G. Clustering (0–10 pts)
    if clustering.context_label == CONTEXT_HUNTER:
        score += int(min(10, clustering.context_score * 0.10))
    if clustering.is_anomaly:
        score += 5

    # H. Probabilidades base (0–10 pts)
    score += int(base_prob_500 * 0.15)
    score += int(base_prob_800 * 0.10)

    # I. Penalización por crash reciente — más agresiva
    if crash_density >= 0.40:
        score -= int(crash_density * 30)

    return max(0, min(100, score))


def _compute_moon_score(
    compression: Optional[CompressionAnalysis],
    tension: Optional[TensionIndex],
    temporal_map: Optional[TemporalMap],
    pre_patterns: list[PreSpikePattern],
    base_prob_800: float,
    base_prob_1500: float,
    crash_density: float,
) -> int:
    score = 0

    if temporal_map:
        dm10 = temporal_map.levels.get(10.0)
        dm20 = temporal_map.levels.get(20.0)
        if dm10 and dm10.overdue:      score += 25
        if dm10 and dm10.very_overdue: score += 20
        if dm20 and dm20.overdue:      score += 15
        if dm20 and dm20.very_overdue: score += 10

    if compression and compression.compression_score >= 0.50:
        score += int(compression.compression_score * 30)

    if tension:
        if tension.is_extreme: score += 15
        elif tension.is_high:  score += 8

    for pp in pre_patterns:
        if pp.level in [10.0, 20.0] and pp.current_matches:
            score += 20

    score += int(base_prob_800 * 0.4 + base_prob_1500 * 0.6)

    if crash_density >= 0.35:
        score -= int(crash_density * 30)

    return max(0, min(100, score))


def _best_hunter_target(
    hunter_score: int,
    moon_score: int,
    base_prob_500: float,
    base_prob_800: float,
    base_prob_1500: float,
    overdue_levels: list[float],
    compression_score: float = 0.0,
    tension_value: float = 0.0,
) -> tuple[str, float]:
    # MOON 10x — solo con score alto y overdue confirmado
    if moon_score >= MOON_MIN_SCORE:
        if any(l >= 10 for l in overdue_levels) and compression_score >= 0.50:
            return "moon", 10.0

    # HUNTER 8x
    if hunter_score >= HUNTER_MIN_SCORE:
        if 8.0 in overdue_levels or (base_prob_800 >= 10 and hunter_score >= 55):
            return "hunter", 8.0
        return "hunter", 5.0

    return "none", 0.0


def evaluate_hunter_mode(
    history: list[float],
    adv_result: Optional[AdvancedAnalysisResult],
    entropy: EntropyResult,
    binary: BinarySequenceResult,
    fibonacci: FibonacciResult,
    clustering: ClusteringResult,
    base_prob_500: float,
    base_prob_800: float,
    base_prob_1500: float,
    crash_density: float,
) -> HunterModeResult:

    compression = adv_result.compression if adv_result else None
    tension = adv_result.tension if adv_result else None
    temporal_map = adv_result.temporal_map if adv_result else None
    pre_patterns = adv_result.pre_spike_patterns if adv_result else []
    overdue = temporal_map.overdue_levels() if temporal_map else []

    hunter_score = _compute_hunter_score(
        compression, tension, temporal_map, pre_patterns,
        binary, fibonacci, clustering,
        base_prob_500, base_prob_800, crash_density,
    )

    moon_score = _compute_moon_score(
        compression, tension, temporal_map, pre_patterns,
        base_prob_800, base_prob_1500, crash_density,
    )

    reasons_off = []
    if entropy.is_dead_market:
        reasons_off.append("Dead Market activo")
    if adv_result and adv_result.post_spike_blocked:
        reasons_off.append(f"Post-spike cooldown {adv_result.post_spike_cooldown_remaining}r")
    if crash_density >= HUNTER_MAX_CRASH_DENSITY and hunter_score < 70:
        reasons_off.append(f"crash_density={crash_density:.2%} sin suficiente score")
    if clustering.context_label == CONTEXT_CHAOTIC:
        reasons_off.append("Contexto CAÓTICO")
    # Bloqueo por falso impulso en hunter también
    if _detect_false_impulse(history):
        reasons_off.append("Falso impulso activo")

    if reasons_off:
        return HunterModeResult(
            active=False, mode="none", target=0.0,
            hunter_score=hunter_score, moon_score=moon_score, confidence=0,
            reason_on="", reason_off=" | ".join(reasons_off),
            compression_contrib=compression.compression_score if compression else 0.0,
            tension_contrib=tension.value if tension else 0.0,
            temporal_contrib=len(overdue) / max(1, len(SPIKE_LEVELS)),
            overdue_levels=overdue,
            fibonacci_note=fibonacci.note,
            binary_rareness=binary.rareness_score,
            anomaly_score=clustering.anomaly_score,
        )

    mode, target = _best_hunter_target(
        hunter_score, moon_score,
        base_prob_500, base_prob_800, base_prob_1500, overdue,
        compression_score=compression.compression_score if compression else 0.0,
        tension_value=tension.value if tension else 0.0,
    )

    if mode == "none":
        return HunterModeResult(
            active=False, mode="none", target=0.0,
            hunter_score=hunter_score, moon_score=moon_score, confidence=0,
            reason_on="", reason_off=f"Scores insuficientes (hunter={hunter_score}/{HUNTER_MIN_SCORE}, moon={moon_score}/{MOON_MIN_SCORE})",
            compression_contrib=compression.compression_score if compression else 0.0,
            tension_contrib=tension.value if tension else 0.0,
            temporal_contrib=len(overdue) / max(1, len(SPIKE_LEVELS)),
            overdue_levels=overdue,
            fibonacci_note=fibonacci.note,
            binary_rareness=binary.rareness_score,
            anomaly_score=clustering.anomaly_score,
        )

    base_conf = moon_score if mode == "moon" else hunter_score
    conf = int(base_conf * 0.50 + base_prob_500 * 0.30 + base_prob_800 * 0.20)
    conf = int(conf * (1 - crash_density * 0.25))
    conf = max(0, min(100, conf))

    parts = []
    if compression and compression.compression_score >= 0.35:
        parts.append(f"comp={compression.compression_score:.2f}")
    if tension and tension.value >= 30:
        parts.append(f"tensión={tension.value:.0f}")
    if overdue:
        parts.append(f"overdue={[int(l) for l in overdue]}")
    reason_on = " | ".join(parts) if parts else f"mode={mode} score={base_conf}"

    return HunterModeResult(
        active=True, mode=mode, target=target,
        hunter_score=hunter_score, moon_score=moon_score, confidence=conf,
        reason_on=reason_on, reason_off="",
        compression_contrib=compression.compression_score if compression else 0.0,
        tension_contrib=tension.value if tension else 0.0,
        temporal_contrib=len(overdue) / max(1, len(SPIKE_LEVELS)),
        overdue_levels=overdue,
        fibonacci_note=fibonacci.note,
        binary_rareness=binary.rareness_score,
        anomaly_score=clustering.anomaly_score,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Motor dual principal
# ══════════════════════════════════════════════════════════════════════════════

def run_dual_engine(
    history: list[float],
    adv_result: Optional[AdvancedAnalysisResult],
    context_memory: ContextMemory,
    spike_distances: dict[float, list[int]],
    prob_190: float = 0.0,
    prob_200: float = 0.0,
    prob_250: float = 0.0,
    prob_300: float = 0.0,
    prob_500: float = 0.0,
    prob_800: float = 0.0,
    prob_1500: float = 0.0,
    crash_density: float = 0.0,
    base_score: int = 0,
) -> DualEngineResult:

    context_logs: list[str] = []

    # ── 1. Análisis de soporte ─────────────────────────────────────────────────
    entropy  = analyze_entropy(history, window=50)
    binary   = analyze_binary_sequences(history, window=30)

    fib_distances = []
    for level in [5.0, 10.0, 20.0]:
        dists = spike_distances.get(level, [])
        fib_distances.extend(dists[-10:])
    fibonacci = analyze_fibonacci(sorted(fib_distances)[-20:] if fib_distances else [])

    # ── 2. Dead Market ─────────────────────────────────────────────────────────
    dead_market = evaluate_dead_market(entropy, history)

    # ── 3. Clustering ─────────────────────────────────────────────────────────
    comp_score  = adv_result.compression.compression_score if (adv_result and adv_result.compression) else 0.0
    tension_val = adv_result.tension.value if (adv_result and adv_result.tension) else 0.0

    clustering = classify_context(
        history=history,
        memory=context_memory,
        compression_score=comp_score,
        entropy_score=entropy.entropy_score,
        tension_value=tension_val,
    )

    # ── 4. SAFE MODE ──────────────────────────────────────────────────────────
    safe_result = evaluate_safe_mode(
        history=history,
        prob_190=prob_190, prob_200=prob_200,
        prob_250=prob_250, prob_300=prob_300,
        crash_density=crash_density,
        entropy=entropy, adv_result=adv_result,
    )

    # ── 5. HUNTER MODE ────────────────────────────────────────────────────────
    hunter_result = evaluate_hunter_mode(
        history=history, adv_result=adv_result,
        entropy=entropy, binary=binary, fibonacci=fibonacci,
        clustering=clustering,
        base_prob_500=prob_500, base_prob_800=prob_800,
        base_prob_1500=prob_1500, crash_density=crash_density,
    )

    # ── 6. Decisión final ─────────────────────────────────────────────────────
    if dead_market.block_all_signals:
        mode, emit, final_target = "dead", False, 0.0
        final_conf, final_type = 0, "none"
        context_logs.append(f"[DEAD MARKET] {dead_market.reason}")

    # SAFE tiene prioridad sobre HUNTER/MOON cuando ambos están activos
    # (maximiza win rate — la vieja versión lo hacía así)
    elif safe_result.active and hunter_result.active:
        # Si el safe score es alto, preferimos safe
        if safe_result.safe_score >= 60:
            mode, emit  = "safe", True
            final_target = safe_result.target
            final_conf   = safe_result.confidence
            final_type   = "safe"
            context_logs.append(f"[SAFE] target={final_target}x score={safe_result.safe_score} — preferido sobre hunter")
        else:
            mode, emit  = hunter_result.mode, True
            final_target = hunter_result.target
            final_conf   = hunter_result.confidence
            final_type   = "moon" if mode == "moon" else "high_risk"
            context_logs.append(f"[{mode.upper()}] target={final_target}x | {hunter_result.reason_on}")

    elif hunter_result.active:
        mode, emit  = hunter_result.mode, True
        final_target = hunter_result.target
        final_conf   = hunter_result.confidence
        final_type   = "moon" if mode == "moon" else "high_risk"
        context_logs.append(f"[{mode.upper()}] target={final_target}x | {hunter_result.reason_on}")

    elif safe_result.active:
        mode, emit  = "safe", True
        final_target = safe_result.target
        final_conf   = safe_result.confidence
        final_type   = "safe"
        context_logs.append(f"[SAFE] target={final_target}x score={safe_result.safe_score} | {safe_result.reason_on}")

    else:
        mode, emit, final_target = "none", False, 0.0
        final_conf, final_type   = 0, "none"

    # Logs adicionales
    if entropy.label in ("estructurado", "muerto"):
        context_logs.append(f"[ENTROPÍA] {entropy.context_note}")
    if binary.rareness_score >= 0.40:
        context_logs.append(f"[BINARIO] rareness={binary.rareness_score:.3f} | run_0={binary.longest_run_0}")
    if adv_result:
        context_logs.extend(adv_result.context_logs)

    log.info(
        "[DUAL ENGINE] mode=%s emit=%s target=%.2fx safe=%d hunter=%d moon=%d entropy=%.3f",
        mode, emit, final_target,
        safe_result.safe_score, hunter_result.hunter_score, hunter_result.moon_score,
        entropy.entropy_score,
    )

    return DualEngineResult(
        mode=mode,
        safe=safe_result,
        hunter=hunter_result,
        entropy=entropy, binary=binary,
        fibonacci=fibonacci, dead_market=dead_market,
        clustering=clustering,
        safe_score=safe_result.safe_score,
        hunter_score=hunter_result.hunter_score,
        moon_score=hunter_result.moon_score,
        entropy_score_val=entropy.entropy_score,
        tension_score=tension_val,
        compression_score=comp_score,
        emit_signal=emit,
        final_target=final_target,
        final_confidence=final_conf,
        final_signal_type=final_type,
        context_logs=context_logs,
    )
