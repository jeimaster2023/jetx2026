"""
analyzer/entropy.py
═══════════════════════════════════════════════════════════════════════════════
Módulo de entropía, secuencias binarias y análisis Fibonacci experimental.

Mide:
  - entropy_score       → caos / estructura / aleatoriedad
  - binary sequences    → patrones H/L, bloques repetitivos
  - fibonacci analysis  → coincidencias matemáticas entre distancias de spikes
  - dead_market_mode    → bloqueo total si el mercado es degenerado

NO inventa patrones. Solo mide coincidencias y describe matemáticamente.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

log = get_logger("entropy")

# ── Constantes ────────────────────────────────────────────────────────────────
SPIKE_THRESHOLD_BINARY  = 2.0   # 1 = spike, 0 = low en secuencia binaria
LOW_THRESHOLD_BINARY    = 1.50  # por debajo = crash

ENTROPY_DEAD_MARKET     = 0.92  # entropía > 0.92 → mercado muerto
ENTROPY_STRUCTURED      = 0.55  # entropía < 0.55 → estructura detectada

FIB_SEQUENCE = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
FIB_TOLERANCE = 0.18  # ±18% para considerar coincidencia Fibonacci


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EntropyResult:
    """Resultado del análisis de entropía."""
    entropy_score: float        # 0.0–1.0 (Shannon normalizada)
    label: str                  # "estructurado" / "normal" / "caótico" / "muerto"
    is_structured: bool         # entropía baja → posible compresión
    is_dead_market: bool        # entropía extrema → bloquear todo
    compression_ratio: float    # LZ-like compresión (0–1)
    run_length_score: float     # score basado en run-length encoding
    context_note: str           # descripción matemática del estado


@dataclass
class BinarySequenceResult:
    """Análisis de secuencias binarias H/L."""
    sequence: list[int]         # 1=spike, 0=low, últimas N rondas
    block_size: int             # tamaño de bloque analizado
    # Bloques repetidos
    repeated_blocks: list[tuple[tuple[int, ...], int]]  # (bloque, frecuencia)
    max_repetition: int         # frecuencia del bloque más repetido
    alternation_score: float    # qué tan alternante es la secuencia (0–1)
    run_lengths: list[int]      # longitudes de rachas
    longest_run_0: int          # racha más larga de ceros (crashes)
    longest_run_1: int          # racha más larga de unos (spikes)
    density_1: float            # densidad de spikes en la ventana
    density_0: float            # densidad de crashes en la ventana
    # Micro-ciclos detectados
    micro_cycles: list[str]     # descripciones de micro-ciclos
    rareness_score: float       # qué tan rara es la secuencia actual (0–1)


@dataclass
class FibonacciResult:
    """Análisis Fibonacci experimental de distancias entre spikes."""
    distances: list[int]                        # distancias brutas
    fib_coincidences: list[tuple[int, int]]     # (distancia, número_fib_cercano)
    coincidence_ratio: float                    # % de distancias con coincidencia fib
    golden_ratio_pairs: list[tuple[int, int]]   # pares con relación ~1.618
    expansion_detected: bool                    # patrón expansión detectado
    compression_detected: bool                  # patrón compresión detectado
    note: str                                   # descripción matemática


@dataclass
class DeadMarketResult:
    """Estado del Dead Market Mode."""
    is_dead: bool
    reason: str
    entropy_score: float
    toxicity_score: float       # 0–1 combinado
    block_all_signals: bool


# ══════════════════════════════════════════════════════════════════════════════
# Entropía de Shannon
# ══════════════════════════════════════════════════════════════════════════════

def _shannon_entropy(values: list[float], bins: int = 10) -> float:
    """
    Calcula la entropía de Shannon normalizada (0–1) sobre un histograma
    de `bins` cubetas para la distribución de valores.
    """
    if len(values) < 4:
        return 0.5

    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return 0.0

    step = (max_v - min_v) / bins
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - min_v) / step))
        counts[idx] += 1

    n = len(values)
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log2(p)

    max_entropy = math.log2(bins)
    return round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0


def _compression_ratio(values: list[float], threshold: float = 0.05) -> float:
    """
    Estimación de compresibilidad: qué fracción de valores consecutivos
    están a menos de `threshold` entre sí (run-length compressible).
    """
    if len(values) < 2:
        return 0.0
    similar = sum(1 for i in range(1, len(values)) if abs(values[i] - values[i-1]) < threshold)
    return round(similar / (len(values) - 1), 4)


def _run_length_score(binary_seq: list[int]) -> float:
    """
    Score basado en run-length encoding: rachas largas = más estructura.
    Retorna valor 0–1, donde 1 = máxima estructura (todo igual).
    """
    if len(binary_seq) < 2:
        return 0.5

    runs = 1
    for i in range(1, len(binary_seq)):
        if binary_seq[i] != binary_seq[i-1]:
            runs += 1

    # Pocas transiciones = alta estructura
    max_runs = len(binary_seq)
    return round(1.0 - (runs / max_runs), 4)


def analyze_entropy(history: list[float], window: int = 50) -> EntropyResult:
    """
    Calcula el score de entropía para la ventana más reciente.
    """
    work = history[:window]
    if len(work) < 10:
        return EntropyResult(
            entropy_score=0.5, label="insuficiente",
            is_structured=False, is_dead_market=False,
            compression_ratio=0.0, run_length_score=0.0,
            context_note="Datos insuficientes para calcular entropía."
        )

    entropy = _shannon_entropy(work)
    comp_r  = _compression_ratio(work)

    # Secuencia binaria simple para run-length
    binary = [1 if v >= SPIKE_THRESHOLD_BINARY else 0 for v in work]
    rl_score = _run_length_score(binary)

    # Dead market: entropía extrema O compresión falsa (todo igual)
    is_dead = entropy >= ENTROPY_DEAD_MARKET
    is_structured = entropy <= ENTROPY_STRUCTURED and rl_score >= 0.4

    if is_dead:
        label = "muerto"
        note = (f"Entropía extrema ({entropy:.3f}) — secuencias degeneradas. "
                f"Mercado sin estructura. TODAS las señales bloqueadas.")
    elif entropy >= 0.82:
        label = "caótico"
        note = (f"Alta aleatoriedad (entropy={entropy:.3f}). "
                f"Contexto volátil sin patrones claros.")
    elif is_structured:
        label = "estructurado"
        note = (f"Entropía baja ({entropy:.3f}) con run-length={rl_score:.3f}. "
                f"Posible compresión estadística activa.")
    else:
        label = "normal"
        note = (f"Entropía dentro de rango normal ({entropy:.3f}). "
                f"Sin anomalías estructurales detectadas.")

    log.debug("[ENTROPÍA] score=%.3f label=%s structured=%s dead=%s",
              entropy, label, is_structured, is_dead)

    return EntropyResult(
        entropy_score=entropy,
        label=label,
        is_structured=is_structured,
        is_dead_market=is_dead,
        compression_ratio=comp_r,
        run_length_score=rl_score,
        context_note=note,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Secuencias binarias
# ══════════════════════════════════════════════════════════════════════════════

def _to_binary(history: list[float], spike_thresh: float = SPIKE_THRESHOLD_BINARY) -> list[int]:
    return [1 if v >= spike_thresh else 0 for v in history]


def _find_repeated_blocks(seq: list[int], block_size: int) -> list[tuple[tuple[int, ...], int]]:
    """Busca bloques de tamaño `block_size` que se repiten en la secuencia."""
    from collections import Counter
    if len(seq) < block_size * 2:
        return []
    blocks = [tuple(seq[i:i+block_size]) for i in range(len(seq) - block_size + 1)]
    counter = Counter(blocks)
    repeated = [(blk, cnt) for blk, cnt in counter.items() if cnt >= 2]
    return sorted(repeated, key=lambda x: -x[1])[:5]


def _get_run_lengths(seq: list[int]) -> tuple[list[int], int, int]:
    """
    Calcula rachas y retorna (run_lengths, longest_0, longest_1).
    """
    if not seq:
        return [], 0, 0

    runs = []
    cur_val = seq[0]
    cur_len = 1
    for v in seq[1:]:
        if v == cur_val:
            cur_len += 1
        else:
            runs.append((cur_val, cur_len))
            cur_val = v
            cur_len = 1
    runs.append((cur_val, cur_len))

    run_lengths = [r[1] for r in runs]
    longest_0 = max((r[1] for r in runs if r[0] == 0), default=0)
    longest_1 = max((r[1] for r in runs if r[0] == 1), default=0)
    return run_lengths, longest_0, longest_1


def _alternation_score(seq: list[int]) -> float:
    """Qué tan alternante es: 0,1,0,1 = máx 1.0. 0,0,0,1,1,1 = mín."""
    if len(seq) < 2:
        return 0.0
    transitions = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
    max_transitions = len(seq) - 1
    return round(transitions / max_transitions, 4)


def _detect_micro_cycles(run_lengths: list[int]) -> list[str]:
    """Detecta micro-ciclos en las rachas (patrones rítmicos)."""
    cycles = []
    if len(run_lengths) < 4:
        return cycles

    # Buscar repetición de longitudes similares
    for i in range(len(run_lengths) - 3):
        window = run_lengths[i:i+4]
        if all(abs(window[j] - window[j-1]) <= 1 for j in range(1, 4)):
            cycles.append(f"micro-ciclo estable ~{window[0]}r en posición {i}")
        # Patrón doble: A,B,A,B
        if (abs(window[0] - window[2]) <= 1 and
                abs(window[1] - window[3]) <= 1 and
                window[0] != window[1]):
            cycles.append(f"alternancia {window[0]}-{window[1]} en posición {i}")

    return list(set(cycles))[:3]  # máx 3 únicos


def _binary_rareness(seq: list[int], longest_0: int, density_0: float) -> float:
    """
    Score de rareza de la secuencia binaria actual.
    Racha larga de ceros + densidad alta de crashes = rareza alta.
    """
    rareness = 0.0
    # Racha larga de crashes
    if longest_0 >= 5:
        rareness += min(0.4, longest_0 * 0.07)
    # Alta densidad de crashes
    if density_0 >= 0.6:
        rareness += min(0.3, (density_0 - 0.6) * 0.75)
    # Secuencia poco común: todos ceros
    if density_0 >= 0.8:
        rareness += 0.3
    return min(1.0, round(rareness, 4))


def analyze_binary_sequences(
    history: list[float],
    window: int = 30,
    block_size: int = 4,
) -> BinarySequenceResult:
    """
    Analiza la secuencia binaria de las últimas `window` rondas.
    1 = spike (>= 2x), 0 = crash/low
    """
    work = history[:window]
    binary = _to_binary(work)

    repeated = _find_repeated_blocks(binary, block_size)
    max_rep = max((r[1] for r in repeated), default=0)
    alt_score = _alternation_score(binary)
    run_lengths, longest_0, longest_1 = _get_run_lengths(binary)
    micro_cycles = _detect_micro_cycles(run_lengths)

    n = len(binary)
    density_1 = round(sum(binary) / n, 4) if n > 0 else 0.0
    density_0 = round(1.0 - density_1, 4)
    rareness = _binary_rareness(binary, longest_0, density_0)

    return BinarySequenceResult(
        sequence=binary,
        block_size=block_size,
        repeated_blocks=repeated,
        max_repetition=max_rep,
        alternation_score=alt_score,
        run_lengths=run_lengths,
        longest_run_0=longest_0,
        longest_run_1=longest_1,
        density_1=density_1,
        density_0=density_0,
        micro_cycles=micro_cycles,
        rareness_score=rareness,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Fibonacci experimental
# ══════════════════════════════════════════════════════════════════════════════

def _nearest_fib(n: int) -> int:
    """Retorna el número de Fibonacci más cercano a n."""
    best = FIB_SEQUENCE[0]
    best_diff = abs(n - best)
    for f in FIB_SEQUENCE:
        d = abs(n - f)
        if d < best_diff:
            best_diff = d
            best = f
    return best


def _is_fib_coincidence(n: int, tolerance: float = FIB_TOLERANCE) -> tuple[bool, int]:
    """Retorna (es_coincidencia, fib_cercano)."""
    if n <= 0:
        return False, 0
    nearest = _nearest_fib(n)
    if nearest == 0:
        return False, 0
    ratio = abs(n - nearest) / nearest
    return ratio <= tolerance, nearest


def _detect_golden_ratio_pairs(distances: list[int]) -> list[tuple[int, int]]:
    """
    Detecta pares consecutivos cuyo ratio se acerca al número áureo (1.618).
    """
    phi = 1.618
    tol = 0.15
    pairs = []
    for i in range(len(distances) - 1):
        a, b = distances[i], distances[i+1]
        if a == 0 or b == 0:
            continue
        ratio = b / a
        if abs(ratio - phi) <= tol or abs(ratio - 1/phi) <= tol:
            pairs.append((a, b))
    return pairs[:5]


def analyze_fibonacci(distances: list[int]) -> FibonacciResult:
    """
    Análisis Fibonacci experimental sobre distancias entre spikes.
    Solo mide coincidencias matemáticas — no predice ni inventa patrones.
    """
    if len(distances) < 3:
        return FibonacciResult(
            distances=distances,
            fib_coincidences=[],
            coincidence_ratio=0.0,
            golden_ratio_pairs=[],
            expansion_detected=False,
            compression_detected=False,
            note="Datos insuficientes para análisis Fibonacci.",
        )

    coincidences = []
    for d in distances:
        ok, nearest = _is_fib_coincidence(d)
        if ok:
            coincidences.append((d, nearest))

    coinc_ratio = round(len(coincidences) / len(distances), 4)
    golden_pairs = _detect_golden_ratio_pairs(distances)

    # Detección de expansión: distancias crecientes con ratio ~phi
    expansion = False
    compression = False
    if len(distances) >= 4:
        diffs = [distances[i+1] - distances[i] for i in range(len(distances)-1)]
        # Expansión: distancias crecen consistentemente
        expansion = sum(1 for d in diffs if d > 0) >= len(diffs) * 0.65
        # Compresión: distancias decrecen consistentemente
        compression = sum(1 for d in diffs if d < 0) >= len(diffs) * 0.65

    parts = []
    if coinc_ratio >= 0.40:
        parts.append(f"{coinc_ratio:.0%} de distancias coinciden con secuencia Fibonacci")
    if golden_pairs:
        parts.append(f"{len(golden_pairs)} par(es) con ratio ~φ (1.618) detectado(s)")
    if expansion:
        parts.append("Patrón de expansión temporal detectado (distancias crecientes)")
    if compression:
        parts.append("Patrón de compresión temporal detectado (distancias decrecientes)")
    if not parts:
        parts.append(f"Sin coincidencias Fibonacci significativas (ratio={coinc_ratio:.2f})")

    note = ". ".join(parts) + "."

    return FibonacciResult(
        distances=distances,
        fib_coincidences=coincidences,
        coincidence_ratio=coinc_ratio,
        golden_ratio_pairs=golden_pairs,
        expansion_detected=expansion,
        compression_detected=compression,
        note=note,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Dead Market Mode
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_dead_market(
    entropy: EntropyResult,
    history: list[float],
    recent_spikes_window: int = 5,
    spike_threshold: float = 8.0,
) -> DeadMarketResult:
    """
    Determina si el mercado está en estado degenerado.
    Bloquea TODAS las señales si se cumplen condiciones tóxicas.
    """
    reasons = []
    toxicity = 0.0

    # 1. Entropía extrema
    if entropy.is_dead_market:
        reasons.append(f"Entropía extrema ({entropy.entropy_score:.3f})")
        toxicity += 0.40

    # 2. Spikes demasiado recientes (anti-FOMO)
    recent = history[:recent_spikes_window]
    recent_high = sum(1 for v in recent if v >= spike_threshold)
    if recent_high >= 2:
        reasons.append(f"{recent_high} spikes ≥{spike_threshold:.0f}x en últimas {recent_spikes_window} rondas")
        toxicity += 0.35

    # 3. Volatilidad absurda: std extrema en ventana corta
    if len(history) >= 10:
        try:
            std10 = statistics.stdev(history[:10])
            if std10 >= 15.0:
                reasons.append(f"Volatilidad absurda (std={std10:.1f} en 10r)")
                toxicity += 0.25
        except statistics.StatisticsError:
            pass

    # 4. Secuencia tóxica: crashes extremos consecutivos seguidos de spike enorme
    if len(history) >= 6:
        ultra_low_streak = 0
        for v in history[:5]:
            if v < 1.20:
                ultra_low_streak += 1
            else:
                break
        if ultra_low_streak >= 3 and history[0] >= 20.0:
            reasons.append(f"Secuencia tóxica: {ultra_low_streak} ultra-low seguidos de {history[0]:.1f}x")
            toxicity += 0.30

    # 5. Compresión falsa: historial todo plano
    if entropy.compression_ratio >= 0.80:
        reasons.append(f"Compresión falsa detectada (ratio={entropy.compression_ratio:.3f})")
        toxicity += 0.20

    toxicity = min(1.0, round(toxicity, 4))
    is_dead = toxicity >= 0.50 or entropy.is_dead_market

    reason_str = " | ".join(reasons) if reasons else "Mercado en estado normal"

    if is_dead:
        log.warning("[DEAD MARKET] Bloqueando todas las señales. toxicity=%.2f | %s",
                    toxicity, reason_str)

    return DeadMarketResult(
        is_dead=is_dead,
        reason=reason_str,
        entropy_score=entropy.entropy_score,
        toxicity_score=toxicity,
        block_all_signals=is_dead,
    )
