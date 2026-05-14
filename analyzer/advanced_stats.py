"""
analyzer/advanced_stats.py
═══════════════════════════════════════════════════════════════════════════════
Motor estadístico avanzado para detección de contextos pre-spike en JetX.

Detecta ANTES de que ocurran cuotas altas (5x / 8x / 10x / 20x / 50x / 100x):
  - Distancia temporal entre spikes vs promedios históricos
  - Compresión extrema de rondas bajas consecutivas
  - Tensión estadística acumulada
  - Análisis de patrones previos a explosiones
  - Clustering de multiplicadores altos
  - Cooldown post-spike para evitar falsos entradas

NO usa EMA, momentum inventado, ni análisis tipo Forex.
TODO es estadística descriptiva pura basada en datos históricos reales.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

log = get_logger("advanced_stats")

# ── Umbrales de spikes que nos interesan ──────────────────────────────────────
SPIKE_LEVELS = [5.0, 8.0, 10.0, 20.0, 50.0, 100.0, 200.0]

# Distancias históricas de referencia — calibradas para JetX real
# JetX saca 10x+ unas 20 veces/día → cada ~30 rondas aprox
# Formato: nivel → (promedio, mediana, desviación, mínimo, máximo)
_SPIKE_DISTANCE_STATS: dict[float, tuple[float, float, float, float, float]] = {
    5.0:   (4.0,  3.0,  3.5,  1,  15),   # sale muy seguido
    8.0:   (7.0,  5.0,  6.0,  1,  25),
    10.0:  (12.0, 9.0,  9.0,  1,  45),   # ~20 veces/día
    20.0:  (18.0, 14.0, 15.0, 1,  80),
    50.0:  (30.0, 22.0, 25.0, 2, 130),
    100.0: (55.0, 40.0, 50.0, 3, 250),   # varias por sesión
    200.0: (120.0, 90.0, 90.0, 5, 500),  # menos frecuente pero existe
}

# Compresión: umbral para considerar una ronda "ultra-baja"
ULTRA_LOW_THRESHOLD = 1.30
LOW_THRESHOLD = 1.50

# Ventanas de análisis
WINDOW_COMPRESSION = 10   # rondas para medir compresión reciente
WINDOW_PRE_SPIKE   = 15   # rondas antes de un spike para analizar patrones
WINDOW_TENSION     = 30   # rondas para calcular tensión estadística

# Cooldown post-spike reducido — JetX puede sacar 2 spikes seguidos
POST_SPIKE_COOLDOWN: dict[float, int] = {
    5.0:  2,
    8.0:  2,
    10.0: 3,
    20.0: 4,
    50.0: 5,
}


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses de resultado
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpikeDistanceMap:
    """Distancia actual desde el último spike de cada nivel vs. histórico."""
    level: float
    rounds_since: int           # rondas desde el último spike de este nivel
    avg_distance: float         # promedio histórico entre spikes
    median_distance: float
    std_distance: float
    min_distance: float
    max_distance: float
    z_score: float              # cuántas desv. estándar sobre la media
    overdue: bool               # True si ya superó el promedio + 1σ
    very_overdue: bool          # True si superó media + 2σ

    @property
    def pressure(self) -> float:
        """0.0 a 1.0: presión acumulada por ausencia de este spike."""
        if self.std_distance == 0:
            return 0.0
        return min(1.0, max(0.0, (self.rounds_since - self.avg_distance) / (self.std_distance * 2)))


@dataclass
class CompressionAnalysis:
    """Análisis de compresión extrema en la ventana reciente."""
    window_size: int
    ultra_low_count: int        # rondas < 1.30x
    low_count: int              # rondas < 1.50x
    ultra_low_pct: float        # % de rondas ultra-bajas
    low_pct: float              # % de rondas bajas
    consecutive_ultra_low: int  # racha máxima consecutiva ultra-bajas
    consecutive_low: int        # racha máxima consecutiva bajas
    compression_score: float    # 0.0 a 1.0
    label: str                  # descripción textual


@dataclass
class PreSpikePattern:
    """Patrones estadísticos detectados en ventanas previas a spikes históricos."""
    level: float
    avg_low_density_before: float   # densidad media de crashes en 15r previas
    avg_std_before: float           # desviación media en 15r previas
    avg_compression_before: float   # compresión media en 15r previas
    samples: int                    # cuántos spikes analizados
    current_matches: bool           # el contexto actual coincide con el patrón


@dataclass
class TemporalMap:
    """Mapa temporal completo de todos los niveles de spike."""
    levels: dict[float, SpikeDistanceMap] = field(default_factory=dict)

    def highest_pressure_level(self) -> Optional[float]:
        if not self.levels:
            return None
        return max(self.levels, key=lambda l: self.levels[l].pressure)

    def overdue_levels(self) -> list[float]:
        return [l for l, d in self.levels.items() if d.overdue]

    def very_overdue_levels(self) -> list[float]:
        return [l for l, d in self.levels.items() if d.very_overdue]


@dataclass
class TensionIndex:
    """
    Índice de tensión estadística global (0–100).
    NO predice — solo mide cuán inusual es el contexto actual.
    """
    value: float                # 0–100
    compression_contribution: float
    temporal_contribution: float
    volatility_contribution: float
    label: str                  # "baja" / "media" / "alta" / "extrema"
    context_tags: list[str] = field(default_factory=list)

    @property
    def is_high(self) -> bool:
        return self.value >= 60

    @property
    def is_extreme(self) -> bool:
        return self.value >= 80


@dataclass
class AdvancedScores:
    """
    Scores separados para cada clase de señal.
    NO mezcla señales conservadoras con agresivas.
    """
    safe_score: int         # orientado a 2x–3x
    high_risk_score: int    # orientado a 5x–8x
    moon_score: int         # orientado a 10x–100x

    safe_confidence: int
    high_risk_confidence: int
    moon_confidence: int

    safe_target: float
    high_risk_target: float
    moon_target: float


@dataclass
class AdvancedAnalysisResult:
    """Resultado completo del análisis avanzado."""
    # Bloqueos
    post_spike_blocked: bool = False
    post_spike_cooldown_remaining: int = 0
    blocking_spike_level: float = 0.0

    # Componentes de análisis
    compression: Optional[CompressionAnalysis] = None
    temporal_map: Optional[TemporalMap] = None
    tension: Optional[TensionIndex] = None
    pre_spike_patterns: list[PreSpikePattern] = field(default_factory=list)

    # Scores avanzados
    scores: Optional[AdvancedScores] = None

    # Señal recomendada
    recommended_signal: str = "none"  # "safe" / "high_risk" / "moon" / "none"
    recommended_target: float = 0.0
    recommended_score: int = 0
    recommended_confidence: int = 0

    # Logs descriptivos
    context_logs: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Post-spike cooldown tracker (por nivel)
# ══════════════════════════════════════════════════════════════════════════════

class MultiLevelSpikeBlocker:
    """
    Bloquea señales después de spikes en cada nivel de forma independiente.
    No perseguir spikes: no entrar inmediatamente después de 5x, 10x, 20x.
    """

    def __init__(self):
        self._cooldowns: dict[float, int] = {}  # nivel → rondas bloqueadas restantes

    def register_round(self, val: float) -> None:
        # Activar cooldown para todos los niveles que este val supera
        for level, cooldown in POST_SPIKE_COOLDOWN.items():
            if val >= level:
                current = self._cooldowns.get(level, 0)
                new_cd = max(current, cooldown)
                self._cooldowns[level] = new_cd
                log.info(
                    "[POST_SPIKE_COOLDOWN] %.2fx >= %.0fx → bloqueando %d rondas",
                    val, level, new_cd
                )

        # Decrementar todos los cooldowns activos
        for level in list(self._cooldowns):
            if self._cooldowns[level] > 0:
                self._cooldowns[level] -= 1

    def is_blocked_for(self, level: float) -> bool:
        return self._cooldowns.get(level, 0) > 0

    def cooldown_remaining(self, level: float) -> int:
        return self._cooldowns.get(level, 0)

    def any_high_level_blocked(self) -> tuple[bool, float, int]:
        """Retorna (bloqueado, nivel, rondas_restantes) para el nivel más alto bloqueado."""
        for level in sorted(self._cooldowns.keys(), reverse=True):
            if self._cooldowns[level] > 0:
                return True, level, self._cooldowns[level]
        return False, 0.0, 0


# ══════════════════════════════════════════════════════════════════════════════
# Funciones de análisis
# ══════════════════════════════════════════════════════════════════════════════

def _analyze_compression(history: list[float]) -> CompressionAnalysis:
    """Detecta compresión extrema: demasiadas rondas ultra-bajas consecutivas."""
    window = history[:WINDOW_COMPRESSION]
    n = len(window)

    ultra_low = sum(1 for v in window if v < ULTRA_LOW_THRESHOLD)
    low = sum(1 for v in window if v < LOW_THRESHOLD)

    # Racha máxima consecutiva
    max_consec_ultra = 0
    max_consec_low = 0
    cur_ultra = 0
    cur_low = 0

    for v in window:
        if v < ULTRA_LOW_THRESHOLD:
            cur_ultra += 1
            max_consec_ultra = max(max_consec_ultra, cur_ultra)
        else:
            cur_ultra = 0

        if v < LOW_THRESHOLD:
            cur_low += 1
            max_consec_low = max(max_consec_low, cur_low)
        else:
            cur_low = 0

    ultra_pct = ultra_low / n if n > 0 else 0.0
    low_pct = low / n if n > 0 else 0.0

    # compression_score: 0.0–1.0
    # Componentes: % ultra-bajas (0.5) + rachas consecutivas (0.3) + % bajas (0.2)
    cs_pct    = min(1.0, ultra_pct / 0.7)
    cs_consec = min(1.0, max_consec_ultra / 5)
    cs_low    = min(1.0, low_pct / 0.8)
    compression_score = round(cs_pct * 0.5 + cs_consec * 0.3 + cs_low * 0.2, 3)

    if compression_score >= 0.80:
        label = "COMPRESIÓN EXTREMA"
    elif compression_score >= 0.55:
        label = "COMPRESIÓN ALTA"
    elif compression_score >= 0.30:
        label = "COMPRESIÓN MODERADA"
    else:
        label = "Sin compresión significativa"

    return CompressionAnalysis(
        window_size=n,
        ultra_low_count=ultra_low,
        low_count=low,
        ultra_low_pct=round(ultra_pct, 3),
        low_pct=round(low_pct, 3),
        consecutive_ultra_low=max_consec_ultra,
        consecutive_low=max_consec_low,
        compression_score=compression_score,
        label=label,
    )


def _build_temporal_map(history: list[float]) -> TemporalMap:
    """
    Calcula cuántas rondas han pasado desde el último spike de cada nivel
    y lo compara con el promedio histórico.
    """
    tmap = TemporalMap()

    for level in SPIKE_LEVELS:
        # Buscar posición del último spike de este nivel
        rounds_since = 0
        found = False
        for i, v in enumerate(history):
            if v >= level:
                rounds_since = i
                found = True
                break

        if not found:
            rounds_since = len(history)

        ref = _SPIKE_DISTANCE_STATS.get(level)
        if ref is None:
            continue

        avg, med, std, mn, mx = ref

        z_score = ((rounds_since - avg) / std) if std > 0 else 0.0
        overdue      = rounds_since > (avg + std)
        very_overdue = rounds_since > (avg + 2 * std)

        tmap.levels[level] = SpikeDistanceMap(
            level=level,
            rounds_since=rounds_since,
            avg_distance=avg,
            median_distance=med,
            std_distance=std,
            min_distance=mn,
            max_distance=mx,
            z_score=round(z_score, 2),
            overdue=overdue,
            very_overdue=very_overdue,
        )

    return tmap


def _analyze_pre_spike_patterns(history: list[float]) -> list[PreSpikePattern]:
    """
    Analiza los WINDOW_PRE_SPIKE rondas previas a cada spike histórico
    para encontrar patrones estadísticos reales.
    Solo usa estadísticas descriptivas — no inventa tendencias.
    """
    patterns: list[PreSpikePattern] = []

    for level in [5.0, 10.0, 20.0]:
        # Encontrar todos los spikes de este nivel en el historial
        spike_positions = [i for i, v in enumerate(history) if v >= level]

        if len(spike_positions) < 3:
            continue

        low_densities_before: list[float] = []
        stds_before: list[float] = []
        compressions_before: list[float] = []

        for pos in spike_positions:
            pre_window_end = pos + WINDOW_PRE_SPIKE
            pre_window = history[pos + 1: pre_window_end + 1]  # rondas anteriores (historial invertido)

            if len(pre_window) < 5:
                continue

            low_dens = sum(1 for v in pre_window if v < LOW_THRESHOLD) / len(pre_window)
            low_densities_before.append(low_dens)

            if len(pre_window) >= 2:
                stds_before.append(statistics.stdev(pre_window))

            ultra_low = sum(1 for v in pre_window if v < ULTRA_LOW_THRESHOLD)
            compressions_before.append(ultra_low / len(pre_window))

        if not low_densities_before:
            continue

        avg_ld  = statistics.mean(low_densities_before)
        avg_std = statistics.mean(stds_before) if stds_before else 0.0
        avg_comp = statistics.mean(compressions_before)

        # ¿El contexto actual coincide con el patrón pre-spike?
        current_pre = history[:WINDOW_PRE_SPIKE]
        if len(current_pre) >= 5:
            cur_ld = sum(1 for v in current_pre if v < LOW_THRESHOLD) / len(current_pre)
            cur_std = statistics.stdev(current_pre) if len(current_pre) >= 2 else 0.0
            cur_comp = sum(1 for v in current_pre if v < ULTRA_LOW_THRESHOLD) / len(current_pre)

            # Coincidencia: crash density y compresión similares al patrón histórico
            ld_match   = abs(cur_ld - avg_ld) <= 0.15
            comp_match = abs(cur_comp - avg_comp) <= 0.15
            matches    = ld_match and comp_match
        else:
            matches = False

        patterns.append(PreSpikePattern(
            level=level,
            avg_low_density_before=round(avg_ld, 3),
            avg_std_before=round(avg_std, 3),
            avg_compression_before=round(avg_comp, 3),
            samples=len(low_densities_before),
            current_matches=matches,
        ))

    return patterns


def _compute_tension_index(
    compression: CompressionAnalysis,
    temporal_map: TemporalMap,
    history: list[float],
) -> TensionIndex:
    """
    Índice de tensión estadística 0–100.
    Mide cuán inusual / acumulado es el contexto actual.
    NO predice — solo detecta contextos estadísticamente raros.
    """
    context_tags: list[str] = []

    # Componente 1: Compresión (0–35 puntos)
    comp_contrib = compression.compression_score * 35
    if compression.compression_score >= 0.80:
        context_tags.append("[COMPRESIÓN EXTREMA]")
    elif compression.compression_score >= 0.55:
        context_tags.append("[COMPRESIÓN ALTA]")

    # Componente 2: Presión temporal (0–45 puntos)
    # Suma de presiones por nivel, ponderando niveles más altos
    level_weights = {5.0: 0.10, 8.0: 0.15, 10.0: 0.20, 20.0: 0.25, 50.0: 0.15, 100.0: 0.15}
    temporal_pressure = 0.0
    for level, dm in temporal_map.levels.items():
        w = level_weights.get(level, 0.1)
        temporal_pressure += dm.pressure * w

    temporal_contrib = min(45, temporal_pressure * 45)

    very_overdue = temporal_map.very_overdue_levels()
    overdue = temporal_map.overdue_levels()

    if very_overdue:
        lvls = "/".join(f"{int(l)}x" for l in very_overdue[:3])
        context_tags.append(f"[DISTANCIA {lvls} ELEVADA]")
    elif overdue:
        lvls = "/".join(f"{int(l)}x" for l in overdue[:3])
        context_tags.append(f"[DISTANCIA {lvls} SOBRE MEDIA]")

    # Componente 3: Volatilidad controlada (0–20 puntos)
    recent30 = history[:30]
    std_val = statistics.stdev(recent30) if len(recent30) >= 2 else 0.0
    # Tensión alta cuando hay volatilidad baja (rondas planas, comprimidas)
    if std_val < 0.8:
        vol_contrib = 20.0
        context_tags.append("[VOLATILIDAD MÍNIMA]")
    elif std_val < 2.0:
        vol_contrib = 12.0
    elif std_val < 5.0:
        vol_contrib = 6.0
    else:
        vol_contrib = 0.0

    total = round(comp_contrib + temporal_contrib + vol_contrib, 1)
    total = min(100.0, total)

    if total >= 80:
        label = "TENSIÓN EXTREMA"
        context_tags.append("[TENSIÓN ALTA]")
    elif total >= 60:
        label = "TENSIÓN ALTA"
        context_tags.append("[TENSIÓN ALTA]")
    elif total >= 35:
        label = "TENSIÓN MEDIA"
    else:
        label = "TENSIÓN BAJA"

    return TensionIndex(
        value=total,
        compression_contribution=round(comp_contrib, 1),
        temporal_contribution=round(temporal_contrib, 1),
        volatility_contribution=round(vol_contrib, 1),
        label=label,
        context_tags=context_tags,
    )


def _compute_advanced_scores(
    compression: CompressionAnalysis,
    temporal_map: TemporalMap,
    tension: TensionIndex,
    pre_patterns: list[PreSpikePattern],
    history: list[float],
    base_prob_200: float,
    base_prob_300: float,
    base_prob_500: float,
    base_prob_800: float,
    base_prob_1500: float,
    crash_density: float,
    base_score: int,
) -> AdvancedScores:
    """
    Calcula tres scores independientes.
    safe_score    → contexto para 2x–3x
    high_risk_score → contexto para 5x–8x
    moon_score    → contexto para 10x+
    """

    # ── SAFE SCORE (2x–3x) ────────────────────────────────────────────────────
    # Se basa principalmente en el score base + ajuste por tensión
    safe = base_score
    # Compresión alta indica posible expansión pero también riesgo
    if compression.compression_score > 0.6:
        safe = int(safe * 0.9)  # riesgo añadido
    if tension.value > 70:
        safe += 5   # contexto inusual puede favorecer salida
    safe = max(0, min(100, safe))

    safe_conf = int(max(0, min(100, safe * 0.65 + base_prob_200 * 0.35)))
    safe_conf = int(safe_conf * (1 - crash_density * 0.4))
    safe_target = 3.0 if base_prob_300 >= 22 and safe >= 68 else 2.0

    # ── HIGH RISK SCORE (5x–8x) ───────────────────────────────────────────────
    # Se basa en: tensión + distancia temporal 5x/8x + patrón pre-spike
    hr = 0

    # Contribución temporal: ¿estamos "atrasados" en 5x/8x?
    dm5 = temporal_map.levels.get(5.0)
    dm8 = temporal_map.levels.get(8.0)
    if dm5 and dm5.overdue:
        hr += 25
    if dm5 and dm5.very_overdue:
        hr += 15
    if dm8 and dm8.overdue:
        hr += 20
    if dm8 and dm8.very_overdue:
        hr += 10

    # Contribución de compresión
    hr += int(compression.compression_score * 20)

    # Contribución de tensión
    hr += int(tension.value * 0.20)

    # Patrón pre-spike confirmado
    for pp in pre_patterns:
        if pp.level in [5.0, 8.0] and pp.current_matches:
            hr += 15

    # Probabilidad base
    hr += int(base_prob_500 * 0.3)

    hr = max(0, min(100, hr))
    hr_conf = int(max(0, min(100, hr * 0.55 + base_prob_500 * 0.45)))
    hr_conf = int(hr_conf * (1 - crash_density * 0.3))
    hr_target = 8.0 if base_prob_800 >= 8 and hr >= 70 else 5.0

    # ── MOON SCORE (10x+) ─────────────────────────────────────────────────────
    ms = 0

    dm10 = temporal_map.levels.get(10.0)
    dm20 = temporal_map.levels.get(20.0)
    dm50 = temporal_map.levels.get(50.0)

    if dm10 and dm10.overdue:
        ms += 20
    if dm10 and dm10.very_overdue:
        ms += 15
    if dm20 and dm20.overdue:
        ms += 20
    if dm20 and dm20.very_overdue:
        ms += 15
    if dm50 and dm50.overdue:
        ms += 10

    # Compresión extrema previa a grandes movimientos
    if compression.compression_score >= 0.6:
        ms += int(compression.compression_score * 25)

    # Tensión extrema
    if tension.is_extreme:
        ms += 15
    elif tension.is_high:
        ms += 8

    # Patrón pre-spike para niveles altos
    for pp in pre_patterns:
        if pp.level in [10.0, 20.0] and pp.current_matches:
            ms += 20

    # Probabilidad base
    ms += int(base_prob_800 * 0.4 + base_prob_1500 * 0.6)

    ms = max(0, min(100, ms))
    ms_conf = int(max(0, min(100, ms * 0.45 + base_prob_1500 * 0.55)))
    ms_conf = int(ms_conf * (1 - crash_density * 0.25))

    # Target moon: el más alto justificado
    if ms >= 70 and base_prob_1500 >= 8:
        ms_target = 15.0
    elif ms >= 60 and base_prob_800 >= 10:
        ms_target = 10.0
    else:
        ms_target = 10.0  # mínimo si entra en moon

    return AdvancedScores(
        safe_score=safe,
        high_risk_score=hr,
        moon_score=ms,
        safe_confidence=safe_conf,
        high_risk_confidence=hr_conf,
        moon_confidence=ms_conf,
        safe_target=safe_target,
        high_risk_target=hr_target,
        moon_target=ms_target,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Función principal de análisis avanzado
# ══════════════════════════════════════════════════════════════════════════════

def analyze_advanced(
    history: list[float],
    blocker: MultiLevelSpikeBlocker,
    base_prob_200: float = 0.0,
    base_prob_300: float = 0.0,
    base_prob_500: float = 0.0,
    base_prob_800: float = 0.0,
    base_prob_1500: float = 0.0,
    crash_density: float = 0.0,
    base_score: int = 0,
) -> AdvancedAnalysisResult:
    """
    Análisis estadístico avanzado orientado a detección de contextos pre-spike.

    Args:
        history:        Lista de valores de rondas, más reciente primero.
        blocker:        MultiLevelSpikeBlocker con estado actual.
        base_prob_*:    Probabilidades calculadas por el motor base.
        crash_density:  Densidad de crashes reciente del motor base.
        base_score:     Score calculado por el motor base.

    Returns:
        AdvancedAnalysisResult con todos los componentes.
    """
    result = AdvancedAnalysisResult()
    context_logs: list[str] = []

    if len(history) < 30:
        return result

    # ── 1. Verificar bloqueo post-spike ───────────────────────────────────────
    blocked, blocking_level, cd_remaining = blocker.any_high_level_blocked()
    if blocked:
        result.post_spike_blocked = True
        result.post_spike_cooldown_remaining = cd_remaining
        result.blocking_spike_level = blocking_level
        context_logs.append(
            f"[POST_SPIKE_COOLDOWN] Bloqueado {cd_remaining} rondas tras {blocking_level:.0f}x"
        )

    # ── 2. Análisis de compresión ──────────────────────────────────────────────
    compression = _analyze_compression(history)
    result.compression = compression

    if compression.compression_score >= 0.55:
        context_logs.append(f"[{compression.label}] score={compression.compression_score:.2f} "
                            f"ultra_low={compression.ultra_low_count}/{compression.window_size} "
                            f"racha_max={compression.consecutive_ultra_low}")
        log.info("[COMPRESIÓN] %s | score=%.2f | ultra_low=%d/%d | racha=%d",
                 compression.label, compression.compression_score,
                 compression.ultra_low_count, compression.window_size,
                 compression.consecutive_ultra_low)

    # ── 3. Mapa temporal ───────────────────────────────────────────────────────
    tmap = _build_temporal_map(history)
    result.temporal_map = tmap

    for level, dm in tmap.levels.items():
        if dm.very_overdue:
            msg = (f"[DISTANCIA {int(level)}x ELEVADA] "
                   f"hace={dm.rounds_since}r | avg={dm.avg_distance:.0f}r | "
                   f"z={dm.z_score:.1f}σ | presión={dm.pressure:.2f}")
            context_logs.append(msg)
            log.info(msg)
        elif dm.overdue:
            msg = (f"[DISTANCIA {int(level)}x SOBRE MEDIA] "
                   f"hace={dm.rounds_since}r | avg={dm.avg_distance:.0f}r")
            context_logs.append(msg)

    # ── 4. Patrones pre-spike ──────────────────────────────────────────────────
    pre_patterns = _analyze_pre_spike_patterns(history)
    result.pre_spike_patterns = pre_patterns

    for pp in pre_patterns:
        if pp.current_matches:
            msg = (f"[PRE-SPIKE DETECTADO] Contexto similar a previo a {int(pp.level)}x "
                   f"(n={pp.samples} muestras | avg_low={pp.avg_low_density_before:.2%})")
            context_logs.append(msg)
            log.info(msg)

    # ── 5. Índice de tensión ───────────────────────────────────────────────────
    tension = _compute_tension_index(compression, tmap, history)
    result.tension = tension

    if tension.is_high:
        msg = f"[TENSIÓN ALTA] tension_index={tension.value:.1f} | {tension.label}"
        context_logs.append(msg)
        log.info(msg)
        context_logs.extend(tension.context_tags)

    # ── 6. Scores avanzados ────────────────────────────────────────────────────
    scores = _compute_advanced_scores(
        compression=compression,
        temporal_map=tmap,
        tension=tension,
        pre_patterns=pre_patterns,
        history=history,
        base_prob_200=base_prob_200,
        base_prob_300=base_prob_300,
        base_prob_500=base_prob_500,
        base_prob_800=base_prob_800,
        base_prob_1500=base_prob_1500,
        crash_density=crash_density,
        base_score=base_score,
    )
    result.scores = scores

    # ── 7. Selección de señal recomendada (si no hay bloqueo) ──────────────────
    if not blocked:
        # Moon: requiere score alto Y tensión alta Y sin crash reciente
        if scores.moon_score >= 65 and tension.value >= 55 and crash_density < 0.35:
            result.recommended_signal = "moon"
            result.recommended_target = scores.moon_target
            result.recommended_score = scores.moon_score
            result.recommended_confidence = scores.moon_confidence
            context_logs.append(f"[CONTEXTO MOON] moon_score={scores.moon_score} target={scores.moon_target}x")
            log.info("[CONTEXTO MOON] score=%d target=%.1fx conf=%d%%",
                     scores.moon_score, scores.moon_target, scores.moon_confidence)

        # High risk: score alto en nivel 5x/8x
        elif scores.high_risk_score >= 60 and crash_density < 0.45:
            result.recommended_signal = "high_risk"
            result.recommended_target = scores.high_risk_target
            result.recommended_score = scores.high_risk_score
            result.recommended_confidence = scores.high_risk_confidence
            context_logs.append(f"[CONTEXTO HIGH RISK] hr_score={scores.high_risk_score} target={scores.high_risk_target}x")

        # Safe: score base suficiente
        elif scores.safe_score >= 55:
            result.recommended_signal = "safe"
            result.recommended_target = scores.safe_target
            result.recommended_score = scores.safe_score
            result.recommended_confidence = scores.safe_confidence

    result.context_logs = context_logs
    return result
