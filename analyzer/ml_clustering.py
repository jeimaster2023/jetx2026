"""
analyzer/ml_clustering.py
═══════════════════════════════════════════════════════════════════════════════
Machine Learning descriptivo para JetX.

NO es IA mágica. Es estadística descriptiva + clustering sin supervisión.

Implementa:
  - KMeans manual (sin sklearn, solo numpy puro o stdlib)
  - DBSCAN simplificado para detección de densidad
  - Isolation Forest numérico para anomalías
  - Clasificación de contextos: safe / neutral / hunter / chaotic

El objetivo es DESCRIBIR el contexto actual, no predecirlo.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

log = get_logger("ml_clustering")

# ── Etiquetas de contexto ─────────────────────────────────────────────────────
CONTEXT_SAFE    = "safe"
CONTEXT_NEUTRAL = "neutral"
CONTEXT_HUNTER  = "hunter"
CONTEXT_CHAOTIC = "chaotic"


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContextFeatures:
    """Vector de características para clasificación de contexto."""
    crash_density_15: float     # densidad crashes < 1.5x en 15r
    crash_density_30: float     # densidad crashes < 1.5x en 30r
    spike_density_30: float     # densidad spikes >= 5x en 30r
    std_15: float               # desviación estándar en 15r
    std_30: float               # desviación estándar en 30r
    mean_15: float              # media en 15r
    max_15: float               # máximo en 15r
    compression_score: float    # score de compresión (0–1)
    entropy_score: float        # score de entropía (0–1)
    tension_value: float        # índice de tensión (0–100)


@dataclass
class ClusterPoint:
    """Punto en el espacio de características."""
    features: list[float]
    label: Optional[str] = None
    cluster_id: int = -1
    anomaly_score: float = 0.0


@dataclass
class ClusteringResult:
    """Resultado del análisis de clustering."""
    context_label: str          # safe / neutral / hunter / chaotic
    context_score: float        # 0–100, qué tan fuerte es la clasificación
    anomaly_score: float        # 0–1, qué tan anómalo es el contexto actual
    is_anomaly: bool            # contexto raro detectado
    cluster_id: int             # cluster asignado (0, 1, 2...)
    similar_contexts: int       # cuántos contextos históricos son similares
    description: str            # descripción matemática del contexto
    feature_vector: list[float] # vector normalizado de características


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades matemáticas (sin numpy)
# ══════════════════════════════════════════════════════════════════════════════

def _euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _normalize_features(features: list[float], mins: list[float], maxs: list[float]) -> list[float]:
    """Normaliza entre 0 y 1 usando min-max."""
    normalized = []
    for v, mn, mx in zip(features, mins, maxs):
        rng = mx - mn
        if rng == 0:
            normalized.append(0.0)
        else:
            normalized.append(max(0.0, min(1.0, (v - mn) / rng)))
    return normalized


def _mean_vector(points: list[list[float]]) -> list[float]:
    if not points:
        return []
    k = len(points[0])
    return [sum(p[i] for p in points) / len(points) for i in range(k)]


# ══════════════════════════════════════════════════════════════════════════════
# KMeans simplificado (stdlib puro)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleKMeans:
    """
    KMeans con k=4 (safe, neutral, hunter, chaotic).
    Solo stdlib — sin numpy ni sklearn.
    """

    def __init__(self, k: int = 4, max_iter: int = 50, seed: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.centroids: list[list[float]] = []
        self.labels: list[int] = []

    def fit(self, data: list[list[float]]) -> "SimpleKMeans":
        if len(data) < self.k:
            self.centroids = data[:]
            self.labels = list(range(len(data)))
            return self

        # Inicializar centroids con KMeans++ simplificado
        rng = random.Random(self.seed)
        centroids = [rng.choice(data)]
        while len(centroids) < self.k:
            dists = [min(_euclidean_distance(p, c) for c in centroids) for p in data]
            total = sum(dists)
            if total == 0:
                centroids.append(rng.choice(data))
                continue
            probs = [d / total for d in dists]
            cumulative = 0.0
            r = rng.random()
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    centroids.append(data[i])
                    break
            else:
                centroids.append(data[-1])

        self.centroids = [c[:] for c in centroids]

        for _ in range(self.max_iter):
            # Asignar clusters
            labels = []
            for p in data:
                dists = [_euclidean_distance(p, c) for c in self.centroids]
                labels.append(dists.index(min(dists)))

            # Actualizar centroids
            new_centroids = []
            for kid in range(self.k):
                cluster_pts = [data[i] for i, l in enumerate(labels) if l == kid]
                if cluster_pts:
                    new_centroids.append(_mean_vector(cluster_pts))
                else:
                    new_centroids.append(self.centroids[kid])

            # Convergencia
            shift = sum(
                _euclidean_distance(self.centroids[i], new_centroids[i])
                for i in range(self.k)
            )
            self.centroids = new_centroids
            self.labels = labels
            if shift < 1e-6:
                break

        return self

    def predict(self, point: list[float]) -> int:
        if not self.centroids:
            return 0
        dists = [_euclidean_distance(point, c) for c in self.centroids]
        return dists.index(min(dists))

    def distance_to_nearest(self, point: list[float]) -> float:
        if not self.centroids:
            return 0.0
        return min(_euclidean_distance(point, c) for c in self.centroids)


# ══════════════════════════════════════════════════════════════════════════════
# Isolation Forest simplificado
# ══════════════════════════════════════════════════════════════════════════════

class SimpleIsolationScorer:
    """
    Isolation Forest simplificado: calcula score de anomalía 0–1.
    Un punto es anómalo si es fácil de aislar (está lejos de todos los demás).
    """

    def __init__(self, n_samples: int = 50):
        self.n_samples = n_samples
        self._data: list[list[float]] = []

    def fit(self, data: list[list[float]]) -> "SimpleIsolationScorer":
        self._data = data[:]
        return self

    def anomaly_score(self, point: list[float]) -> float:
        """
        Score de anomalía 0–1.
        0 = completamente normal, 1 = completamente anómalo.
        """
        if len(self._data) < 5:
            return 0.0

        # Distancia promedio a los 5 vecinos más cercanos
        dists = sorted(_euclidean_distance(point, p) for p in self._data)
        k = min(5, len(dists))
        avg_knn_dist = sum(dists[:k]) / k if k > 0 else 0.0

        # Normalizar respecto a la distancia media global
        all_dists = []
        sample = self._data[:min(self.n_samples, len(self._data))]
        for i, p in enumerate(sample):
            for j, q in enumerate(sample):
                if i < j:
                    all_dists.append(_euclidean_distance(p, q))

        if not all_dists:
            return 0.0

        global_avg = sum(all_dists) / len(all_dists)
        if global_avg == 0:
            return 0.0

        score = avg_knn_dist / (global_avg + 1e-9)
        return min(1.0, round(score, 4))


# ══════════════════════════════════════════════════════════════════════════════
# Extractor de características
# ══════════════════════════════════════════════════════════════════════════════

def _extract_features(
    history: list[float],
    compression_score: float = 0.0,
    entropy_score: float = 0.5,
    tension_value: float = 0.0,
) -> list[float]:
    """
    Extrae vector de 10 características del historial actual.
    Retorna lista normalizable.
    """
    if len(history) < 15:
        return [0.0] * 10

    h15 = history[:15]
    h30 = history[:min(30, len(history))]

    crash_d15 = sum(1 for v in h15 if v < 1.50) / len(h15)
    crash_d30 = sum(1 for v in h30 if v < 1.50) / len(h30)
    spike_d30 = sum(1 for v in h30 if v >= 5.0) / len(h30)

    try:
        std15 = statistics.stdev(h15) if len(h15) >= 2 else 0.0
        std30 = statistics.stdev(h30) if len(h30) >= 2 else 0.0
    except statistics.StatisticsError:
        std15 = std30 = 0.0

    mean15 = sum(h15) / len(h15)
    max15  = max(h15)

    return [
        round(crash_d15, 4),
        round(crash_d30, 4),
        round(spike_d30, 4),
        round(min(std15, 20.0) / 20.0, 4),   # normalizado a 20
        round(min(std30, 20.0) / 20.0, 4),
        round(min(mean15, 10.0) / 10.0, 4),  # normalizado a 10
        round(min(max15, 50.0) / 50.0, 4),   # normalizado a 50
        round(compression_score, 4),
        round(entropy_score, 4),
        round(tension_value / 100.0, 4),      # normalizado a 100
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Clasificador de contexto basado en reglas + clustering
# ══════════════════════════════════════════════════════════════════════════════

def _rule_based_label(features: list[float]) -> tuple[str, float]:
    """
    Clasificación por reglas directas antes de intentar clustering.
    Retorna (label, score 0–100).
    """
    if len(features) < 10:
        return CONTEXT_NEUTRAL, 50.0

    crash_d15 = features[0]
    crash_d30 = features[1]
    spike_d30 = features[2]
    std15_n   = features[3]
    compression = features[7]
    entropy     = features[8]
    tension_n   = features[9]

    score = 0.0

    # CHAOTIC: alta volatilidad + spikes recientes + entropía alta
    if std15_n >= 0.50 and spike_d30 >= 0.15 and entropy >= 0.75:
        score = 60 + std15_n * 40
        return CONTEXT_CHAOTIC, min(100, score)

    # HUNTER: compresión + tensión + crashes sin spikes recientes
    if compression >= 0.50 and tension_n >= 0.50 and spike_d30 <= 0.05:
        score = 55 + compression * 25 + tension_n * 20
        return CONTEXT_HUNTER, min(100, score)

    if crash_d15 >= 0.60 and crash_d30 >= 0.50 and spike_d30 <= 0.03 and entropy <= 0.70:
        score = 55 + crash_d15 * 30
        return CONTEXT_HUNTER, min(100, score)

    # SAFE: baja densidad crashes, baja volatilidad, sin spikes extremos recientes
    if crash_d15 <= 0.20 and crash_d30 <= 0.25 and std15_n <= 0.20 and entropy <= 0.70:
        score = 60 + (1 - crash_d15) * 20 + (1 - std15_n) * 20
        return CONTEXT_SAFE, min(100, score)

    # NEUTRAL: el resto
    score = 40 + (1 - entropy) * 20 + (1 - crash_d15) * 15 + (1 - std15_n) * 15
    return CONTEXT_NEUTRAL, min(100, score)


# ══════════════════════════════════════════════════════════════════════════════
# Memoria de contextos históricos (buffer en memoria)
# ══════════════════════════════════════════════════════════════════════════════

class ContextMemory:
    """
    Buffer circular de los últimos N vectores de características históricos.
    Permite hacer clustering online sin DB.
    """

    def __init__(self, max_size: int = 500):
        self._buffer: list[list[float]] = []
        self._labels: list[str] = []
        self._max_size = max_size
        self._kmeans: Optional[SimpleKMeans] = None
        self._isolation: Optional[SimpleIsolationScorer] = None
        self._fitted: bool = False
        self._fit_every: int = 50  # re-entrenar cada 50 nuevos puntos
        self._since_fit: int = 0

    def add(self, features: list[float], label: str) -> None:
        self._buffer.append(features)
        self._labels.append(label)
        if len(self._buffer) > self._max_size:
            self._buffer.pop(0)
            self._labels.pop(0)
        self._since_fit += 1
        if self._since_fit >= self._fit_every or not self._fitted:
            self._fit()

    def _fit(self) -> None:
        if len(self._buffer) < 20:
            return
        try:
            self._kmeans = SimpleKMeans(k=4).fit(self._buffer)
            self._isolation = SimpleIsolationScorer().fit(self._buffer)
            self._fitted = True
            self._since_fit = 0
            log.debug("[CLUSTERING] Modelos re-entrenados con %d muestras", len(self._buffer))
        except Exception as e:
            log.warning("Fallo al entrenar clustering: %s", e)

    def classify(self, features: list[float]) -> ClusteringResult:
        """Clasifica el contexto actual."""
        label, score = _rule_based_label(features)

        cluster_id = 0
        anomaly_score = 0.0
        similar = 0

        if self._fitted and self._kmeans and self._isolation:
            cluster_id = self._kmeans.predict(features)
            anomaly_score = self._isolation.anomaly_score(features)

            # Contar contextos similares
            dist_thresh = 0.25
            similar = sum(
                1 for hist_feat in self._buffer
                if _euclidean_distance(features, hist_feat) <= dist_thresh
            )

            # Si el punto está muy lejos del centroide, ajustar score
            dist_to_center = self._kmeans.distance_to_nearest(features)
            if dist_to_center >= 0.40:
                anomaly_score = max(anomaly_score, 0.60)

        is_anomaly = anomaly_score >= 0.55

        # Descripción matemática
        desc_parts = [
            f"Contexto: {label.upper()} (score={score:.0f})",
            f"Crash density 15r={features[0]:.2%}" if features else "",
            f"Compresión={features[7]:.3f}" if len(features) > 7 else "",
            f"Entropía={features[8]:.3f}" if len(features) > 8 else "",
            f"Anomalía={anomaly_score:.3f}" if is_anomaly else "",
        ]
        description = " | ".join(p for p in desc_parts if p)

        return ClusteringResult(
            context_label=label,
            context_score=round(score, 1),
            anomaly_score=round(anomaly_score, 4),
            is_anomaly=is_anomaly,
            cluster_id=cluster_id,
            similar_contexts=similar,
            description=description,
            feature_vector=features,
        )

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)


# ══════════════════════════════════════════════════════════════════════════════
# Función principal
# ══════════════════════════════════════════════════════════════════════════════

def classify_context(
    history: list[float],
    memory: ContextMemory,
    compression_score: float = 0.0,
    entropy_score: float = 0.5,
    tension_value: float = 0.0,
) -> ClusteringResult:
    """
    Clasifica el contexto estadístico actual usando clustering descriptivo.

    Args:
        history:          Lista de valores recientes (más reciente primero).
        memory:           Buffer de contextos históricos para comparar.
        compression_score: Score de compresión del motor avanzado (0–1).
        entropy_score:    Score de entropía (0–1).
        tension_value:    Índice de tensión del motor avanzado (0–100).

    Returns:
        ClusteringResult con clasificación y descripción matemática.
    """
    features = _extract_features(history, compression_score, entropy_score, tension_value)
    result = memory.classify(features)

    # Guardar en memoria para aprendizaje continuo
    memory.add(features, result.context_label)

    log.debug(
        "[ML] contexto=%s score=%.0f anomalía=%.3f similar=%d buffer=%d",
        result.context_label, result.context_score,
        result.anomaly_score, result.similar_contexts, memory.buffer_size
    )

    return result
