"""
analyzer/spike_tracker.py
═══════════════════════════════════════════════════════════════════════════════
Tracker en tiempo real de spikes y sus distancias.

Funciones:
  - SpikeTracker.register_round()   → procesa cada ronda nueva
  - SpikeTracker.get_rounds_since() → rondas desde el último spike de cada nivel
  - SpikeTracker.pending_compression_event_id → ID del último evento de compresión
    activo (para actualizar retroactivamente si ocurre un spike).

Este módulo NO hace análisis — solo mantiene el estado y persiste en la DB.
El análisis estadístico lo hace advanced_stats.py.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import Optional

import aiosqlite

from analyzer.advanced_stats import SPIKE_LEVELS, WINDOW_PRE_SPIKE, LOW_THRESHOLD, ULTRA_LOW_THRESHOLD
from database.db_advanced import (
    insert_spike_distance,
    insert_compression_event,
    update_compression_event_spike,
    save_pre_spike_context,
)
from utils.logger import get_logger, log_error

log = get_logger("spike_tracker")


class SpikeTracker:
    """
    Mantiene contadores en memoria de distancia desde el último spike por nivel,
    y persiste en la DB los eventos relevantes.
    """

    def __init__(self) -> None:
        # rounds_since[level] = cuántas rondas han pasado desde el último spike
        self._rounds_since: dict[float, int] = {level: 0 for level in SPIKE_LEVELS}
        # Historial de últimas N rondas (para calcular contexto pre-spike)
        self._recent: list[float] = []
        self._recent_max = 50  # mantener últimas 50 en memoria

        # Tracking de compresión: ID del último evento de compresión sin spike aún
        self._pending_compression_ids: dict[float, int] = {}  # nivel → event_id

        # Contador de rondas total desde que arrancó el tracker
        self._total_rounds: int = 0

        # Última tensión conocida (se actualiza desde main)
        self._last_tension: float = 0.0

    def set_last_tension(self, tension: float) -> None:
        self._last_tension = tension

    def get_rounds_since(self, level: float) -> int:
        return self._rounds_since.get(level, 0)

    async def register_round(
        self,
        val: float,
        ts: datetime,
        conn: aiosqlite.Connection,
    ) -> None:
        """
        Procesa una nueva ronda:
          1. Incrementa contadores de distancia
          2. Si es un spike, guarda en spike_distances y contexto pre-spike
          3. Actualiza pending compression events retroactivamente
        """
        self._total_rounds += 1

        # Incrementar todos los contadores
        for level in SPIKE_LEVELS:
            self._rounds_since[level] += 1

        # Guardar en historial reciente
        self._recent.insert(0, val)
        if len(self._recent) > self._recent_max:
            self._recent.pop()

        # Verificar si este valor es spike para cada nivel
        for level in SPIKE_LEVELS:
            if val >= level:
                distance = self._rounds_since[level]

                # Guardar en spike_distances
                await insert_spike_distance(conn, ts=ts, level=level, val=val, distance=distance)

                # Guardar contexto pre-spike (las N rondas anteriores)
                pre_window = self._recent[1: WINDOW_PRE_SPIKE + 1]  # excluir el spike actual
                if len(pre_window) >= 5:
                    low_dens = sum(1 for v in pre_window if v < LOW_THRESHOLD) / len(pre_window)
                    std_val  = statistics.stdev(pre_window) if len(pre_window) >= 2 else 0.0
                    comp_val = sum(1 for v in pre_window if v < ULTRA_LOW_THRESHOLD) / len(pre_window)

                    await save_pre_spike_context(
                        conn,
                        spike_ts=ts,
                        spike_val=val,
                        spike_level=level,
                        low_density_pre=round(low_dens, 3),
                        std_pre=round(std_val, 3),
                        compression_pre=round(comp_val, 3),
                        tension_pre=self._last_tension,
                        distance_from_prev=distance,
                    )

                # Actualizar compression events pendientes retroactivamente
                if level in [5.0, 10.0, 20.0]:
                    for comp_level, event_id in list(self._pending_compression_ids.items()):
                        if event_id > 0:
                            kwargs = {}
                            if level == 5.0:
                                kwargs["rounds_to_spike5"] = distance
                            elif level == 10.0:
                                kwargs["rounds_to_spike10"] = distance
                            elif level == 20.0:
                                kwargs["rounds_to_spike20"] = distance

                            await update_compression_event_spike(conn, event_id, **kwargs)
                            # Solo limpiar si ya completamos los niveles relevantes
                            if level == 5.0:
                                self._pending_compression_ids.pop(comp_level, None)

                # Resetear contador para este nivel
                self._rounds_since[level] = 0
                log.info(
                    "[SPIKE] %.2fx >= %.0fx | distancia=%d rondas",
                    val, level, distance
                )

    def register_compression_event(self, event_id: int) -> None:
        """Registra el ID de un evento de compresión para actualizar retroactivamente."""
        for level in [5.0, 10.0, 20.0]:
            self._pending_compression_ids[level] = event_id

    def summary(self) -> dict:
        """Resumen del estado actual del tracker."""
        return {
            "total_rounds": self._total_rounds,
            "rounds_since": {f"{int(k)}x": v for k, v in self._rounds_since.items()},
            "recent_values": self._recent[:10],
        }