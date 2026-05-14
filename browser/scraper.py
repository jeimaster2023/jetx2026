"""
browser/scraper.py
Módulo Playwright con:
  - Reinicio automático del navegador cada BROWSER_RESTART_HOURS horas
  - Recuperación automática de frames si el iframe cambia
  - Watchdog interno para detectar congelamiento
  - Limpieza de memoria y cookies periódica
  - Logs detallados de reconexión
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Optional

from playwright.async_api import async_playwright, BrowserContext, Frame, Page, Playwright

from core.config import USER_DATA_DIR, BETPLAY_URL, BROWSER_RESTART_HOURS
from utils.logger import get_logger, log_error

log = get_logger("browser")

# Selector del elemento que contiene los resultados
_SPINS_SELECTOR = "#last100Spins"
_ITEM_SELECTOR = "#last100Spins .row:not(.current)"

# Selector del multiplicador en vuelo (ronda activa de BetPlay)
_CURRENT_SELECTOR = "#last100Spins .row.current"

# Intervalo de polling (segundos)
_POLL_INTERVAL = 1.5
# Intervalo de tick para el multiplicador en vuelo (250ms = tiempo real)
_TICK_INTERVAL = 0.25
# Watchdog: si no hay datos nuevos en N segundos → reconectar
_WATCHDOG_TIMEOUT = 120


class JetXScraper:
    """
    Scraper de JetX sobre BetPlay.
    Uso:
        async with JetXScraper() as scraper:
            async for val in scraper.round_stream():
                process(val)
    """

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._start_time: float = 0.0
        self._last_data_time: float = 0.0
        self._launch_count: int = 0

    # ── Ciclo de vida ──────────────────────────────────────────────────────────

    async def __aenter__(self) -> "JetXScraper":
        await self._launch()
        return self

    async def __aexit__(self, *_) -> None:
        await self._close()

    async def _launch(self) -> None:
        self._launch_count += 1
        log.info("Lanzando navegador (sesión #%d)…", self._launch_count)

        self._playwright = await async_playwright().start()
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-web-security",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
            ignore_default_args=["--enable-automation"],
        )

        self._page = await self._context.new_page()
        await self._page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

        log.info("Navegando a %s — esperando login manual si es necesario.", BETPLAY_URL)
        await self._page.goto(BETPLAY_URL, wait_until="domcontentloaded", timeout=60_000)
        self._start_time = time.monotonic()
        self._last_data_time = time.monotonic()

    async def _close(self) -> None:
        log.info("Cerrando navegador (sesión #%d)…", self._launch_count)
        try:
            if self._context:
                await self._context.close()
        except Exception as e:
            log_error("Error al cerrar context", e)
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            log_error("Error al detener playwright", e)
        self._context = None
        self._page = None
        self._playwright = None

    async def _restart(self) -> None:
        log.warning("Reiniciando navegador completo…")
        await self._close()
        await asyncio.sleep(3)
        await self._launch()
        log.info("Navegador reiniciado exitosamente.")

    # ── Detección de frame ─────────────────────────────────────────────────────

    async def _find_game_frame(self) -> Optional[Frame]:
        """Busca el iframe que contiene el elemento #last100Spins."""
        if not self._page:
            return None
        for frame in self._page.frames:
            try:
                count = await frame.locator(_SPINS_SELECTOR).count()
                if count > 0:
                    return frame
            except Exception:
                continue
        return None

    # ── Lectura de valor ───────────────────────────────────────────────────────

    async def _parse_value(self, raw_text: str) -> Optional[float]:
        """Parsea un texto crudo a float."""
        try:
            clean = (
                raw_text.lower()
                .replace("x", "")
                .replace("\n", "")
                .replace(",", ".")
                .strip()
            )
            return float(clean)
        except (ValueError, AttributeError):
            return None

    async def _read_latest_value(self, frame: Frame) -> Optional[float]:
        """Lee el último valor registrado (el más reciente)."""
        try:
            items = await frame.locator(_ITEM_SELECTOR).all_inner_texts()
            if not items:
                return None
            return await self._parse_value(items[0])
        except Exception as e:
            log_error("Error leyendo frame", e)
            return None

    async def read_live_multiplier(self, frame: Frame) -> Optional[float]:
        """
        Lee el multiplicador de la ronda ACTUALMENTE en vuelo en BetPlay.
        Intenta varios selectores posibles para encontrar el elemento activo.
        Retorna None si no hay ronda activa (fase de espera entre rondas).
        """
        # Selectores candidatos — BetPlay puede usar cualquiera de estos
        candidates = [
            "#last100Spins .row.current",
            "#last100Spins .current",
            "#last100Spins .row.active",
            "#last100Spins .row.flying",
            "#last100Spins .row.live",
            # Fallback: primer .row SIN clase de resultado (aún en vuelo)
            "#last100Spins .row.bet-in-progress",
        ]
        for selector in candidates:
            try:
                locator = frame.locator(selector)
                count = await locator.count()
                if count > 0:
                    raw = await locator.first.inner_text()
                    val = await self._parse_value(raw)
                    if val is not None and 1.0 <= val <= 100_000.0:
                        return val
            except Exception:
                continue
        return None

    async def read_full_history(self, frame: Frame) -> list[float]:
        """
        Lee TODO el historial visible en la columna izquierda.
        Retorna lista ordenada del más reciente al más antiguo.
        Úsalo al arrancar para pre-cargar la DB con datos reales.
        """
        try:
            items = await frame.locator(_ITEM_SELECTOR).all_inner_texts()
            values = []
            for raw in items:
                val = await self._parse_value(raw)
                if val is not None and 1.0 <= val <= 100000.0:
                    values.append(val)
            log.info("Historial inicial leído: %d valores (%.2fx – %.2fx)",
                     len(values),
                     min(values) if values else 0,
                     max(values) if values else 0)
            return values
        except Exception as e:
            log_error("Error leyendo historial completo", e)
            return []

    # ── Stream principal ───────────────────────────────────────────────────────

    async def round_stream(self):
        """
        Generador asíncrono que emite (float, datetime) por cada nueva ronda.
        Al arrancar: emite TODO el historial visible de más antiguo a más reciente.
        Luego: polling normal detectando solo valores nuevos.
        """
        last_val: Optional[float] = None
        history_loaded = False

        while True:
            try:
                # ── Reinicio periódico ─────────────────────────────────────────
                uptime_hours = (time.monotonic() - self._start_time) / 3600
                if uptime_hours >= BROWSER_RESTART_HOURS:
                    log.info("Reinicio programado tras %.1f horas.", uptime_hours)
                    await self._restart()
                    last_val = None
                    history_loaded = False
                    continue

                # ── Watchdog ───────────────────────────────────────────────────
                idle_seconds = time.monotonic() - self._last_data_time
                if idle_seconds > _WATCHDOG_TIMEOUT:
                    log.warning("Watchdog: sin datos por %.0fs. Reiniciando…", idle_seconds)
                    await self._restart()
                    last_val = None
                    history_loaded = False
                    continue

                # ── Buscar frame ───────────────────────────────────────────────
                frame = await self._find_game_frame()
                if frame is None:
                    log.debug("Frame no encontrado — reintentando…")
                    await asyncio.sleep(_POLL_INTERVAL)
                    continue

                # ── Cargar historial completo al arrancar ──────────────────────
                if not history_loaded:
                    full_history = await self.read_full_history(frame)
                    if full_history:
                        # Emitir del más antiguo al más reciente (invertir lista)
                        ts_base = datetime.now()
                        for i, hval in enumerate(reversed(full_history[1:])):  # skip el más reciente
                            yield hval, ts_base
                            await asyncio.sleep(0)  # ceder control sin delay
                        last_val = full_history[0]  # el más reciente ya conocido
                        log.info("Historial inicial cargado: %d valores.", len(full_history))
                    history_loaded = True
                    self._last_data_time = time.monotonic()
                    continue

                # ── Leer valor nuevo ───────────────────────────────────────────
                val = await self._read_latest_value(frame)
                if val is None:
                    await asyncio.sleep(_POLL_INTERVAL)
                    continue

                self._last_data_time = time.monotonic()

                if val != last_val:
                    last_val = val
                    yield val, datetime.now()

                await asyncio.sleep(_POLL_INTERVAL)

                if val != last_val:
                    last_val = val
                    yield val, datetime.now()

                await asyncio.sleep(_POLL_INTERVAL)

            except Exception as e:
                log_error("Error crítico en round_stream — reconectando en 5s", e)
                await asyncio.sleep(5)
                try:
                    await self._restart()
                    last_val = None
                except Exception as e2:
                    log_error("Fallo en restart tras error crítico", e2)
                    await asyncio.sleep(10)

    async def live_tick_stream(self):
        """
        Generador asíncrono que emite el multiplicador EN VUELO cada ~250ms.
        Emite float cuando BetPlay tiene una ronda activa, None cuando está
        en fase de espera entre rondas.

        Úsalo en paralelo con round_stream() para el simulador en tiempo real:
            async for tick_val in scraper.live_tick_stream():
                broadcast_to_ws({'type': 'tick', 'val': tick_val})
        """
        _diag_counter = 0
        _last_logged_val = None
        while True:
            try:
                frame = await self._find_game_frame()
                if frame is None:
                    _diag_counter += 1
                    if _diag_counter % 20 == 1:
                        log.debug("[live_tick] frame no encontrado")
                    await asyncio.sleep(_TICK_INTERVAL)
                    yield None
                    continue

                val = await self.read_live_multiplier(frame)

                # Log de diagnóstico: mostrar en consola cuando hay valor nuevo
                if val is not None and val != _last_logged_val:
                    log.info("[live_tick] ⚡ %.2fx (en vuelo)", val)
                    _last_logged_val = val
                elif val is None and _last_logged_val is not None:
                    log.info("[live_tick] ⏸ entre rondas")
                    _last_logged_val = None

                yield val
                await asyncio.sleep(_TICK_INTERVAL)

            except Exception as e:
                log_error("Error en live_tick_stream", e)
                await asyncio.sleep(_TICK_INTERVAL)
                yield None
