# JetX Bot v3 — Sistema de Doble Motor Estadístico

Sistema cuantitativo experimental para observación estadística de JetX.
**No predice JetX. Detecta contextos estadísticamente inusuales.**

---

## Arquitectura

```
jetx_bot_v3/
│
├── main_v3.py                    # Orquestador principal v3
│
├── core/
│   └── config.py                 # Variables de entorno y constantes
│
├── analyzer/
│   ├── stats.py                  # Motor estadístico base (original)
│   ├── advanced_stats.py         # Motor avanzado: compresión / tensión / temporal
│   ├── spike_tracker.py          # Tracker de distancias entre spikes
│   │
│   ├── dual_engine.py            # ★ NUEVO: Motor dual SAFE + HUNTER
│   ├── entropy.py                # ★ NUEVO: Entropía / Secuencias / Fibonacci / Dead Market
│   ├── ml_clustering.py          # ★ NUEVO: KMeans + Isolation Forest descriptivo
│   └── ai_analyzer.py            # ★ NUEVO: IA con análisis descriptivo (Groq/fallback)
│
├── database/
│   ├── db.py                     # DB base: rondas, señales, métricas
│   ├── db_advanced.py            # DB avanzada: spikes, compresión, tensión
│   └── db_v2.py                  # ★ NUEVO: Entropía, clustering, dual signals
│
├── browser/
│   └── scraper.py                # Playwright con watchdog y reinicio automático
│
├── tg_bot/
│   ├── notifier.py               # Notificador original
│   └── notifier_v2.py            # ★ NUEVO: Notificador dual con scores separados
│
└── utils/
    └── logger.py                 # 3 logs rotativos (runtime, signals, error)
```

---

## Los dos modos

### 🔵 SAFE MODE
| Parámetro          | Valor                        |
|--------------------|------------------------------|
| Targets            | 1.90x / 2.00x / 2.50x / 3.00x |
| Score mínimo       | 55 / 100                     |
| Crash density máx  | 45%                          |
| Entropía máx       | 0.80                         |
| Señales por sesión | 3–8                          |

**Se activa cuando:**
- Baja volatilidad (std_15 ≤ 2.5)
- Baja densidad de crashes (< 45%)
- Entropía normal (< 0.80)
- Sin tensión extrema (tensión < 70 no bloquea safe, pero sí redirige a hunter)
- Sin spikes recientes en ventana 5r

**Se bloquea cuando:**
- Dead Market activo
- Entropía ≥ 0.80
- Tensión extrema ≥ 70 (contexto más adecuado para hunter)

---

### 🔴 HUNTER MODE / 🌕 MOON MODE
| Parámetro          | Hunter                       | Moon                         |
|--------------------|------------------------------|------------------------------|
| Targets            | 5x / 6x / 7x / 8x           | 10x / 20x / 50x / 100x      |
| Score mínimo       | 60 / 100                     | 65 / 100                     |
| Compresión mínima  | 0.40                         | 0.60                         |
| Tensión mínima     | 35                           | 50                           |
| Señales por sesión | 0–3 (pocas)                  | 0–1 (rarísimas)              |

**Se activa cuando hay ≥2 de:**
- Compresión extrema (score ≥ 0.40)
- Tensión estadística elevada (≥ 35)
- Distancia temporal sobre media (overdue levels)
- Patrón pre-spike confirmado
- Rareza binaria elevada (≥ 0.40)
- Anomalía de clustering detectada
- Fibonacci: compresión/expansión temporal

**NUNCA se activa si:**
- Dead Market activo
- Post-spike cooldown activo
- Contexto clasificado como CAÓTICO
- Crash density ≥ 40% sin score ≥ 70

---

## Scores separados (no se mezclan)

```
safe_score       = f(prob_190, crash_density, std_dev, entropía, consistencia)
hunter_score     = f(compresión, tensión, overdue_levels, pre_patterns, binario, fibonacci)
moon_score       = f(overdue_10x/20x/50x, compresión_extrema, tensión_extrema, pre_patterns)
entropy_score    = Shannon normalizada (0–1)
tension_score    = índice de tensión del motor avanzado (0–100)
compression_score = score de compresión (0–1)
```

---

## Módulos nuevos

### Entropía (`analyzer/entropy.py`)
- **Shannon entropy**: mide aleatoriedad real de la distribución (0–1)
- **Run-length score**: detecta estructura/repetición en la secuencia
- **Dead Market Mode**: bloquea TODO si entropía ≥ 0.92 o toxicidad ≥ 0.50

### Secuencias binarias (`analyzer/entropy.py`)
- Convierte historial en secuencia binaria (1=spike, 0=low)
- Detecta bloques repetidos, micro-ciclos, rachas
- `rareness_score` mide qué tan inusual es la secuencia actual

### Fibonacci experimental (`analyzer/entropy.py`)
- Mide coincidencias entre distancias de spikes y secuencia Fibonacci
- Detecta pares con ratio ≈ φ (1.618)
- Detecta expansión/compresión temporal
- **No inventa patrones — solo mide coincidencias matemáticas**

### ML Clustering (`analyzer/ml_clustering.py`)
- KMeans con k=4: safe / neutral / hunter / chaotic
- Isolation Forest para detección de anomalías
- Buffer histórico de 500 contextos en memoria
- Re-entrena automáticamente cada 50 nuevas muestras

### Motor dual (`analyzer/dual_engine.py`)
- Evalúa SAFE y HUNTER independientemente
- HUNTER tiene prioridad sobre SAFE si está activo
- Dead Market bloquea ambos modos

---

## Sistema de tiempo entre explosiones

Por cada nivel (5x, 8x, 10x, 20x, 50x, 100x) se guarda:

```python
# En spike_distance_history y spike_distances
{
    "level": 10.0,
    "rounds_since": 52,          # rondas actuales desde último 10x
    "avg_distance": 34.0,        # promedio histórico
    "std_distance": 8.4,         # desviación estándar
    "z_score": 2.14,             # cuántas σ sobre la media
    "overdue": True,             # ¿superó media + 1σ?
    "very_overdue": True,        # ¿superó media + 2σ?
    "pressure": 0.57,            # presión acumulada (0–1)
}
```

**IMPORTANTE**: Esto NO se usa para decir "ya debe salir". Solo es:
- Presión estadística descriptiva
- Rareza temporal cuantificada
- Contexto acumulativo para el score hunter/moon

---

## Dead Market Mode

Bloquea TODAS las señales si:

| Condición | Umbral |
|-----------|--------|
| Entropía extrema | ≥ 0.92 |
| Spikes recientes grandes | ≥ 2 spikes ≥ 8x en últimas 5r |
| Volatilidad absurda | std_10r ≥ 15.0 |
| Secuencia tóxica | 3+ ultra-low seguidos de ≥ 20x |
| Compresión falsa | compression_ratio ≥ 0.80 |
| Toxicidad combinada | score ≥ 0.50 |

---

## Base de datos v3

| Tabla | Contenido |
|-------|-----------|
| `rounds` | Cada ronda registrada |
| `signals` | Señales base con resultado |
| `spike_distances` | Distancias entre spikes por nivel |
| `tension_history` | Snapshots de tensión cada 10r |
| `compression_events` | Eventos de compresión extrema |
| `pre_spike_contexts` | Contextos guardados antes de spikes confirmados |
| `advanced_signals` | Señales del motor avanzado |
| `entropy_snapshots` | Historial de entropía cada 15r |
| `binary_events` | Eventos binarios raros |
| `clustering_snapshots` | Estado de clustering cada 20r |
| `dual_signals` | Señales del motor dual con scores completos |
| `fibonacci_events` | Coincidencias Fibonacci relevantes |
| `mode_effectiveness` | Efectividad acumulada por modo y target |
| `spike_distance_history` | Historial de distancias para Fibonacci |

---

## Cómo migrar desde v2

```bash
# main.py original → seguir funcionando
python main.py

# main_v3.py → sistema nuevo
python main_v3.py
```

Ambos usan la misma DB (`jetx_data.db`). Las nuevas tablas se crean automáticamente.

---

## Variables de entorno nuevas (opcionales)

No se requieren cambios al `.env`. Todas las nuevas constantes tienen defaults sensatos.

---

## Aviso

Este sistema es una herramienta de observación estadística experimental.
No garantiza resultados ni constituye asesoramiento financiero.
JetX es un sistema pseudoaleatorio — ningún análisis estadístico puede predecirlo con certeza.
