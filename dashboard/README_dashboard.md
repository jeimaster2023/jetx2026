# JetX Dashboard v3 — Guía de Instalación

## Arquitectura

```
jetx_v3/
├── main_v3.py              ← Bot principal (ya existente)
├── dashboard_server.py     ← ★ NUEVO: Servidor FastAPI para el dashboard
└── dashboard/
    └── index.html          ← ★ NUEVO: Dashboard frontend completo
```

El `dashboard_server.py` lee la misma DB (`jetx_data.db`) en modo **solo lectura**.
Corre **en paralelo** con `main_v3.py` — no interfiere con él.

---

## Instalación rápida

```bash
pip install fastapi uvicorn aiosqlite
```

---

## Ejecutar

### Terminal 1 — Bot principal (ya lo tienes corriendo)
```bash
python main_v3.py
```

### Terminal 2 — Servidor del dashboard
```bash
cd jetx_v3
python dashboard_server.py
```

Inicia en `http://localhost:8765`

### Abrir el dashboard
Abre `dashboard/index.html` directamente en Chrome/Firefox.

El dashboard se conecta automáticamente al WebSocket en `ws://localhost:8765/ws`.

---

## Sin servidor (Modo Demo)

Si no tienes el servidor corriendo, el dashboard detecta la falta de conexión
en 3 segundos y activa **modo demo** con datos simulados realistas.
Esto te permite ver el dashboard funcionando sin necesidad del backend.

---

## API REST disponible

| Endpoint | Descripción |
|----------|-------------|
| `GET /api/history?limit=200` | Últimas N rondas |
| `GET /api/stats` | Estadísticas completas |
| `GET /api/signals?limit=20` | Señales del motor dual |
| `GET /api/full` | Todo junto (para init) |
| `WS /ws` | Stream en tiempo real |

---

## ¿Qué muestra el dashboard?

### 🛩️ Simulador
- Avión animado en Canvas que despega y vuela siguiendo la curva del multiplicador
- Animación fluida con trail, humo y explosión al terminar
- El multiplicador crece en tiempo real visible en pantalla
- Colores: azul (normal) → verde (10x+) → dorado (100x+) → rojo (explosión)

### 📊 Historial en tiempo real
- Últimas 80 rondas con badges de colores (rojo/amarillo/verde/morado)
- Las rondas nuevas aparecen con animación

### 📈 Gráficas (4 tabs)
- **Línea**: comportamiento de los últimas 60 rondas (escala log)
- **Histograma**: distribución de frecuencias por rangos
- **Rangos**: barras horizontales por categorías
- **Heatmap**: grid de intensidad de frecuencias

### 🧠 Panel IA
- Estado del mercado (Estable / Agresivo / Peligroso / etc.)
- Barras de volatilidad, estabilidad, momentum, riesgo crash
- Alertas automáticas

### 📋 Estadísticas avanzadas
- Media, mediana, desviación estándar, máximo
- Rondas sin 10x / 25x / 50x / 100x
- Rachas consecutivas altas/bajas
- Multiplicadores calientes y fríos
- Indicador de riesgo con cursor

### 🎯 Señales recientes
- Señales del motor dual (SAFE / HUNTER / MOON)
- Scores separados y resultado (✅/❌)

### 📝 Log en tiempo real
- Eventos importantes con timestamps
