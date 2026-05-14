"""
core/config.py
Variables de entorno y constantes del sistema JetX v3.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_KEY         = os.getenv("GROQ_KEY", "")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID          = os.getenv("CHAT_ID", "")

# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS  = int(os.getenv("GROQ_MAX_TOKENS", "120"))
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.05"))

# ── Base de datos ─────────────────────────────────────────────────────────────
DB_PATH          = os.getenv("DB_PATH", "jetx_data.db")

# ── Browser ───────────────────────────────────────────────────────────────────
USER_DATA_DIR         = os.getenv("USER_DATA_DIR", "browser_profile")
BETPLAY_URL           = os.getenv("BETPLAY_URL", "https://betplay.com.co")
BROWSER_RESTART_HOURS = float(os.getenv("BROWSER_RESTART_HOURS", "4.0"))

# ── Logs ──────────────────────────────────────────────────────────────────────
LOG_DIR          = os.getenv("LOG_DIR", "logs")

# ── Scores — umbrales por tipo de señal ──────────────────────────────────────
SCORE_CONSERVATIVE = int(os.getenv("SCORE_CONSERVATIVE", "55"))
SCORE_INTERMEDIATE = int(os.getenv("SCORE_INTERMEDIATE", "68"))
SCORE_AGGRESSIVE   = int(os.getenv("SCORE_AGGRESSIVE", "80"))

# ── Bloqueos ──────────────────────────────────────────────────────────────────
SPIKE_THRESHOLD           = float(os.getenv("SPIKE_THRESHOLD", "5.0"))
POST_SPIKE_BLOCK_ROUNDS   = int(os.getenv("POST_SPIKE_BLOCK_ROUNDS", "3"))

# ── Cooldown entre señales ────────────────────────────────────────────────────
COOLDOWN_ROUNDS  = int(os.getenv("COOLDOWN_ROUNDS", "5"))
COOLDOWN_MINUTES = float(os.getenv("COOLDOWN_MINUTES", "3.0"))

# ── Historia mínima para analizar ────────────────────────────────────────────
MIN_HISTORY      = int(os.getenv("MIN_HISTORY", "30"))
