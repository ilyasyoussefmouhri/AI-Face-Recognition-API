import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
import sys


logger = logging.getLogger(__name__)




# Different configs for dev vs prod
ENV = os.getenv("APP_ENV", "development")

if ENV == "production":
    LOG_LEVEL = logging.INFO
    LOG_DIR = Path("/var/log/ai_face_recognition")
    console_enabled = False
else:
    LOG_LEVEL = logging.DEBUG
    LOG_DIR = Path(__file__).parent / "logs"
    console_enabled = True

try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    print(f"WARNING: Cannot create log directory {LOG_DIR}, using console only",
              file=sys.stderr)
    LOG_DIR = None

if LOG_DIR:
    LOG_FILE = LOG_DIR / "app.log"

if not logger.handlers:
    fmt = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    file_handler = RotatingFileHandler(str(LOG_FILE), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.setLevel(LOG_LEVEL)
logger.propagate = False
