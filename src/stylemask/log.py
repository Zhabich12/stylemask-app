"""Event log for diagnosing the hotkey flow: data/stylemask.log.

Only events and sizes are logged — never the text being rewritten.
"""
import logging
from pathlib import Path

PATH = Path("data/stylemask.log")


def get() -> logging.Logger:
    logger = logging.getLogger("stylemask")
    if not logger.handlers:
        PATH.parent.mkdir(exist_ok=True)
        handler = logging.FileHandler(PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
