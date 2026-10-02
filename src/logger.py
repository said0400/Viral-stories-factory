"""Tagged logging with secret redaction."""
from __future__ import annotations

import logging
import sys


class RedactFilter(logging.Filter):
    def __init__(self, secrets: list[str] | None = None) -> None:
        super().__init__()
        self.secrets = [s for s in (secrets or []) if s]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for s in self.secrets:
            if s in msg:
                msg = msg.replace(s, "***")
        record.msg, record.args = msg, ()
        return True


_logger = logging.getLogger("factory")


def setup_logging(secrets: list[str] | None = None, level: int = logging.INFO) -> logging.Logger:
    _logger.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    h.addFilter(RedactFilter(secrets))
    _logger.addHandler(h)
    _logger.setLevel(level)
    _logger.propagate = False
    # keep third-party libs from leaking URLs with keys
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "twilio.http_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return _logger


def log(tag: str, message: str, level: int = logging.INFO) -> None:
    _logger.log(level, "[%s] %s", tag, message)


def warn(tag: str, message: str) -> None:
    log(tag, message, logging.WARNING)


def error(tag: str, message: str) -> None:
    log(tag, message, logging.ERROR)
