"""Tagged logging with secret redaction."""
from __future__ import annotations

import logging
import sys


class RedactFilter(logging.Filter):
    def __init__(self, secrets: list[str] | None = None) -> None:
        super().__init__()
        self.secrets = [s for s in (secrets or []) if s]

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True

        for secret in self.secrets:
            if secret in msg:
                msg = msg.replace(secret, "***")

        record.msg = msg
        record.args = ()
        return True


_logger = logging.getLogger("factory")


def setup_logging(
    secrets: list[str] | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    _logger.handlers.clear()

    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            "%H:%M:%S",
        )
    )
    h.addFilter(RedactFilter(secrets))

    _logger.addHandler(h)
    _logger.setLevel(level)
    _logger.propagate = False

    # Keep third-party libraries quiet enough to avoid leaking
    # URLs, credentials, request details, or API keys.
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "google_genai",
        "google",
        "twilio",
        "twilio.http_client",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return _logger


def log(
    tag: str,
    message: str,
    level: int = logging.INFO,
) -> None:
    _logger.log(level, "[%s] %s", tag, message)


def warn(tag: str, message: str) -> None:
    log(tag, message, logging.WARNING)


def error(tag: str, message: str) -> None:
    log(tag, message, logging.ERROR)
