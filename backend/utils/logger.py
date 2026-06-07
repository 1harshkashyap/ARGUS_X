import logging
import re
import sys
from typing import Optional


# ── Key redaction patterns ────────────────────────────────────────────
# These patterns match real API key formats and replace them with [REDACTED]
_REDACT_PATTERNS = [
    re.compile(r'AIza[A-Za-z0-9_\-]{30,}'),   # Gemini keys
    re.compile(r'sk-[A-Za-z0-9]{40,}'),         # OpenAI keys
    re.compile(r'eyJ[A-Za-z0-9_\-\.]{50,}'),    # JWT tokens (Supabase keys)
]


def redact_keys(message: str) -> str:
    """Replace any API key values in log messages with [REDACTED]."""
    if not isinstance(message, str):
        message = str(message)
    for pattern in _REDACT_PATTERNS:
        message = pattern.sub("[REDACTED]", message)
    return message


class RedactingFormatter(logging.Formatter):
    """Custom formatter that redacts sensitive values from all log output."""

    def format(self, record: logging.LogRecord) -> str:
        # Redact the message
        record.msg = redact_keys(str(record.msg))
        # Redact any args passed to the logger
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: redact_keys(str(v)) for k, v in record.args.items()}
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(redact_keys(str(a)) for a in record.args)
        return super().format(record)


def get_logger(name: str = "argus_x") -> logging.Logger:
    """
    Create and return a secure logger with key redaction.
    Safe to call multiple times — returns same logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Secure formatter with redaction
    formatter = RedactingFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


# Global logger instance — import this everywhere
logger = get_logger("argus_x")
