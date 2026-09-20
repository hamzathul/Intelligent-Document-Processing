"""Structured stage logging helper."""

from __future__ import annotations

import logging
import time


def get_logger(name: str = "uvicorn.error") -> logging.Logger:
    return logging.getLogger(name)


def log_stage(route: str, stage: str, started: float, **fields: object) -> float:
    """Log a finished pipeline stage with duration. Returns a fresh timestamp."""
    elapsed = time.perf_counter() - started
    msg = f"[{route}] {stage} done in {elapsed:.1f}s"
    if fields:
        msg += " (" + " ".join(f"{k}={v}" for k, v in fields.items()) + ")"
    get_logger().info(msg)
    return time.perf_counter()
