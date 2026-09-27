"""Centralised runtime settings (no new dependency).

Reads env (`.env` is loaded by :mod:`app`), falls back to sane defaults
matching `.env.example`. Use :func:`get_settings` instead of scattering
``os.getenv`` across routes/services so limits and model names stay in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    ocr_model: str = field(default_factory=lambda: os.getenv("OCR_MODEL", "PP-OCRv6_medium"))
    ocr_engine: str = field(default_factory=lambda: os.getenv("OCR_ENGINE", "paddle"))
    ocr_device: str = field(default_factory=lambda: os.getenv("OCR_DEVICE", "cpu"))
    ocr_cpu_threads: int = field(default_factory=lambda: _get_int("OCR_CPU_THREADS", 4))
    ocr_max_mb: float = field(default_factory=lambda: _get_float("OCR_MAX_MB", 15.0))

    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_timeout_s: float = field(default_factory=lambda: _get_float("LLM_TIMEOUT_S", 60.0))

    # Direct-VLM invoice route (POST /extract/invoice/vlm): vision model that
    # reads page images, no PaddleOCR involved. Defaults to Google AI Studio
    # (Gemini Developer API, OpenAI-compatible endpoint). VLM_* explicitly
    # set always wins; GOOGLE_API_KEY / GEMINI_MODEL are the primary names.
    vlm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "VLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    )
    vlm_api_key: str = field(
        default_factory=lambda: os.getenv(
            "VLM_API_KEY",
            os.getenv(
                "GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("LLM_API_KEY", ""))
            ),
        )
    )
    vlm_model: str = field(
        default_factory=lambda: os.getenv(
            "VLM_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        )
    )
    vlm_timeout_s: float = field(default_factory=lambda: _get_float("VLM_TIMEOUT_S", 120.0))
    vlm_max_pages: int = field(default_factory=lambda: _get_int("VLM_MAX_PAGES", 8))
    vlm_max_side_px: int = field(default_factory=lambda: _get_int("VLM_MAX_SIDE_PX", 1568))
    vlm_jpeg_quality: int = field(default_factory=lambda: _get_int("VLM_JPEG_QUALITY", 85))

    # Per-request debug traces (see app.core.debug_trace).
    debug_trace_enabled: bool = field(default_factory=lambda: _get_bool("IDP_DEBUG_TRACE", True))
    debug_trace_dir: str = field(default_factory=lambda: os.getenv("IDP_DEBUG_DIR", "output/debug"))
    debug_trace_keep: int = field(default_factory=lambda: _get_int("IDP_DEBUG_KEEP", 100))
    debug_trace_save_uploads: bool = field(
        default_factory=lambda: _get_bool("IDP_DEBUG_SAVE_UPLOADS", False)
    )

    allowed_suffixes: frozenset[str] = frozenset(
        {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".pdf"}
    )
    allowed_mimes: frozenset[str] = frozenset(
        {
            "image/png",
            "image/jpeg",
            "image/bmp",
            "image/tiff",
            "image/webp",
            "application/pdf",
        }
    )

    @property
    def max_bytes(self) -> int:
        return int(self.ocr_max_mb * 1024 * 1024)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings. Call ``get_settings.cache_clear()`` in tests if env changes."""
    return Settings()
