"""Shared FastAPI dependencies: upload validation (used by all file routes)."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.config import Settings


async def read_validated_upload(file: UploadFile, settings: Settings) -> tuple[str, bytes]:
    """Validate type/size and return (suffix, bytes)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.allowed_suffixes and (file.content_type or "") not in {
        "image/png",
        "image/jpeg",
        "image/bmp",
        "image/tiff",
        "image/webp",
        "application/pdf",
    }:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix or file.content_type}. "
            f"Allowed: {sorted(settings.allowed_suffixes)}",
        )
    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(blob) > settings.max_bytes:
        raise HTTPException(
            status_code=413, detail=f"File too large (> {settings.ocr_max_mb:g} MB)"
        )
    return suffix, blob
