"""Per-request debug traces for invoice extraction.

Each ``POST /extract/invoice`` call gets its own directory::

    output/debug/<unique_ref_no>_<UTC-timestamp>_<shortid>/
        01_request.json       form metadata (filename, size, flags, counts)
        00_upload.<ext>       original file (only if IDP_DEBUG_SAVE_UPLOADS=1)
        02_ocr.json / .txt    parsed OCR pages + full text (boxes stripped)
        03_llm_request.json   exact messages + JSON schema sent to the LLM
        04_llm_response.json  parsed LLM output (or 04_llm_error.json on failure)
        05_extracted.json     coerced headers + line items
        06_match.json         match_results
        07_response.json      final sync payload
        08_meta.json          timings, model, trace dir
        error.json            stage + detail when the request fails

Tracing never breaks a request: all writes are best-effort (warnings only).
Disable with ``IDP_DEBUG_TRACE=0``. Old traces are pruned to
``IDP_DEBUG_KEEP`` directories (default 100).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("uvicorn.error")

_SANITIZE = re.compile(r"[^A-Za-z0-9_-]+")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


@dataclass
class DebugTrace:
    """A single request's trace directory. No-op when disabled."""

    enabled: bool = False
    dir: Path | None = None

    def save_json(self, name: str, obj: Any) -> None:
        if not self.enabled or self.dir is None:
            return
        try:
            (self.dir / name).write_text(_dumps(obj), encoding="utf-8")
        except OSError as exc:
            log.warning("debug trace write failed (%s): %s", name, exc)

    def save_text(self, name: str, text: str) -> None:
        if not self.enabled or self.dir is None:
            return
        try:
            (self.dir / name).write_text(text, encoding="utf-8")
        except OSError as exc:
            log.warning("debug trace write failed (%s): %s", name, exc)

    def save_bytes(self, name: str, blob: bytes) -> None:
        if not self.enabled or self.dir is None:
            return
        try:
            (self.dir / name).write_bytes(blob)
        except OSError as exc:
            log.warning("debug trace write failed (%s): %s", name, exc)


NULL_TRACE = DebugTrace(enabled=False, dir=None)


def _prune(base: Path, keep: int) -> None:
    try:
        dirs = sorted(
            (d for d in base.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
        )
    except OSError:
        return
    for stale in dirs[: max(0, len(dirs) - keep)]:
        try:
            for child in stale.iterdir():
                child.unlink()
            stale.rmdir()
        except OSError as exc:
            log.warning("debug trace prune failed (%s): %s", stale, exc)


def new_trace(
    unique_ref_no: str,
    *,
    base_dir: str = "output/debug",
    enabled: bool = True,
    keep: int = 100,
) -> DebugTrace:
    """Create a trace directory for one request (or a no-op when disabled)."""
    if not enabled:
        return NULL_TRACE
    safe_ref = _SANITIZE.sub("-", (unique_ref_no or "noref").strip())[:40] or "noref"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    short = uuid.uuid4().hex[:6]
    path = Path(base_dir) / f"{safe_ref}_{stamp}_{short}"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("debug trace disabled for this request (mkdir failed): %s", exc)
        return NULL_TRACE
    if keep > 0:
        _prune(path.parent, keep)
    return DebugTrace(enabled=True, dir=path)
