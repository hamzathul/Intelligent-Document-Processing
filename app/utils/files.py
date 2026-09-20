"""Temporary-file handling with guaranteed cleanup."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def saved_temp_file(suffix: str, blob: bytes) -> Iterator[str]:
    """Write ``blob`` to a temp file and yield its path; always unlink afterwards."""
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".png") as tmp:
            tmp.write(blob)
            tmp_path = tmp.name
        yield tmp_path
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
