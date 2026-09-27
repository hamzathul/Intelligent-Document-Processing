"""Backward-compatible alias: canonical implementation lives in :mod:`app.infra.ocr_service`.

Kept so existing imports (``from app import ocr_service``) and test
monkeypatches keep working — both names resolve to the same module object.
"""

from __future__ import annotations

import sys

from app.infra import ocr_service as _impl

sys.modules[__name__] = _impl  # type: ignore[assignment]
