"""Backward-compatible alias: canonical implementation lives in :mod:`app.infra.llm_client`.

Kept so existing imports (``from app import llm_client``) and test
monkeypatches keep working — both names resolve to the same module object.
"""

from __future__ import annotations

import sys

from app.infra import llm_client as _impl

sys.modules[__name__] = _impl  # type: ignore[assignment]
