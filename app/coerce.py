"""Deterministic type coercion for extracted values.

The LLM *reads*; this module *validates*. Every helper is pure (no network,
no state) so it is cheap to unit-test. Anything unparseable becomes ``None``
instead of raising — a missing value must never break the whole response.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from app.fields import FieldType

_CURRENCY_CHARS = "$€£¥₹"
_CURRENCY_CODES = re.compile(
    r"\b(SAR|USD|AED|EUR|GBP|INR|PKR|QAR|KWD|BHD|OMR|EGP|TRY|CNY|JPY|CHF|CAD|AUD)\b",
    re.IGNORECASE,
)
_THOUSAND_SEPS = (",", " ", "\u00a0")

# Day-first order: most invoice locales write DD/MM/YYYY. Ambiguous dates
# like 05/06/2026 therefore parse as 5 June. Documented, not guessed.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y-%b-%d",
    "%Y-%B-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%m-%d-%Y",
    "%m/%d/%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)

_TRUE_WORDS = {"true", "yes", "y", "1", "on", "paid", "inclusive"}
_FALSE_WORDS = {"false", "no", "n", "0", "off", "unpaid", "exclusive", "none", "n/a"}


def to_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def to_number(value: object) -> float | None:
    """Parse 1,299.50 / $12.50 / (7.25) / 18% style amounts."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "na", "-", "--", "nil", "null"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()%")
    text = _CURRENCY_CODES.sub("", text)
    for ch in _CURRENCY_CHARS:
        text = text.replace(ch, "")
    for sep in _THOUSAND_SEPS:
        text = text.replace(sep, "")
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def to_integer(value: object) -> int | None:
    number = to_number(value)
    if number is None:
        return None
    return int(number) if number.is_integer() else None


def to_boolean(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    word = str(value).strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    return None


def to_date(value: object) -> str | None:
    """Return an ISO ``YYYY-MM-DD`` string, or ``None`` if unparseable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()[:10]
    text = str(value).strip().rstrip("Z")
    if not text:
        return None
    try:  # ISO first: 2026-08-12, 2026-08-12T00:00:00
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def apply_type(value: object, field_type: FieldType) -> object:
    """Coerce one raw LLM value to the declared type (None on failure)."""
    if field_type == FieldType.NUMBER:
        return to_number(value)
    if field_type == FieldType.INTEGER:
        return to_integer(value)
    if field_type == FieldType.DATE:
        return to_date(value)
    if field_type == FieldType.BOOLEAN:
        return to_boolean(value)
    return to_string(value)
