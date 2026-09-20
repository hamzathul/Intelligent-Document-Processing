"""Minimal PO matcher for ``match=true`` requests.

Implements the MD ``matching_conditions`` contract (priority ordered,
``exact | partial | fuzzy | numeric``) over already-extracted invoice rows.
Scores are 0..5 to match the MD example. ``fuzzy`` uses difflib as a
placeholder for a future crossencoder upgrade — the response shape stays
identical so core is unaffected.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from app import coerce
from app.dtos.invoice import FieldMapping, MatchingCondition, MatchResult

FUZZY_THRESHOLD = 3.0


def _str_norm(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    return str(value).strip().lower()


def _score_exact(invoice_val: Any, po_val: Any) -> float:
    if invoice_val is None or po_val is None:
        return 0.0
    inv_num = coerce.to_number(invoice_val) if not isinstance(invoice_val, bool) else None
    po_num = coerce.to_number(po_val) if not isinstance(po_val, bool) else None
    if inv_num is not None and po_num is not None:
        # Only treat as numeric-exact when both sides look numeric.
        try:
            if float(str(invoice_val).strip()) == float(str(po_val).strip()):
                return 5.0
        except (TypeError, ValueError):
            pass
        return 0.0
    return 5.0 if _str_norm(invoice_val) == _str_norm(po_val) and _str_norm(invoice_val) else 0.0


def _score_numeric(invoice_val: Any, po_val: Any) -> float:
    inv = coerce.to_number(invoice_val)
    po = coerce.to_number(po_val)
    if inv is None or po is None:
        return 0.0
    return 5.0 if abs(inv - po) <= max(1e-9, 1e-6 * abs(po)) else 0.0


def _score_partial(invoice_val: Any, po_val: Any) -> float:
    a, b = _str_norm(invoice_val), _str_norm(po_val)
    if not a or not b:
        return 0.0
    return 4.0 if (a in b or b in a) else 0.0


def _score_fuzzy(invoice_val: Any, po_val: Any) -> float:
    a, b = _str_norm(invoice_val), _str_norm(po_val)
    if not a or not b:
        return 0.0
    return round(SequenceMatcher(None, a, b).ratio() * 5.0, 2)


_SCORERS = {
    "exact": (_score_exact, "exact", 5.0),
    "numeric": (_score_numeric, "numeric", 5.0),
    "partial": (_score_partial, "partial", 4.0),
    "fuzzy": (_score_fuzzy, "fuzzy", FUZZY_THRESHOLD),
}


def _po_value(po_item: Any, field: str) -> Any:
    if isinstance(po_item, dict):
        return po_item.get(field)
    get_value = getattr(po_item, "get_value", None)
    if callable(get_value):
        return get_value(field)
    return getattr(po_item, field, None)


def match_invoice_rows(
    rows: list[dict[str, Any]],
    po_items: list[Any],
    conditions: list[MatchingCondition],
    fields: list[FieldMapping] | None = None,
) -> list[MatchResult]:
    """Match each invoice row against PO items; first matching priority wins per row."""
    if not rows or not po_items or not conditions:
        return []
    # Same unique_key may exist as both a header total and a line value;
    # matching runs on rows, so line_item mappings win for document labels.
    lookup: dict[str, FieldMapping] = {}
    for f in fields or []:
        if f.unique_key not in lookup or f.category == "line_item":
            lookup[f.unique_key] = f
    ordered = sorted(conditions, key=lambda c: c.priority)
    results: list[MatchResult] = []
    for row in rows:
        for cond in ordered:
            scorer, method, threshold = _SCORERS[cond.match_type]
            invoice_val = row.get(cond.invoice_field_key)
            if invoice_val is None or (isinstance(invoice_val, str) and not invoice_val.strip()):
                continue
            best_score = 0.0
            best_po: Any = None
            for po in po_items:
                score = scorer(invoice_val, _po_value(po, cond.po_field))
                if score > best_score:
                    best_score, best_po = score, po
            if best_po is not None and best_score >= threshold:
                mapping = lookup.get(cond.invoice_field_key)
                doc_field = (
                    (mapping.document_field or mapping.display_name or mapping.invoice_field)
                    if mapping
                    else cond.invoice_field_key
                )
                results.append(
                    MatchResult(
                        score=best_score,
                        best_match=None
                        if _po_value(best_po, cond.po_field) is None
                        else str(_po_value(best_po, cond.po_field)),
                        invoice_item=None if invoice_val is None else str(invoice_val),
                        po_usage_count=1,
                        po_field=cond.po_field,
                        invoice_field=cond.invoice_field_key,
                        invoice_document_field=doc_field,
                        match_type=cond.match_type,
                        matching_method=method,
                        priority=cond.priority,
                    )
                )
                break  # priority wins for this row
    return results
