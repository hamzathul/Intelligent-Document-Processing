"""Shared multipart-form validators for the invoice routes.

Extracted from the original ``POST /extract/invoice`` handler so the OCR
route (``invoice.py``) and the direct-VLM route (``invoice_vlm.py``) share
one validation implementation and identical 400 semantics.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

from app.dtos.invoice import FieldMapping, MatchingCondition

_fields_adapter = TypeAdapter(list[FieldMapping])
_conditions_adapter = TypeAdapter(list[MatchingCondition])

_TRUE_TOKENS = {"true", "1", "yes", "y", "on"}
_FALSE_TOKENS = {"false", "0", "no", "n", "off", ""}


def parse_json_form(raw: str | None, name: str, *, required: bool) -> Any | None:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if required:
            raise HTTPException(status_code=400, detail=f"{name} is required")
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{name} is not valid JSON: {exc}") from exc


def parse_match_flag(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        raise HTTPException(status_code=400, detail="match is required")
    token = str(raw).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise HTTPException(status_code=400, detail="match must be true or false")


def validated_fields(raw: Any) -> list[FieldMapping]:
    if not isinstance(raw, list) or not raw:
        raise HTTPException(status_code=400, detail="fields must be a non-empty JSON array")
    try:
        fields = _fields_adapter.validate_python(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"invalid fields: {exc}") from exc
    # unique_key must be unique *within* each category (header / line_item).
    # The same key may appear once as a header total and once as a per-row
    # line value (e.g. GST header totals vs row tax).
    for category in ("header", "line_item"):
        seen: set[str] = set()
        dupes: set[str] = set()
        for f in fields:
            if f.category != category:
                continue
            if f.unique_key in seen:
                dupes.add(f.unique_key)
            seen.add(f.unique_key)
        if dupes:
            raise HTTPException(
                status_code=400,
                detail=f"duplicate unique_key in {category} fields: {sorted(dupes)}",
            )
    return fields


def validated_conditions(raw: Any, fields: list[FieldMapping]) -> list[MatchingCondition]:
    if not isinstance(raw, list) or not raw:
        raise HTTPException(
            status_code=400, detail="matching_conditions must be a non-empty JSON array"
        )
    try:
        conditions = _conditions_adapter.validate_python(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid matching_conditions: {exc}"
        ) from exc
    known = {f.unique_key for f in fields}
    for cond in conditions:
        if cond.invoice_field_key not in known:
            raise HTTPException(
                status_code=400,
                detail=f"matching_conditions invoice_field_key "
                f"'{cond.invoice_field_key}' is not in fields[].unique_key",
            )
    return conditions


def validated_po_items(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise HTTPException(
            status_code=400, detail="po_items must be a non-empty JSON array"
        )
    rows: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            rows.append(item)
        elif hasattr(item, "model_dump"):
            rows.append(item.model_dump())
        else:
            raise HTTPException(status_code=400, detail="po_items must be an array of objects")
    return rows


def validated_prior_extraction(raw: str | None) -> dict[str, Any] | None:
    """Parse optional ``extraction_json`` (must be a JSON object when present)."""
    if raw is None or not str(raw).strip():
        return None
    parsed = parse_json_form(raw, "extraction_json", required=False)
    if parsed is not None and not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="extraction_json must be a JSON object")
    return parsed


def validated_ref(unique_ref_no: str | None) -> str:
    ref = (unique_ref_no or "").strip()
    if not ref:
        raise HTTPException(status_code=400, detail="unique_ref_no is required")
    return ref
