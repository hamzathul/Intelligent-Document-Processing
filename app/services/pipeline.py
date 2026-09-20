"""Sync invoice pipeline: OCR pages -> LLM extraction -> MD response shape.

Pure orchestration over the existing engine (``ocr_service``,
``prompt``, ``llm_client``, ``extract_service``, ``coerce``). No FastAPI
imports so it stays unit-testable with stubbed OCR/LLM.
"""

from __future__ import annotations

import json
from typing import Any

from app import coerce, extract_service, llm_client, prompt
from app.core.debug_trace import NULL_TRACE, DebugTrace
from app.dtos.invoice import (
    FieldMapping,
    KeyValuePair,
    MatchingCondition,
    MatchResult,
    ProcessedInvoiceResponse,
    TableCell,
)
from app.fields import ExtractSpec
from app.services import field_adapter, matcher


def stringify(value: object) -> str | None:
    """MD contract: ``value`` is always a string (or null for missing)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_invoice_messages(
    spec: ExtractSpec,
    ocr_text: str,
    document_type: str | None = None,
    generic_hint: str | None = None,
    extraction_json: dict[str, Any] | None = None,
    pages: list[dict] | None = None,
) -> list[dict]:
    """Wrap :func:`prompt.build_messages` with MD-level context (no signature break).

    When ``pages`` carry OCR lines (text/score/box), a position+confidence
    annotated layout is sent instead of the bare ``ocr_text``.
    """
    layout = prompt.render_layout_text(pages) if pages else None
    messages = prompt.build_messages(spec, ocr_text, layout=layout or None)
    extras: list[str] = []
    if document_type:
        extras.append(f"DOCUMENT TYPE: {document_type}")
    if generic_hint:
        extras.append(f"VENDOR/SYSTEM HINT: {generic_hint}")
    if extraction_json is not None:
        extras.append(
            "PRIOR EXTRACTION (re-extract or correct it based on the document text):\n"
            + json.dumps(extraction_json)[:4000]
        )
    if extras:
        user = messages[1]["content"]
        marker = "DOCUMENT TEXT (OCR, top-to-bottom):"
        context = "\n".join(extras)
        if marker in user:
            user = user.replace(marker, f"ADDITIONAL CONTEXT:\n{context}\n\n{marker}", 1)
        else:
            user = f"{user}\n\nADDITIONAL CONTEXT:\n{context}"
        messages[1]["content"] = user
    return messages


def extract_from_ocr_text(
    ocr_text: str,
    fields: list[FieldMapping],
    document_type: str | None = None,
    generic_hint: str | None = None,
    extraction_json: dict[str, Any] | None = None,
    trace: DebugTrace | None = None,
    pages: list[dict] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], ExtractSpec]:
    """Run LLM extraction over OCR text. Returns (headers, rows, spec).

    Raises ``llm_client.LLMNotConfigured`` / ``llm_client.LLMError``.
    When ``trace`` is given, the exact LLM request/response and the coerced
    output are persisted for debugging (best-effort, never raises).
    """
    tr = trace or NULL_TRACE
    spec = field_adapter.adapt_fields(fields)
    messages = build_invoice_messages(
        spec, ocr_text, document_type, generic_hint, extraction_json, pages=pages
    )
    schema = prompt.build_json_schema(spec)
    tr.save_json(
        "03_llm_request.json",
        {"model": llm_client.model_name(), "messages": messages, "json_schema": schema},
    )
    try:
        raw = llm_client.complete_json(messages, schema)
    except (llm_client.LLMNotConfigured, llm_client.LLMError) as exc:
        tr.save_json(
            "04_llm_error.json",
            {"error": str(exc), "raw": getattr(exc, "raw", None)},
        )
        raise
    tr.save_json("04_llm_response.json", raw)

    raw_headers = raw.get("headers") or {}
    headers: dict[str, Any] = {}
    for f in spec.headers:
        headers[f.key] = coerce.apply_type(raw_headers.get(f.key), f.type)

    rows: list[dict[str, Any]] = []
    if spec.line_items:
        for row in raw.get("line_items") or []:
            if not isinstance(row, dict):
                continue
            rows.append({f.key: coerce.apply_type(row.get(f.key), f.type) for f in spec.line_items})
    tr.save_json("05_extracted.json", {"headers": headers, "line_items": rows})
    return headers, rows, spec


def to_response(
    unique_ref_no: str,
    fields: list[FieldMapping],
    headers: dict[str, Any],
    rows: list[dict[str, Any]],
    match_results: list[MatchResult] | None = None,
) -> ProcessedInvoiceResponse:
    """Map engine output to the MD §3 sync payload (confidence null in v1)."""
    header_order = [f.unique_key for f in fields if f.category == "header"]
    item_order = [f.unique_key for f in fields if f.category == "line_item"]
    pairs = [
        KeyValuePair(key=k, value=stringify(headers.get(k)), confidence=None)
        for k in header_order
    ]
    tables = [
        [TableCell(key=k, value=stringify(row.get(k))) for k in item_order] for row in rows
    ]
    return ProcessedInvoiceResponse(
        unique_ref_no=unique_ref_no,
        status="success",
        key_value_pairs=pairs,
        tables=tables,
        match_results=match_results or [],
    )


def run_matching(
    rows: list[dict[str, Any]],
    match: bool,
    po_items: list[Any] | None,
    conditions: list[MatchingCondition] | None,
    fields: list[FieldMapping],
) -> list[MatchResult]:
    if not match:
        return []
    return matcher.match_invoice_rows(rows, po_items or [], conditions or [], fields)


def extract_from_pages(
    pages: list[dict],
    fields: list[FieldMapping],
    unique_ref_no: str,
    document_type: str | None = None,
    match: bool = False,
    po_items: list[Any] | None = None,
    conditions: list[MatchingCondition] | None = None,
    generic_hint: str | None = None,
    extraction_json: dict[str, Any] | None = None,
    trace: DebugTrace | None = None,
) -> ProcessedInvoiceResponse:
    """Full in-memory pipeline used by the route (OCR already done) and tests."""
    tr = trace or NULL_TRACE
    ocr_text = "\n".join(p.get("text", "") for p in pages).strip()
    headers, rows, _ = extract_from_ocr_text(
        ocr_text, fields, document_type, generic_hint, extraction_json, trace=tr, pages=pages
    )
    # Re-inject page text per row? No — rows already carry extracted values.
    matched = run_matching(rows, match, po_items, conditions, fields)
    tr.save_json("06_match.json", [m.model_dump() for m in matched])
    # Attach ocr_text indirectly via headers? No — response shape has no ocr_text field.
    _ = extract_service  # keep engine import explicit for future confidence support
    response = to_response(unique_ref_no, fields, headers, rows, matched)
    tr.save_json("07_response.json", response.model_dump())
    return response
