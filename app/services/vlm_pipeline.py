"""Direct-VLM invoice pipeline: page images -> VLM extraction -> MD response shape.

Pure orchestration over the existing engine (``field_adapter``, ``coerce``,
``matcher``, :mod:`app.services.pipeline` response mapping). No FastAPI
imports so it stays unit-testable with a stubbed VLM client. PaddleOCR is
never touched on this path.
"""

from __future__ import annotations

import json
from typing import Any

from app import coerce, prompt
from app.core.debug_trace import NULL_TRACE, DebugTrace
from app.dtos.invoice import (
    FieldMapping,
    MatchingCondition,
    MatchResult,
    ProcessedInvoiceResponse,
)
from app.fields import ExtractSpec
from app.infra import vlm_client
from app.services import field_adapter, matcher, pipeline, vlm_prompt
from app.services.images import PageImage


def build_vlm_messages(
    spec: ExtractSpec,
    pages: list[PageImage],
    document_type: str | None = None,
    generic_hint: str | None = None,
    extraction_json: dict[str, Any] | None = None,
) -> list[dict]:
    """Wrap :mod:`app.services.vlm_prompt` (kept separate for test patch points)."""
    return vlm_prompt.build_vision_messages(
        spec, pages, document_type, generic_hint, extraction_json
    )


def extract_from_images(
    pages: list[PageImage],
    fields: list[FieldMapping],
    document_type: str | None = None,
    generic_hint: str | None = None,
    extraction_json: dict[str, Any] | None = None,
    trace: DebugTrace | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], ExtractSpec]:
    """Run VLM extraction over rendered pages. Returns (headers, rows, spec).

    Raises ``vlm_client.LLMNotConfigured`` / ``vlm_client.LLMError``.
    """
    tr = trace or NULL_TRACE
    spec = field_adapter.adapt_fields(fields)
    messages = build_vlm_messages(spec, pages, document_type, generic_hint, extraction_json)
    schema = prompt.build_json_schema(spec)  # same contract as the OCR path
    tr.save_json(
        "03_vlm_request.json",
        {
            "model": vlm_client.model_name(),
            "messages": vlm_prompt.redact_for_trace(messages),
            "json_schema": schema,
        },
    )
    try:
        raw = vlm_client.complete_vision(messages, schema)
    except (vlm_client.LLMNotConfigured, vlm_client.LLMError) as exc:
        tr.save_json(
            "04_vlm_error.json",
            {"error": str(exc), "raw": getattr(exc, "raw", None)},
        )
        raise
    tr.save_json("04_vlm_response.json", raw)

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


def extract_invoice_vlm(
    pages: list[PageImage],
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
    """Full in-memory VLM pipeline used by the route (rendering already done) and tests."""
    tr = trace or NULL_TRACE
    headers, rows, _ = extract_from_images(
        pages, fields, document_type, generic_hint, extraction_json, trace=tr
    )
    matched: list[MatchResult] = pipeline.run_matching(rows, match, po_items, conditions, fields)
    tr.save_json("06_match.json", [m.model_dump() for m in matched])
    response = pipeline.to_response(unique_ref_no, fields, headers, rows, matched)
    response.intermediate_data = {
        "model": vlm_client.model_name(),
        "mode": "vlm-direct",
        "pages": len(pages),
    }
    tr.save_json("07_response.json", response.model_dump())
    return response


# Keep engine imports explicit for future confidence support.
_ = (json, matcher)
