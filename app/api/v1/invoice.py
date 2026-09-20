"""Sync invoice extraction endpoint (MD §1 request -> §3 payload, no callback).

``POST /extract/invoice`` accepts the core multipart contract and returns
:class:`ProcessedInvoiceResponse` directly. ``callback_url`` is accepted for
contract compatibility but ignored — nothing is POSTed anywhere.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import TypeAdapter, ValidationError

from app import llm_client, ocr_service
from app.api.deps import read_validated_upload
from app.core.config import get_settings
from app.core.debug_trace import new_trace
from app.core.logging import get_logger, log_stage
from app.dtos.invoice import (
    FieldMapping,
    MatchingCondition,
    ProcessedInvoiceResponse,
)
from app.services import pipeline
from app.utils.files import saved_temp_file

log = get_logger()
router = APIRouter(tags=["extract"])

_fields_adapter = TypeAdapter(list[FieldMapping])
_conditions_adapter = TypeAdapter(list[MatchingCondition])

_TRUE_TOKENS = {"true", "1", "yes", "y", "on"}
_FALSE_TOKENS = {"false", "0", "no", "n", "off", ""}


def _parse_json_form(raw: str | None, name: str, *, required: bool) -> Any | None:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if required:
            raise HTTPException(status_code=400, detail=f"{name} is required")
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{name} is not valid JSON: {exc}") from exc


def _parse_match_flag(raw: Any) -> bool:
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


def _validated_fields(raw: Any) -> list[FieldMapping]:
    if not isinstance(raw, list) or not raw:
        raise HTTPException(status_code=400, detail="fields must be a non-empty JSON array")
    try:
        fields = _fields_adapter.validate_python(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"invalid fields: {exc}") from exc
    # unique_key must be unique *within* each category (header / line_item),
    # mirroring ExtractSpec. The same key may appear once as a header total
    # and once as a per-row line value (e.g. GST header totals vs row tax).
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


def _validated_conditions(raw: Any, fields: list[FieldMapping]) -> list[MatchingCondition]:
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


def _validated_po_items(raw: Any) -> list[dict[str, Any]]:
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


async def _ocr_text_pages(
    tmp_path: str, page_ranges: str | None
) -> tuple[list[dict], str]:
    t0 = time.perf_counter()
    log.info("[extract/invoice] ocr starting (PP-OCRv6 local inference)...")
    try:
        raw = await run_in_threadpool(ocr_service.predict_path, tmp_path, page_ranges)
        pages = ocr_service.parse_result(raw)
    except Exception as exc:  # noqa: BLE001
        log.exception("[extract/invoice] ocr FAILED after %.1fs", time.perf_counter() - t0)
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc
    full = "\n\n".join(p.get("text", "") for p in pages if p.get("text"))
    log_stage("extract/invoice", "ocr", t0, pages=len(pages), chars=len(full))
    return pages, full


@router.post("/extract/invoice", response_model=ProcessedInvoiceResponse)
async def extract_invoice(
    file: UploadFile = File(..., description="Invoice document (PDF/image)"),
    unique_ref_no: str = Form(..., description="Invoice id as string"),
    fields: str = Form(..., description="Field mappings JSON array"),
    match: str = Form(..., description="JSON bool string: 'true' or 'false'"),
    callback_url: Optional[str] = Form(default=None, description="Accepted but ignored (sync mode)"),
    document_type: Optional[str] = Form(default=None),
    po_items: Optional[str] = Form(default=None),
    matching_conditions: Optional[str] = Form(default=None),
    extraction_json: Optional[str] = Form(default=None),
    generic_extraction_hint: Optional[str] = Form(default=None),
    page_ranges: Optional[str] = Query(
        default=None, description='PDF pages, e.g. "1-3,5". Omit for all pages.'
    ),
) -> ProcessedInvoiceResponse:
    """Sync extraction: OCR -> LLM -> match, returned inline (no callback POST)."""
    t0 = time.time()
    ref = (unique_ref_no or "").strip()
    if not ref:
        raise HTTPException(status_code=400, detail="unique_ref_no is required")

    fields_raw = _parse_json_form(fields, "fields", required=True)
    field_mappings = _validated_fields(fields_raw)
    do_match = _parse_match_flag(match)

    po_rows: list[dict[str, Any]] | None = None
    conditions: list[MatchingCondition] | None = None
    if do_match:
        po_rows = _validated_po_items(_parse_json_form(po_items, "po_items", required=True))
        conditions = _validated_conditions(
            _parse_json_form(matching_conditions, "matching_conditions", required=True),
            field_mappings,
        )

    prior: dict[str, Any] | None = None
    if extraction_json is not None and str(extraction_json).strip():
        parsed_prior = _parse_json_form(extraction_json, "extraction_json", required=False)
        if parsed_prior is not None and not isinstance(parsed_prior, dict):
            raise HTTPException(status_code=400, detail="extraction_json must be a JSON object")
        prior = parsed_prior

    settings = get_settings()
    trace = new_trace(
        ref,
        base_dir=settings.debug_trace_dir,
        enabled=settings.debug_trace_enabled,
        keep=settings.debug_trace_keep,
    )
    suffix, blob = await read_validated_upload(file, settings)
    trace.save_json(
        "01_request.json",
        {
            "unique_ref_no": ref,
            "filename": file.filename,
            "size_kb": len(blob) // 1024,
            "suffix": suffix,
            "document_type": document_type,
            "match": do_match,
            "page_ranges": page_ranges,
            "model": llm_client.model_name(),
            "fields": [f.model_dump() for f in field_mappings],
            "po_items": po_rows,
            "matching_conditions": (
                [c.model_dump() for c in conditions] if conditions is not None else None
            ),
            "generic_extraction_hint": generic_extraction_hint,
            "extraction_json": prior,
            "callback_url_ignored": callback_url,
        },
    )
    if settings.debug_trace_save_uploads:
        trace.save_bytes(f"00_upload{suffix or '.bin'}", blob)
    log.info(
        "[extract/invoice] request ref=%s file=%s size_kb=%d fields=%d match=%s trace=%s",
        ref,
        file.filename,
        len(blob) // 1024,
        len(field_mappings),
        do_match,
        trace.dir,
    )

    with saved_temp_file(suffix, blob) as tmp_path:
        try:
            pages, full_text = await _ocr_text_pages(tmp_path, page_ranges)
        except HTTPException as exc:
            trace.save_json("error.json", {"stage": "ocr", "detail": exc.detail})
            raise
        trace.save_json(
            "02_ocr.json",
            {
                "pages": [
                    {
                        "page_index": p.get("page_index", i),
                        "text": p.get("text", ""),
                        "lines": [
                            {
                                "text": ln.get("text", ""),
                                "score": ln.get("score", 0.0),
                                "box": ln.get("box"),
                            }
                            for ln in p.get("lines", [])
                        ],
                    }
                    for i, p in enumerate(pages)
                ],
                "full_text": full_text,
            },
        )
        trace.save_text("02_ocr.txt", full_text)
        # Keep geometry + OCR confidence: the LLM resolves layout from them.
        ocr_pages = [
            {
                "page_index": p.get("page_index", i),
                "text": p.get("text", ""),
                "lines": [
                    {"text": ln.get("text", ""), "score": ln.get("score"), "box": ln.get("box")}
                    for ln in p.get("lines", [])
                ],
            }
            for i, p in enumerate(pages)
        ]
        log.info("[extract/invoice] llm starting model=%s ...", llm_client.model_name())
        t_llm = time.perf_counter()
        try:
            response = await run_in_threadpool(
                pipeline.extract_from_pages,
                ocr_pages,
                field_mappings,
                ref,
                document_type,
                do_match,
                po_rows,
                conditions,
                generic_extraction_hint,
                prior,
                trace,
            )
        except llm_client.LLMNotConfigured as exc:
            trace.save_json("error.json", {"stage": "llm", "error": str(exc)})
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except llm_client.LLMError as exc:
            trace.save_json(
                "error.json",
                {"stage": "llm", "error": str(exc), "raw": getattr(exc, "raw", None)},
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        log_stage(
            "extract/invoice",
            "llm",
            t_llm,
            headers=len(response.key_value_pairs),
            rows=len(response.tables),
        )

    total = round(time.time() - t0, 3)
    trace.save_json(
        "08_meta.json",
        {"unique_ref_no": ref, "total_s": total, "model": llm_client.model_name()},
    )
    log.info("[extract/invoice] request done ref=%s total=%.1fs", ref, total)
    return response
