"""Direct-VLM invoice extraction endpoint (no OCR).

``POST /extract/invoice/vlm`` accepts the same multipart contract as
``POST /extract/invoice`` (minus ``page_ranges`` — all pages are always read)
and returns :class:`ProcessedInvoiceResponse` directly. ``callback_url`` is
accepted for contract compatibility but ignored.

Pipeline: upload bytes → :mod:`app.services.images` (PDF rasterize / resize /
data-URL) → vision LLM (:mod:`app.infra.vlm_client`) → coerce → local PO
match. PaddleOCR is never touched on this path.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.deps import read_validated_upload
from app.api.v1 import forms
from app.core.config import get_settings
from app.core.debug_trace import new_trace
from app.core.logging import get_logger, log_stage
from app.dtos.invoice import (
    FieldMapping,
    MatchingCondition,
    ProcessedInvoiceResponse,
)
from app.infra import vlm_client
from app.services import vlm_pipeline
from app.services.images import ImageRenderError, TooManyPagesError, describe_for_trace, render_pages

log = get_logger()
router = APIRouter(tags=["extract"])


def _render_all_pages(blob: bytes, suffix: str):
    """Render every page of the upload. Maps render errors to HTTP codes."""
    settings = get_settings()
    try:
        return render_pages(
            blob,
            suffix,
            max_pages=settings.vlm_max_pages,
            max_side_px=settings.vlm_max_side_px,
            jpeg_quality=settings.vlm_jpeg_quality,
        )
    except TooManyPagesError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImageRenderError as exc:
        raise HTTPException(status_code=500, detail=f"Image rendering failed: {exc}") from exc


@router.post("/extract/invoice/vlm", response_model=ProcessedInvoiceResponse)
async def extract_invoice_vlm(
    file: UploadFile = File(..., description="Invoice document (PDF/image, all pages are read)"),
    unique_ref_no: str = Form(..., description="Invoice id as string"),
    fields: str = Form(..., description="Field mappings JSON array"),
    match: str = Form(..., description="JSON bool string: 'true' or 'false'"),
    callback_url: Optional[str] = Form(default=None, description="Accepted but ignored (sync mode)"),
    document_type: Optional[str] = Form(default=None),
    po_items: Optional[str] = Form(default=None),
    matching_conditions: Optional[str] = Form(default=None),
    extraction_json: Optional[str] = Form(default=None),
    generic_extraction_hint: Optional[str] = Form(default=None),
) -> ProcessedInvoiceResponse:
    """Sync direct-vision extraction: page images -> VLM -> match, returned inline."""
    t0 = time.time()
    ref = forms.validated_ref(unique_ref_no)

    fields_raw = forms.parse_json_form(fields, "fields", required=True)
    field_mappings = forms.validated_fields(fields_raw)
    do_match = forms.parse_match_flag(match)

    po_rows: list[dict[str, Any]] | None = None
    conditions: list[MatchingCondition] | None = None
    if do_match:
        po_rows = forms.validated_po_items(
            forms.parse_json_form(po_items, "po_items", required=True)
        )
        conditions = forms.validated_conditions(
            forms.parse_json_form(matching_conditions, "matching_conditions", required=True),
            field_mappings,
        )

    prior = forms.validated_prior_extraction(extraction_json)

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
            "mode": "vlm-direct",
            "model": vlm_client.model_name(),
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
        "[extract/invoice/vlm] request ref=%s file=%s size_kb=%d fields=%d match=%s trace=%s",
        ref,
        file.filename,
        len(blob) // 1024,
        len(field_mappings),
        do_match,
        trace.dir,
    )

    t_render = time.perf_counter()
    try:
        pages = await run_in_threadpool(_render_all_pages, blob, suffix)
    except HTTPException as exc:
        trace.save_json("error.json", {"stage": "render", "detail": exc.detail})
        raise
    log_stage("extract/invoice/vlm", "render", t_render, pages=len(pages))
    trace.save_json("02_vlm_input.json", {"pages": describe_for_trace(pages)})

    log.info("[extract/invoice/vlm] vlm starting model=%s pages=%d ...", vlm_client.model_name(), len(pages))
    t_vlm = time.perf_counter()
    try:
        response = await run_in_threadpool(
            vlm_pipeline.extract_invoice_vlm,
            pages,
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
    except vlm_client.LLMNotConfigured as exc:
        trace.save_json("error.json", {"stage": "vlm", "error": str(exc)})
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except vlm_client.LLMError as exc:
        trace.save_json(
            "error.json",
            {"stage": "vlm", "error": str(exc), "raw": getattr(exc, "raw", None)},
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    log_stage(
        "extract/invoice/vlm",
        "vlm",
        t_vlm,
        headers=len(response.key_value_pairs),
        rows=len(response.tables),
    )

    total = round(time.time() - t0, 3)
    trace.save_json(
        "08_meta.json",
        {
            "unique_ref_no": ref,
            "total_s": total,
            "model": vlm_client.model_name(),
            "mode": "vlm-direct",
            "pages": len(pages),
        },
    )
    log.info("[extract/invoice/vlm] request done ref=%s total=%.1fs", ref, total)
    return response
