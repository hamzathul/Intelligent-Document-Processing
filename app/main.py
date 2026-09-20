"""Application entrypoint: app factory + legacy routes.

Legacy ``/api/ocr`` and ``/api/extract`` are kept for backward compatibility.
New contract lives in :mod:`app.api.v1.invoice` (``POST /extract/invoice``
+ ``POST /api/v1/extract/invoice``) which returns the MD §3 payload
synchronously instead of POSTing to ``callback_url``.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app import extract_service, llm_client, ocr_service
from app.api.deps import read_validated_upload
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, log_stage
from app.fields import ExtractSpec
from app.schemas import ExtractResponse, HealthResponse, OcrLine, OcrPage, OcrResponse
from app.utils.files import saved_temp_file

log = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Warm up the model so first request isn't cold; keep serving even on failure.
    try:
        await run_in_threadpool(ocr_service.get_ocr)
        log.info("PP-OCRv6 warmed up")
    except Exception as exc:  # noqa: BLE001
        log.warning("PP-OCRv6 preload failed (lazy-load on first request): %s", exc)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="IDP OCR + Invoice Extraction", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    # Core contract path (no prefix) + versioned alias.
    app.include_router(api_router)
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "idp-ocr",
        "model": ocr_service.MODEL_NAME,
        "docs": "/docs",
        "health": "/api/health",
        "ocr": "POST /api/ocr (multipart field: file)",
        "extract": "POST /api/extract (multipart fields: file + spec JSON)",
        "extract_invoice": "POST /extract/invoice (sync, multipart: file + unique_ref_no + fields + match)",
        "extract_invoice_v1": "POST /api/v1/extract/invoice (sync alias)",
    }


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok",
        model=ocr_service.MODEL_NAME,
        engine=ocr_service.ENGINE,
        device=ocr_service.DEVICE,
        ocr_loaded=ocr_service.is_loaded(),
    )


async def _ocr_pages(
    tmp_path: str, page_ranges: str | None, route: str = "ocr"
) -> tuple[list[OcrPage], str]:
    """Run PP-OCRv6 on a stored file. Returns (pages, full_text)."""
    log.info("[%s] ocr starting (PP-OCRv6 local inference)...", route)
    t0 = time.perf_counter()
    try:
        raw = await run_in_threadpool(ocr_service.predict_path, tmp_path, page_ranges)
        parsed = ocr_service.parse_result(raw)
    except Exception as exc:  # noqa: BLE001
        log.exception("[%s] ocr FAILED after %.1fs", route, time.perf_counter() - t0)
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc
    pages = []
    full_parts = []
    for p in parsed:
        lines = [OcrLine(**ln) for ln in p["lines"]]
        mean = sum(ln.score for ln in lines) / len(lines) if lines else 0.0
        pages.append(
            OcrPage(page_index=p["page_index"], text=p["text"], lines=lines, mean_score=mean)
        )
        if p["text"]:
            full_parts.append(p["text"])
    return pages, "\n\n".join(full_parts)


@app.post("/api/ocr", response_model=OcrResponse, tags=["ocr"])
async def ocr_extract(
    file: UploadFile = File(..., description="Image or PDF document"),
    page_ranges: str | None = Query(
        default=None,
        description='PDF pages, e.g. "1-3,5". Omit for all pages.',
    ),
):
    settings = get_settings()
    suffix, blob = await read_validated_upload(file, settings)
    log.info("[ocr] request file=%s size_kb=%d", file.filename, len(blob) // 1024)
    t0 = time.time()
    with saved_temp_file(suffix, blob) as tmp_path:
        t_ocr = time.perf_counter()
        pages, full_text = await _ocr_pages(tmp_path, page_ranges, route="ocr")
        log_stage("ocr", "ocr", t_ocr, pages=len(pages), chars=len(full_text))
    total = round(time.time() - t0, 3)
    log.info("[ocr] request done file=%s total=%.1fs", file.filename, total)
    return OcrResponse(
        filename=file.filename or "upload",
        model=ocr_service.MODEL_NAME,
        engine=ocr_service.ENGINE,
        pages=pages,
        full_text=full_text,
        time_s=round(time.time() - t0, 3),
    )


def _parse_spec(spec_raw: str) -> ExtractSpec:
    try:
        spec_json = json.loads(spec_raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"spec is not valid JSON: {exc}") from exc
    try:
        return ExtractSpec.model_validate(spec_json)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"invalid spec: {exc}") from exc


@app.post("/api/extract", response_model=ExtractResponse, tags=["extract"])
async def guided_extract(
    file: UploadFile = File(..., description="Invoice image or PDF"),
    spec: str = Form(..., description="Extraction spec JSON: {headers:[...], line_items:[...]}"),
    page_ranges: str | None = Query(
        default=None,
        description='PDF pages, e.g. "1-3,5". Omit for all pages.',
    ),
):
    """Extract caller-defined fields: OCR the doc, then let the LLM fill your keys."""
    spec_obj = _parse_spec(spec)
    settings = get_settings()
    suffix, blob = await read_validated_upload(file, settings)
    n_fields = len(spec_obj.headers) + len(spec_obj.line_items)
    log.info(
        "[extract] request file=%s size_kb=%d fields=%d (headers=%d items=%d)",
        file.filename,
        len(blob) // 1024,
        n_fields,
        len(spec_obj.headers),
        len(spec_obj.line_items),
    )
    t0 = time.time()
    with saved_temp_file(suffix, blob) as tmp_path:
        t_ocr = time.perf_counter()
        pages, _ = await _ocr_pages(tmp_path, page_ranges, route="extract")
        log_stage("extract", "ocr", t_ocr, pages=len(pages))
        plain_pages = [
            {
                "page_index": p.page_index,
                "text": p.text,
                "lines": [
                    {"text": ln.text, "score": ln.score, "box": ln.box} for ln in p.lines
                ],
            }
            for p in pages
        ]
        log.info(
            "[extract] llm starting model=%s fields=%d...",
            llm_client.model_name(),
            n_fields,
        )
        t_llm = time.perf_counter()
        try:
            result = await run_in_threadpool(
                extract_service.extract_from_pages, plain_pages, spec_obj
            )
        except llm_client.LLMNotConfigured as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except llm_client.LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        log_stage(
            "extract",
            "llm",
            t_llm,
            headers=len(result["headers"]),
            rows=len(result["line_items"]),
        )
    total = round(time.time() - t0, 3)
    log.info("[extract] request done file=%s total=%.1fs", file.filename, total)
    return ExtractResponse(
        filename=file.filename or "upload",
        headers=result["headers"],
        line_items=result["line_items"],
        ocr_text=result["ocr_text"],
        model=result["model"],
        time_s=total,
    )


def main() -> None:
    """Easy start entrypoint: ``uv run idp``.

    Reads ``HOST``/``PORT``/``RELOAD`` env vars (with sensible defaults)
    so there is no long uvicorn command to remember.
    """
    import argparse
    import os

    import uvicorn

    parser = argparse.ArgumentParser(description="Start the IDP OCR + Invoice API")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("RELOAD", "").lower() in {"1", "true", "yes"},
    )
    args = parser.parse_args()

    print(f"Serving IDP OCR at http://{args.host}:{args.port}  (docs: /docs)")
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
