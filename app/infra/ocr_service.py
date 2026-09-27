"""Singleton wrapper around PaddleOCR PP-OCRv6 local inference."""

from __future__ import annotations

import logging
import os
import threading
import time

log = logging.getLogger("uvicorn.error")

_lock = threading.Lock()
_ocr = None
_load_error: str | None = None

MODEL_NAME = os.getenv("OCR_MODEL", "PP-OCRv6_medium")
ENGINE = os.getenv("OCR_ENGINE", "paddle")
DEVICE = os.getenv("OCR_DEVICE", "cpu")


def get_ocr():
    """Lazily create and return the shared PaddleOCR instance."""
    global _ocr, _load_error
    if _ocr is not None:
        return _ocr
    with _lock:
        if _ocr is not None:
            return _ocr
        from paddleocr import PaddleOCR

        cpu_threads = int(os.getenv("OCR_CPU_THREADS", "4"))
        log.info("Loading PaddleOCR model=%s engine=%s device=%s", MODEL_NAME, ENGINE, DEVICE)
        t0 = time.time()
        try:
            _ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=DEVICE,
                cpu_threads=cpu_threads,
            )
        except TypeError:
            # Older paddleocr builds don't accept device/cpu_threads.
            _ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        log.info("PaddleOCR loaded in %.1fs", time.time() - t0)
        return _ocr


def is_loaded() -> bool:
    return _ocr is not None


def load_error() -> str | None:
    return _load_error


def predict_path(file_path: str, page_ranges: str | None = None):
    """Run OCR on an image or PDF path. Returns raw PaddleOCR result list."""
    ocr = get_ocr()
    kwargs: dict = {}
    if page_ranges:
        kwargs["page_ranges"] = page_ranges
    log.info("PP-OCRv6 inference starting...")
    t0 = time.time()
    try:
        result = ocr.predict(file_path, **kwargs)
    except TypeError:
        # predict() without page_ranges support
        result = ocr.predict(file_path)
    log.info("PP-OCRv6 inference done in %.1fs", time.time() - t0)
    return result


def parse_result(raw_result) -> list[dict]:
    """Normalize raw PaddleOCR 3.x output to [{page_index, lines, text}]."""
    pages: list[dict] = []
    if not raw_result:
        return pages

    for page_idx, res in enumerate(raw_result):
        data: dict = {}
        # OCRResult objects expose .to_dict() / .json in 3.x; plain dicts otherwise.
        if hasattr(res, "to_dict"):
            try:
                data = res.to_dict() or {}
            except Exception:
                data = {}
        elif hasattr(res, "json") and isinstance(getattr(res, "json"), dict):
            data = res.json  # type: ignore[assignment]
        elif isinstance(res, dict):
            data = res
        elif hasattr(res, "__dict__"):
            data = dict(getattr(res, "__dict__", {}) or {})

        # Nested shapes: some builds wrap payload under keys.
        for wrapper in ("res", "result", "data", "prunedResult"):
            if isinstance(data, dict) and wrapper in data and isinstance(data[wrapper], dict):
                inner = data[wrapper]
                if "rec_texts" in inner:
                    data = inner
                    break

        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        # PP-OCRv6 shapes: rec_polys/dt_polys are (N,4,2) quadrilaterals,
        # rec_boxes is an (N,4) ndarray in xyxy form (ambiguous truthiness!).
        polys = data.get("rec_polys")
        if polys is None:
            polys = data.get("dt_polys")
        boxes_xyxy = data.get("rec_boxes")

        def _as_list(v):
            if v is None:
                return []
            try:
                return list(v)
            except TypeError:
                return []

        polys = _as_list(polys)
        boxes_xyxy = _as_list(boxes_xyxy)

        def _poly_to_box(poly) -> list[list[float]] | None:
            try:
                pts = [[float(c) for c in pt] for pt in poly]
                if len(pts) == 4 and all(len(p) == 2 for p in pts):
                    return pts
            except Exception:
                pass
            return None

        def _xyxy_to_box(row) -> list[list[float]] | None:
            try:
                x1, y1, x2, y2 = (float(v) for v in list(row)[:4])
                return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            except Exception:
                return None

        lines: list[dict] = []
        for i, text in enumerate(texts):
            try:
                score = float(scores[i]) if i < len(scores) else 0.0
            except Exception:
                score = 0.0
            box = _poly_to_box(polys[i]) if i < len(polys) else None
            if box is None and i < len(boxes_xyxy):
                box = _xyxy_to_box(boxes_xyxy[i])
            lines.append({"text": str(text), "score": score, "box": box})

        full = "\n".join(str(t) for t in texts)
        pages.append({"page_index": page_idx, "lines": lines, "text": full})

    return pages
