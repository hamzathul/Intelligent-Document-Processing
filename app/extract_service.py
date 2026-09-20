"""Guided extraction orchestrator: OCR pages -> prompt -> LLM -> coerce.

One public function, three steps. The LLM *proposes* values for the caller's
keys; :mod:`app.coerce` *disposes* (validates them into declared types).
"""

from __future__ import annotations

from app import coerce, llm_client, prompt
from app.fields import ExtractSpec

ExtractedValue = str | float | int | bool | None


def extract_from_pages(pages: list[dict], spec: ExtractSpec) -> dict:
    """Run guided extraction over already-OCR'd pages.

    pages: [{"page_index": int, "lines": [{"text","score","box",..}], "text": str}, ...]
    Pages carrying OCR lines are sent position+confidence annotated; pages
    with text only fall back to the plain-text prompt.
    Returns {"headers": {...}, "line_items": [{...}], "ocr_text": str, "model": str}.
    Raises llm_client.LLMNotConfigured / llm_client.LLMError.
    """
    ocr_text = "\n".join(p.get("text", "") for p in pages).strip()
    has_lines = any(isinstance(p.get("lines"), list) and p["lines"] for p in pages)
    layout = prompt.render_layout_text(pages) if has_lines else None
    messages = prompt.build_messages(spec, ocr_text, layout=layout or None)
    schema = prompt.build_json_schema(spec)
    raw = llm_client.complete_json(messages, schema)

    raw_headers = raw.get("headers") or {}
    headers: dict[str, ExtractedValue] = {
        f.key: coerce.apply_type(raw_headers.get(f.key), f.type) for f in spec.headers
    }

    line_items: list[dict[str, ExtractedValue]] = []
    if spec.line_items:
        raw_rows = raw.get("line_items") or []
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            line_items.append(
                {f.key: coerce.apply_type(row.get(f.key), f.type) for f in spec.line_items}
            )

    return {
        "headers": headers,
        "line_items": line_items,
        "ocr_text": ocr_text,
        "model": llm_client.model_name(),
    }
