"""Vision message builder for direct-VLM extraction.

Pure functions: given an :class:`ExtractSpec` and rendered :class:`PageImage`
pages they return OpenAI-style vision chat messages. No network, no side
effects — safe to test for free.

Unlike :mod:`app.prompt` (which serializes OCR text), the document here is
the pixels themselves, so the text part only carries the field list plus
caller context; one ``image_url`` part per page follows.
"""

from __future__ import annotations

import json
from typing import Any

from app.fields import ExtractSpec, FieldSpec
from app.services.images import PageImage

SYSTEM_PROMPT_VISION = """\
You extract structured fields from an invoice or business document.
You are given the document as page images plus a list of fields to find. \
Each field has a unique key, the label as it appears in the document, a \
datatype, and an optional hint. Read the values directly from the pixels.

Rules:
1. Use ONLY what is visible in the page images. Never invent values from training knowledge.
2. If a field is not visible and its hint does not let you derive it, return null for that key.
3. Follow each hint literally. Hints may ask you to compute a value, e.g. \
"Payment Terms 30 days" means: read the invoice date, add 30 days, and return \
the resulting exact date.
4. Dates must be ISO YYYY-MM-DD. Numbers must be plain (no currency symbols, \
no thousand separators). Booleans are true/false.
5. Line items: return one object per table row, in document order. If no table \
is found, return an empty list.
6. Reply with JSON ONLY, shaped as {"headers": {...}, "line_items": [...]}, \
using exactly the keys listed. No explanations outside the JSON.\
"""


def _describe(field: FieldSpec) -> str:
    line = f'- key "{field.key}": document field "{field.field}" ({field.type.value})'
    if field.hint:
        line += f". Hint: {field.hint}"
    return line


def build_vision_text(
    spec: ExtractSpec,
    document_type: str | None = None,
    generic_hint: str | None = None,
    extraction_json: dict[str, Any] | None = None,
    page_count: int = 1,
) -> str:
    """Render the spec + caller context into the single text part."""
    parts = [
        "Extract exactly these fields from the attached page image(s). Reply with JSON only.",
        "",
        "HEADERS (one value per key):",
        *(_describe(f) for f in spec.headers),
    ]
    if spec.line_items:
        parts += [
            "",
            "LINE ITEMS (one object per table row, same keys each row):",
            *(_describe(f) for f in spec.line_items),
        ]
    else:
        parts += ["", "LINE ITEMS: none requested, return an empty list."]
    extras: list[str] = []
    if document_type:
        extras.append(f"DOCUMENT TYPE: {document_type}")
    if generic_hint:
        extras.append(f"VENDOR/SYSTEM HINT: {generic_hint}")
    if extraction_json is not None:
        extras.append(
            "PRIOR EXTRACTION (re-extract or correct it based on the page images):\n"
            + json.dumps(extraction_json)[:4000]
        )
    if page_count > 1:
        extras.append(
            f"NOTE: the document has {page_count} pages, attached in order. "
            "Rows and values may span pages; read all of them."
        )
    if extras:
        parts += ["", "ADDITIONAL CONTEXT:", *extras]
    return "\n".join(parts)


def build_vision_messages(
    spec: ExtractSpec,
    pages: list[PageImage],
    document_type: str | None = None,
    generic_hint: str | None = None,
    extraction_json: dict[str, Any] | None = None,
) -> list[dict]:
    """Build ``[{role, content}]`` vision messages for the VLM client."""
    text = build_vision_text(spec, document_type, generic_hint, extraction_json, len(pages))
    content: list[dict] = [{"type": "text", "text": text}]
    for p in pages:
        content.append({"type": "image_url", "image_url": {"url": p.data_url}})
    return [
        {"role": "system", "content": SYSTEM_PROMPT_VISION},
        {"role": "user", "content": content},
    ]


def redact_for_trace(messages: list[dict], preview_chars: int = 120) -> list[dict]:
    """Trace-safe copy of vision messages with base64 payloads truncated."""
    redacted: list[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            parts: list[dict] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = ((part.get("image_url") or {}).get("url")) or ""
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (url[:preview_chars] + "..." if len(url) > preview_chars else url),
                                "url_chars": len(url),
                            },
                        }
                    )
                else:
                    parts.append(part)
            redacted.append({"role": m.get("role"), "content": parts})
        else:
            redacted.append(m)
    return redacted
