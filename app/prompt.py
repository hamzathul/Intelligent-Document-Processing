"""Prompt + JSON-schema builders for guided extraction.

Pure functions: given a spec and OCR text they return chat messages and the
response schema. No network, no side effects — safe to test for free.
"""

from __future__ import annotations

from app.fields import ExtractSpec, FieldSpec, FieldType

MAX_OCR_CHARS = 12_000

#: OCR lines with a score below this are flagged `[!]` (uncertain reading).
LOW_CONF_THRESHOLD = 0.70

SYSTEM_PROMPT = """\
You extract structured fields from an invoice or business document.
You are given the document text (from OCR, top-to-bottom reading order) and a \
list of fields to find. Each field has a unique key, the label as it appears \
in the document, a datatype, and an optional hint.

Rules:
1. Use ONLY the document text. Never invent values from training knowledge.
2. If a field is not present in the document and its hint does not let you \
derive it, return null for that key.
3. Follow each hint literally. Hints may ask you to compute a value, e.g. \
"Payment Terms 30 days" means: read the invoice date, add 30 days, and return \
the resulting exact date. Show the reasoning briefly, then the value.
4. Dates must be ISO YYYY-MM-DD. Numbers must be plain (no currency symbols, \
no thousand separators). Booleans are true/false.
5. Line items: return one object per table row, in document order. If no table \
is found, return an empty list.
6. Reply with JSON ONLY, shaped as {"headers": {...}, "line_items": [...]}, \
using exactly the keys listed. No explanations outside the JSON.\
"""

SYSTEM_PROMPT_LAYOUT_EXTRA = """\

Layout readings: each document line looks like
"L012 x=1452 y=118 w=312 h=44 conf=0.99 | <text>".
x/y/w/h is the OCR box in pixels from the page top-left; conf is OCR
confidence (0-1); [!] marks an uncertain reading that may contain character
errors. Use the geometry: a value usually sits on the same row (similar y) to
the right of its label, and table cells align in columns (similar x) and rows
(similar y). When readings conflict, prefer high-confidence lines.\
"""

_TYPE_JSON = {
    FieldType.STRING: {"type": ["string", "null"]},
    FieldType.NUMBER: {"type": ["number", "null"]},
    FieldType.INTEGER: {"type": ["number", "null"]},
    FieldType.DATE: {"type": ["string", "null"], "description": "ISO date YYYY-MM-DD or null"},
    FieldType.BOOLEAN: {"type": ["boolean", "null"]},
}


def _describe(field: FieldSpec) -> str:
    line = f'- key "{field.key}": document field "{field.field}" ({field.type.value})'
    if field.hint:
        line += f". Hint: {field.hint}"
    return line


def _box_xywh(box: object) -> tuple[int, int, int, int] | None:
    """Reduce a quad ``[[x,y] x4]`` to integer ``(x, y, w, h)``. None if unusable."""
    try:
        pts = [(float(p[0]), float(p[1])) for p in box]  # type: ignore[union-attr]
    except (TypeError, ValueError, IndexError):
        return None
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    return (round(x0), round(y0), round(max(xs) - x0), round(max(ys) - y0))


def _line_confidence(line: dict) -> float | None:
    try:
        score = float(line.get("score"))  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    if score < 0.0 or score > 1.0:
        return None
    return score


def render_layout_text(pages: list[dict] | None) -> str:
    """Render OCR pages as position+confidence annotated lines.

    Each line: ``L001 x=.. y=.. w=.. h=.. conf=0.99 | <text>`` (coords/conf
    omitted when the OCR result lacks them). Blank lines are dropped and
    output is truncated to whole lines within :data:`MAX_OCR_CHARS`.
    PaddleOCR detection order (top-to-bottom) is preserved.
    """
    rendered: list[str] = []
    seq = 0
    for page in pages or []:
        block: list[str] = []
        lines = page.get("lines") or []
        if not lines and page.get("text"):
            for raw in str(page["text"]).splitlines():
                if raw.strip():
                    seq += 1
                    block.append(f"L{seq:03d} | {raw.strip()}")
        else:
            for line in lines:
                if not isinstance(line, dict):
                    continue
                text = str(line.get("text", "")).strip()
                if not text:
                    continue
                seq += 1
                tags: list[str] = []
                xywh = _box_xywh(line.get("box"))
                if xywh is not None:
                    tags.append(f"x={xywh[0]} y={xywh[1]} w={xywh[2]} h={xywh[3]}")
                conf = _line_confidence(line)
                if conf is not None:
                    tags.append(f"conf={conf:.2f}" + (" [!]" if conf < LOW_CONF_THRESHOLD else ""))
                prefix = f"L{seq:03d}" + (f" {' '.join(tags)}" if tags else "")
                block.append(f"{prefix} | {text}")
        if block:
            rendered.append(f"[page {page.get('page_index', 0)}]")
            rendered.extend(block)
    if not rendered:
        return ""
    text = "\n".join(rendered)
    if len(text) <= MAX_OCR_CHARS:
        return text
    # Truncate to whole trailing lines so no line (and its geometry) is cut mid-way.
    while rendered and len("\n".join(rendered)) > MAX_OCR_CHARS:
        rendered.pop()
    if not rendered:
        return f"[document truncated to {MAX_OCR_CHARS} chars]"
    return "\n".join(rendered) + f"\n[document truncated to {MAX_OCR_CHARS} chars]"


def build_user_message(
    spec: ExtractSpec, ocr_text: str = "", *, layout: str | None = None
) -> str:
    """Render the spec + document text into the user message.

    Pass ``layout`` (see :func:`render_layout_text`) to send position and
    OCR-confidence annotated lines; otherwise plain ``ocr_text`` is used.
    """
    parts = [
        "Extract exactly these fields. Reply with JSON only.",
        "",
        "HEADERS (one value per key):",
        *( _describe(f) for f in spec.headers ),
    ]
    if spec.line_items:
        parts += [
            "",
            "LINE ITEMS (one object per table row, same keys each row):",
            *( _describe(f) for f in spec.line_items ),
        ]
    else:
        parts += ["", "LINE ITEMS: none requested, return an empty list."]
    if layout is not None:
        parts += ["", "DOCUMENT TEXT (OCR with positions + confidence):", layout]
    else:
        # Drop blank OCR lines (empty detection boxes) to cut noise and tokens.
        cleaned = "\n".join(line for line in ocr_text.splitlines() if line.strip())
        text = cleaned if len(cleaned) <= MAX_OCR_CHARS else cleaned[:MAX_OCR_CHARS]
        parts += ["", "DOCUMENT TEXT (OCR, top-to-bottom):", text]
        if len(cleaned) > MAX_OCR_CHARS:
            parts.append(f"[document truncated to {MAX_OCR_CHARS} chars]")
    return "\n".join(parts)


def build_messages(
    spec: ExtractSpec, ocr_text: str = "", *, layout: str | None = None
) -> list[dict]:
    system = SYSTEM_PROMPT + (SYSTEM_PROMPT_LAYOUT_EXTRA if layout is not None else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_message(spec, ocr_text, layout=layout)},
    ]


def build_json_schema(spec: ExtractSpec) -> dict:
    """Dynamic response schema so the model must reply with our exact keys."""

    def _props(fields: list[FieldSpec]) -> dict:
        return {f.key: _TYPE_JSON[f.type] for f in fields}

    header_keys = [f.key for f in spec.headers]
    item_keys = [f.key for f in spec.line_items]
    return {
        "type": "object",
        "properties": {
            "headers": {
                "type": "object",
                "properties": _props(spec.headers),
                "required": header_keys,
                "additionalProperties": False,
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": _props(spec.line_items),
                    "required": item_keys,
                    "additionalProperties": False,
                },
            },
        },
        "required": ["headers", "line_items"],
        "additionalProperties": False,
    }
