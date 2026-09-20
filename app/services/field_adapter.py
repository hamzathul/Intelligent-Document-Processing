"""Adapter: MD ``FieldMapping`` list -> internal :class:`ExtractSpec`.

MD sends ``unique_key / display_name / extract_hint / invoice_field /
document_field / data_type / category`` (``transform_rule`` is accepted for
core compatibility but ignored); the existing extraction engine understands
``key / field / type / hint`` grouped into ``headers`` vs ``line_items``.
This module bridges the two without touching the engine.
"""

from __future__ import annotations

from app.dtos.invoice import FieldMapping
from app.fields import ExtractSpec, FieldSpec, FieldType

_DATA_TYPE_MAP: dict[str, FieldType] = {
    "string": FieldType.STRING,
    "number": FieldType.NUMBER,
    "integer": FieldType.INTEGER,
    "date": FieldType.DATE,
    "boolean": FieldType.BOOLEAN,
}


def _label(f: FieldMapping) -> str:
    return f.document_field or f.display_name or f.invoice_field or f.unique_key


def adapt_fields(fields: list[FieldMapping]) -> ExtractSpec:
    """Split MD fields by ``category`` into an :class:`ExtractSpec`."""
    headers: list[FieldSpec] = []
    line_items: list[FieldSpec] = []
    for f in fields:
        spec = FieldSpec(
            key=f.unique_key,
            field=_label(f)[:200] or f.unique_key,
            type=_DATA_TYPE_MAP.get(f.data_type, FieldType.STRING),
            hint=f.extract_hint or "",
        )
        if f.category == "line_item":
            line_items.append(spec)
        else:
            headers.append(spec)
    return ExtractSpec(headers=headers, line_items=line_items)


def apply_transform_rule(value: object, rule: str | None = None) -> object:
    """Deprecated no-op kept for backward compatibility.

    ``transform_rule`` is not needed: values are coerced purely by
    ``data_type`` (string/number/date/boolean). The argument is accepted and
    ignored so callers/core payloads sending it keep working.
    """
    return value


def field_index(fields: list[FieldMapping]) -> dict[str, FieldMapping]:
    """Index by unique_key; line_item entries win on header/line_item reuse."""
    index: dict[str, FieldMapping] = {}
    for f in fields:
        if f.unique_key not in index or f.category == "line_item":
            index[f.unique_key] = f
    return index


def document_label(mapping: FieldMapping | None, fallback_key: str) -> str:
    if mapping is None:
        return fallback_key
    return mapping.document_field or mapping.display_name or mapping.invoice_field or fallback_key
