"""Invoice extraction DTOs mirroring `.temp/INVOICE_EXTRACTION_API.md`.

The sync endpoint returns :class:`ProcessedInvoiceResponse` directly
(same shape core used to receive via ``callback_url``).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

DataTypeName = Literal["string", "number", "date", "boolean", "integer"]
CategoryName = Literal["header", "line_item"]
MatchTypeName = Literal["exact", "partial", "fuzzy", "numeric"]


class FieldMapping(BaseModel):
    """One entry of the ``fields`` JSON array sent by core."""

    unique_key: str
    display_name: str = ""
    extract_hint: str | None = None
    transform_rule: str | None = None  # accepted for core compat, ignored (not needed)
    invoice_field: str | None = None
    document_field: str | None = None
    data_type: DataTypeName = "string"
    category: CategoryName = "header"

    @field_validator("unique_key")
    @classmethod
    def _valid_key(cls, v: str) -> str:
        if not _KEY_PATTERN.match(v):
            raise ValueError(
                "unique_key must start with a letter and contain only "
                "letters, digits, underscores (max 64 chars)"
            )
        return v


class PoItem(BaseModel):
    """Reference PO line item. Known columns are optional; extras allowed."""

    model_config = {"extra": "allow"}

    item_number: str | None = None
    material_number: str | None = None
    material_description: str | None = None
    quantity: float | int | str | None = None
    unit_price: float | int | str | None = None
    total_price: float | int | str | None = None
    unit_of_measure: str | None = None
    delivery_date: str | None = None
    net_value: float | int | str | None = None
    material_group: str | None = None
    tax_amount: float | int | str | None = None
    goods_delivered_quantity: float | int | str | None = None
    open_invoice_quantity: float | int | str | None = None
    po_number: str | None = None

    def get_value(self, field: str) -> Any:
        if field in self.model_fields_set or field in type(self).model_fields:
            return getattr(self, field, None)
        return (self.model_extra or {}).get(field)


class MatchingCondition(BaseModel):
    priority: int
    po_field: str
    invoice_field_key: str
    match_type: MatchTypeName
    description: str | None = None


class KeyValuePair(BaseModel):
    key: str
    value: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class TableCell(BaseModel):
    key: str
    value: str | None = None


class MatchResult(BaseModel):
    score: float = Field(ge=0.0, le=5.0)
    best_match: str | None = None
    invoice_item: str | None = None
    po_usage_count: int = 1
    po_field: str
    invoice_field: str
    invoice_document_field: str | None = None
    match_type: str
    matching_method: str
    priority: int


class ProcessedInvoiceResponse(BaseModel):
    """Sync response for ``POST /extract/invoice`` (MD §3 shape)."""

    unique_ref_no: str
    status: Literal["success", "failed"] = "success"
    key_value_pairs: list[KeyValuePair] = Field(default_factory=list)
    tables: list[list[TableCell]] = Field(default_factory=list)
    match_results: list[MatchResult] = Field(default_factory=list)
    intermediate_data: dict[str, Any] | None = None
