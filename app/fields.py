"""Caller-supplied field specs for guided extraction.

The caller tells us *what* to extract (document label, datatype, unique key,
optional hint). The LLM figures out *how* from the OCR text. Values come back
keyed by ``key`` so the caller can map them straight into their own system.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_FIELDS = 50
KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class FieldType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    DATE = "date"
    BOOLEAN = "boolean"


class FieldSpec(BaseModel):
    """One extractable field.

    key:   unique response key, e.g. "due_date".
    field: label as it appears in the document, e.g. "Due Date".
    type:  one of string/number/integer/date/boolean.
    hint:  free-text extraction help, e.g. "Payment Terms 30 days means
           Invoice Date + 30 days; return the exact ISO date".
    """

    key: str
    field: str = Field(min_length=1, max_length=200)
    type: FieldType = FieldType.STRING
    hint: str = Field(default="", max_length=1000)

    @field_validator("key")
    @classmethod
    def _valid_key(cls, v: str) -> str:
        if not KEY_PATTERN.match(v):
            raise ValueError(
                "key must start with a letter and contain only "
                "letters, digits, underscores (max 64 chars)"
            )
        return v


class ExtractSpec(BaseModel):
    """Full extraction request: header fields + repeating line-item fields."""

    headers: list[FieldSpec] = Field(default_factory=list)
    line_items: list[FieldSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_spec(self) -> "ExtractSpec":
        total = len(self.headers) + len(self.line_items)
        if total == 0:
            raise ValueError("spec must define at least one header or line-item field")
        if total > MAX_FIELDS:
            raise ValueError(f"spec defines {total} fields, max is {MAX_FIELDS}")
        for group in ("headers", "line_items"):
            keys = [f.key for f in getattr(self, group)]
            dupes = {k for k in keys if keys.count(k) > 1}
            if dupes:
                raise ValueError(f"duplicate keys in {group}: {sorted(dupes)}")
        return self


# Re-exported for convenience so callers import from one place.
FieldTypeName = Literal["string", "number", "integer", "date", "boolean"]
