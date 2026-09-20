"""Offline tests for sync `POST /extract/invoice` (no OCR model / LLM key needed)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app import llm_client, ocr_service
from app.dtos.invoice import FieldMapping
from app.main import app
from app.services import field_adapter, matcher, pipeline
from app.services.matcher import match_invoice_rows


def _stub_ocr(monkeypatch, texts=("INV-2026-0091",)):
    monkeypatch.setattr(
        ocr_service,
        "predict_path",
        lambda *a, **k: [
            {
                "rec_texts": list(texts),
                "rec_scores": [0.99] * len(texts),
                "rec_polys": [],
                "rec_boxes": [],
            }
        ],
    )


def _stub_llm(monkeypatch, payload: dict):
    monkeypatch.setattr(llm_client, "complete_json", lambda messages, schema: payload)


def _fields() -> list:
    return [
        {
            "unique_key": "invoiceNumber",
            "display_name": "Invoice Number",
            "extract_hint": "Top right",
            "transform_rule": "trim",
            "invoice_field": "Invoice No",
            "document_field": "Invoice No",
            "data_type": "string",
            "category": "header",
        },
        {
            "unique_key": "total",
            "display_name": "Total",
            "data_type": "number",
            "category": "header",
        },
        {
            "unique_key": "description",
            "display_name": "Description",
            "data_type": "string",
            "category": "line_item",
        },
        {
            "unique_key": "quantity",
            "display_name": "Qty",
            "data_type": "number",
            "category": "line_item",
        },
    ]


def _post(client: TestClient, **overrides):
    data = {
        "unique_ref_no": "1042",
        "fields": json.dumps(_fields()),
        "match": "false",
    }
    data.update(overrides)
    return client.post(
        "/extract/invoice",
        files={"file": ("inv.png", b"fakepng", "image/png")},
        data=data,
    )


def test_sync_endpoint_happy_path_match_false(monkeypatch):
    _stub_ocr(monkeypatch)
    _stub_llm(
        monkeypatch,
        {
            "headers": {"invoiceNumber": " INV-2026-0091 ", "total": "$4,777.50"},
            "line_items": [{"description": "Steel Pipe", "quantity": "100"}],
        },
    )
    with TestClient(app) as client:
        r = _post(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["unique_ref_no"] == "1042"
    assert body["status"] == "success"
    # MD §3 shape: header pairs keyed by unique_key, values stringified.
    pairs = {p["key"]: p for p in body["key_value_pairs"]}
    assert pairs["invoiceNumber"]["value"] == "INV-2026-0091"
    assert pairs["total"]["value"] == "4777.5"
    assert pairs["total"]["confidence"] is None
    # Tables: rows of [{key, value}], values are strings.
    assert body["tables"][0] == [
        {"key": "description", "value": "Steel Pipe"},
        {"key": "quantity", "value": "100.0"},
    ]
    assert body["match_results"] == []


def test_sync_endpoint_alias_v1(monkeypatch):
    _stub_ocr(monkeypatch)
    _stub_llm(monkeypatch, {"headers": {"invoiceNumber": "X", "total": None}, "line_items": []})
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/extract/invoice",
            files={"file": ("inv.png", b"fakepng", "image/png")},
            data={
                "unique_ref_no": "7",
                "fields": json.dumps(_fields()),
                "match": "false",
            },
        )
    assert r.status_code == 200, r.text
    assert r.json()["unique_ref_no"] == "7"


def test_sync_endpoint_match_true_exact(monkeypatch):
    _stub_ocr(monkeypatch)
    _stub_llm(
        monkeypatch,
        {
            "headers": {"invoiceNumber": "INV-1", "total": 100},
            "line_items": [{"description": "Steel Pipe 2in x 6m", "quantity": 100}],
        },
    )
    po_items = [
        {"material_number": "MAT-1001", "material_description": "Steel Pipe 2in x 6m"}
    ]
    conditions = [
        {
            "priority": 2,
            "po_field": "material_description",
            "invoice_field_key": "description",
            "match_type": "fuzzy",
            "description": "Fallback description similarity",
        }
    ]
    with TestClient(app) as client:
        r = _post(
            client,
            match="true",
            po_items=json.dumps(po_items),
            matching_conditions=json.dumps(conditions),
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["match_results"]) == 1
    m = body["match_results"][0]
    assert m["invoice_field"] == "description"
    assert m["po_field"] == "material_description"
    assert m["score"] == 5.0
    assert m["best_match"] == "Steel Pipe 2in x 6m"


def test_sync_endpoint_validation(monkeypatch):
    _stub_ocr(monkeypatch)
    _stub_llm(monkeypatch, {"headers": {}, "line_items": []})
    with TestClient(app) as client:
        # missing unique_ref_no
        r = client.post(
            "/extract/invoice",
            files={"file": ("inv.png", b"fakepng", "image/png")},
            data={"unique_ref_no": "  ", "fields": json.dumps(_fields()), "match": "false"},
        )
        assert r.status_code == 400

        # bad fields JSON
        r = _post(client, fields="not-json{")
        assert r.status_code == 400

        # duplicate unique_key
        dup = _fields() + [_fields()[0]]
        r = _post(client, fields=json.dumps(dup))
        assert r.status_code == 400

        # match=true without po_items
        r = _post(client, match="true")
        assert r.status_code == 400

        # unknown invoice_field_key
        r = _post(
            client,
            match="true",
            po_items=json.dumps([{"a": 1}]),
            matching_conditions=json.dumps(
                [
                    {
                        "priority": 1,
                        "po_field": "a",
                        "invoice_field_key": "nope",
                        "match_type": "exact",
                    }
                ]
            ),
        )
        assert r.status_code == 400

        # invalid match flag
        r = _post(client, match="maybe")
        assert r.status_code == 400


def test_sync_endpoint_llm_errors_mapped(monkeypatch):
    _stub_ocr(monkeypatch)

    def _raise(messages, schema):
        raise llm_client.LLMNotConfigured("no key")

    monkeypatch.setattr(llm_client, "complete_json", _raise)
    with TestClient(app) as client:
        assert _post(client).status_code == 503


def test_field_adapter_groups_and_ignores_transform_rule():
    fields = [FieldMapping.model_validate(f) for f in _fields()]
    spec = field_adapter.adapt_fields(fields)
    assert [f.key for f in spec.headers] == ["invoiceNumber", "total"]
    assert [f.key for f in spec.line_items] == ["description", "quantity"]
    # transform_rule is accepted-but-ignored: values pass through untouched.
    assert field_adapter.apply_transform_rule(" abc ", "trim") == " abc "
    assert field_adapter.apply_transform_rule("abc", "upper") == "abc"


def test_cross_category_key_reuse_allowed():
    """GST-style payloads reuse tax keys as header totals and row values."""
    from app.api.v1.invoice import _validated_fields

    raw = [
        {"unique_key": "cgstAmount", "data_type": "number", "category": "header"},
        {"unique_key": "cgstAmount", "data_type": "number", "category": "line_item"},
        {"unique_key": "sgstAmount", "data_type": "number", "category": "header"},
        {"unique_key": "sgstAmount", "data_type": "number", "category": "line_item"},
    ]
    fields = _validated_fields(raw)
    assert len(fields) == 4
    spec = field_adapter.adapt_fields(fields)
    assert [f.key for f in spec.headers] == ["cgstAmount", "sgstAmount"]
    assert [f.key for f in spec.line_items] == ["cgstAmount", "sgstAmount"]


def test_same_category_duplicate_still_rejected():
    from fastapi import HTTPException

    from app.api.v1.invoice import _validated_fields

    with_test = [
        {"unique_key": "a", "data_type": "string", "category": "header"},
        {"unique_key": "a", "data_type": "string", "category": "header"},
    ]
    try:
        _validated_fields(with_test)
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("expected 400 for same-category duplicate")


def test_matcher_priority_wins():
    rows = [{"description": "Pipe Clamp 2in", "itemCode": "MAT-2002"}]
    fields = [
        FieldMapping(unique_key="description", document_field="Description"),
        FieldMapping(unique_key="itemCode", document_field="Item Code"),
    ]
    po = [{"material_number": "MAT-2002", "material_description": "Something else"}]
    from app.dtos.invoice import MatchingCondition

    conditions = [
        MatchingCondition(
            priority=1,
            po_field="material_number",
            invoice_field_key="itemCode",
            match_type="exact",
        ),
        MatchingCondition(
            priority=2,
            po_field="material_description",
            invoice_field_key="description",
            match_type="fuzzy",
        ),
    ]
    out = match_invoice_rows(rows, po, conditions, fields)
    assert len(out) == 1
    assert out[0].priority == 1
    assert out[0].invoice_document_field == "Item Code"


def test_pipeline_injects_generic_hint(monkeypatch):
    captured: dict = {}

    def _fake(messages, schema):
        captured["user"] = messages[1]["content"]
        return {"headers": {"invoiceNumber": "X", "total": None}, "line_items": []}

    monkeypatch.setattr(llm_client, "complete_json", _fake)
    fields = [FieldMapping.model_validate(f) for f in _fields()]
    pipeline.extract_from_ocr_text(
        "INV",
        fields,
        document_type="invoice",
        generic_hint="Amounts are in SAR.",
    )
    assert "Amounts are in SAR" in captured["user"]
    assert "DOCUMENT TYPE: invoice" in captured["user"]
