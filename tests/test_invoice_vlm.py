"""Offline tests for `POST /extract/invoice/vlm` (no OCR model / VLM key needed)."""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from app import ocr_service
from app.api.v1 import invoice_vlm
from app.infra import vlm_client
from app.main import app
from app.services.images import PageImage, TooManyPagesError, render_pages
from app.services import vlm_prompt


def _fake_page() -> PageImage:
    return PageImage(
        page_index=0,
        mime="image/jpeg",
        data_url="data:image/jpeg;base64,/9j/fake",
        width=800,
        height=600,
        source_bytes=1234,
    )


def _stub_render(monkeypatch, pages=None):
    pages = pages if pages is not None else [_fake_page()]
    monkeypatch.setattr(invoice_vlm, "render_pages", lambda *a, **k: pages)


def _stub_vlm(monkeypatch, payload: dict):
    monkeypatch.setattr(vlm_client, "complete_vision", lambda messages, schema: payload)


def _assert_no_ocr(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("OCR must not run on the VLM route")

    monkeypatch.setattr(ocr_service, "predict_path", _boom)
    monkeypatch.setattr(ocr_service, "get_ocr", _boom)


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


def _post(client: TestClient, path="/extract/invoice/vlm", **overrides):
    data = {
        "unique_ref_no": "1042",
        "fields": json.dumps(_fields()),
        "match": "false",
    }
    data.update(overrides)
    return client.post(
        path,
        files={"file": ("inv.png", b"fakepng", "image/png")},
        data=data,
    )


def test_vlm_endpoint_happy_path_match_false(monkeypatch):
    _assert_no_ocr(monkeypatch)
    _stub_render(monkeypatch)
    _stub_vlm(
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
    pairs = {p["key"]: p for p in body["key_value_pairs"]}
    assert pairs["invoiceNumber"]["value"] == "INV-2026-0091"
    assert pairs["total"]["value"] == "4777.5"
    assert pairs["total"]["confidence"] is None
    assert body["tables"][0] == [
        {"key": "description", "value": "Steel Pipe"},
        {"key": "quantity", "value": "100.0"},
    ]
    assert body["match_results"] == []
    assert body["intermediate_data"]["mode"] == "vlm-direct"
    assert body["intermediate_data"]["pages"] == 1


def test_vlm_endpoint_alias_v1(monkeypatch):
    _assert_no_ocr(monkeypatch)
    _stub_render(monkeypatch)
    _stub_vlm(monkeypatch, {"headers": {"invoiceNumber": "X", "total": None}, "line_items": []})
    with TestClient(app) as client:
        r = _post(client, path="/api/v1/extract/invoice/vlm")
    assert r.status_code == 200, r.text
    assert r.json()["unique_ref_no"] == "1042"


def test_vlm_endpoint_match_true_exact(monkeypatch):
    _assert_no_ocr(monkeypatch)
    _stub_render(monkeypatch)
    _stub_vlm(
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


def test_vlm_endpoint_validation(monkeypatch):
    _assert_no_ocr(monkeypatch)
    _stub_render(monkeypatch)
    _stub_vlm(monkeypatch, {"headers": {}, "line_items": []})
    with TestClient(app) as client:
        # missing unique_ref_no
        r = client.post(
            "/extract/invoice/vlm",
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


def test_vlm_endpoint_vlm_errors_mapped(monkeypatch):
    _assert_no_ocr(monkeypatch)
    _stub_render(monkeypatch)

    def _raise_not_configured(messages, schema):
        raise vlm_client.LLMNotConfigured("no key")

    monkeypatch.setattr(vlm_client, "complete_vision", _raise_not_configured)
    with TestClient(app) as client:
        assert _post(client).status_code == 503

    def _raise_upstream(messages, schema):
        raise vlm_client.LLMError("boom")

    monkeypatch.setattr(vlm_client, "complete_vision", _raise_upstream)
    with TestClient(app) as client:
        assert _post(client).status_code == 502


def test_vlm_endpoint_too_many_pages_is_400(monkeypatch):
    _assert_no_ocr(monkeypatch)

    def _boom(*a, **k):
        raise TooManyPagesError("Document has 14 pages, max 8 for the VLM route")

    monkeypatch.setattr(invoice_vlm, "render_pages", _boom)
    _stub_vlm(monkeypatch, {"headers": {}, "line_items": []})
    with TestClient(app) as client:
        r = _post(client)
    assert r.status_code == 400
    assert "pages" in r.json()["detail"].lower()


def test_vlm_sends_page_images_not_ocr_text(monkeypatch):
    """The vision request must carry image_url parts (one per page)."""
    _assert_no_ocr(monkeypatch)
    _stub_render(monkeypatch, pages=[_fake_page(), _fake_page()])
    captured: dict = {}

    def _fake(messages, schema):
        captured["messages"] = messages
        return {"headers": {"invoiceNumber": "X", "total": None}, "line_items": []}

    monkeypatch.setattr(vlm_client, "complete_vision", _fake)
    with TestClient(app) as client:
        r = _post(client)
    assert r.status_code == 200, r.text
    user = captured["messages"][1]
    image_parts = [p for p in user["content"] if p.get("type") == "image_url"]
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"]["url"].startswith("data:image/")


def test_vlm_prompt_redacts_base64_for_trace():
    from app.fields import ExtractSpec, FieldSpec

    spec = ExtractSpec(headers=[FieldSpec(key="a", field="A")], line_items=[])
    big = PageImage(
        page_index=0,
        mime="image/jpeg",
        data_url="data:image/jpeg;base64," + "A" * 5000,
        width=10,
        height=10,
    )
    messages = vlm_prompt.build_vision_messages(spec, [big])
    redacted = vlm_prompt.redact_for_trace(messages)
    url = redacted[1]["content"][1]["image_url"]["url"]
    assert len(url) < 500
    assert redacted[1]["content"][1]["image_url"]["url_chars"] > 5000


def test_render_pages_real_png():
    """End-to-end through Pillow with a generated PNG (no network, no OCR)."""
    from PIL import Image

    img = Image.new("RGB", (64, 32), (10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    pages = render_pages(
        buf.getvalue(), ".png", max_pages=8, max_side_px=1568, jpeg_quality=85
    )
    assert len(pages) == 1
    assert pages[0].mime == "image/jpeg"
    assert pages[0].data_url.startswith("data:image/jpeg;base64,")
    assert pages[0].width == 64 and pages[0].height == 32


def test_render_pages_rejects_garbage():
    with pytest.raises(Exception):
        render_pages(b"not-an-image", ".png", max_pages=8, max_side_px=1568, jpeg_quality=85)


def test_vlm_settings_google_defaults_and_fallbacks(monkeypatch):
    """GOOGLE_API_KEY / GEMINI_MODEL drive the VLM route; VLM_* overrides win."""
    from app.core.config import Settings, get_settings

    get_settings.cache_clear()
    for var in (
        "VLM_BASE_URL",
        "VLM_API_KEY",
        "VLM_MODEL",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "LLM_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    s = Settings()
    assert s.vlm_base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert s.vlm_model == "gemini-2.5-flash"
    assert s.vlm_api_key == ""

    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("LLM_API_KEY", "legacy-key")
    s = Settings()
    assert s.vlm_api_key == "g-key"
    assert s.vlm_model == "gemini-2.5-flash"

    # Explicit VLM_* overrides win over GOOGLE_*/GEMINI_*.
    monkeypatch.setenv("VLM_API_KEY", "v-key")
    monkeypatch.setenv("VLM_MODEL", "other-model")
    s = Settings()
    assert s.vlm_api_key == "v-key"
    assert s.vlm_model == "other-model"

    get_settings.cache_clear()
    assert get_settings().vlm_base_url.startswith("https://generativelanguage.")
    get_settings.cache_clear()
