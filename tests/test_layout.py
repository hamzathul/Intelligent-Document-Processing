"""Tests for position+confidence annotated prompts (offline)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app import llm_client, ocr_service
from app.core.config import get_settings
from app.dtos.invoice import FieldMapping
from app.fields import ExtractSpec
from app.main import app
from app.services import pipeline
from app import prompt


def _spec() -> ExtractSpec:
    return ExtractSpec(headers=[{"key": "a", "field": "A"}])


def _pages():
    return [
        {
            "page_index": 0,
            "text": "Tax Invoice No\nINV-1",
            "lines": [
                {
                    "text": "Tax Invoice No",
                    "score": 0.99,
                    "box": [[1452, 118], [1764, 118], [1764, 162], [1452, 162]],
                },
                {"text": "INV-1", "score": 0.62, "box": [[1452, 170], [1700, 170], [1700, 210], [1452, 210]]},
                {"text": "   ", "score": 0.5, "box": None},
            ],
        }
    ]


def test_render_layout_positions_and_confidence():
    text = prompt.render_layout_text(_pages())
    assert "[page 0]" in text
    assert "L001 x=1452 y=118 w=312 h=44 conf=0.99 | Tax Invoice No" in text
    # Low-confidence reading is flagged.
    assert "conf=0.62 [!] | INV-1" in text
    # Blank OCR lines are dropped.
    assert "L003" not in text


def test_render_layout_missing_geometry():
    text = prompt.render_layout_text(
        [{"page_index": 0, "text": "", "lines": [{"text": "Hi", "score": 0.9, "box": None}]}]
    )
    assert "L001 conf=0.90 | Hi" in text
    assert "x=" not in text
    text = prompt.render_layout_text(
        [{"page_index": 0, "text": "", "lines": [{"text": "Hi"}]}]
    )
    assert "L001 | Hi" in text


def test_render_layout_truncates_whole_lines():
    pages = [
        {
            "page_index": 0,
            "text": "",
            "lines": [{"text": f"row-{i:04d}-" + "x" * 200, "score": 1.0,
                       "box": [[0, i * 10], [100, i * 10], [100, i * 10 + 8], [0, i * 10 + 8]]}
                      for i in range(500)],
        }
    ]
    text = prompt.render_layout_text(pages)
    assert len(text) <= prompt.MAX_OCR_CHARS + 100
    assert "truncated" in text
    # No cut mid-line: every content line still parses as L### ... | text.
    for line in text.splitlines():
        if line.startswith("["):
            continue
        assert " | " in line


def test_build_messages_layout_switches_prompt():
    spec = _spec()
    plain = prompt.build_messages(spec, "hello")
    assert "DOCUMENT TEXT (OCR, top-to-bottom):" in plain[1]["content"]
    assert "conf" not in plain[0]["content"]

    layout = prompt.render_layout_text(_pages())
    with_layout = prompt.build_messages(spec, "hello", layout=layout)
    assert "positions + confidence" in with_layout[1]["content"]
    assert "conf" in with_layout[0]["content"]
    assert "x=1452" in with_layout[1]["content"]


def test_pipeline_sends_layout_when_lines_present(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        llm_client, "complete_json",
        lambda messages, schema: (captured.update(user=messages[1]["content"]),
                                  {"headers": {"a": "x"}, "line_items": []})[1],
    )
    fields = [FieldMapping(unique_key="a", document_field="A")]
    pipeline.extract_from_pages(
        _pages(), fields, "1042", document_type="invoice",
    )
    assert "x=1452 y=118" in captured["user"]
    assert "conf=0.99" in captured["user"]
    assert "[!]" in captured["user"]


def test_pipeline_falls_back_to_plain_text(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        llm_client, "complete_json",
        lambda messages, schema: (captured.update(
            system=messages[0]["content"], user=messages[1]["content"]),
            {"headers": {"a": "x"}, "line_items": []})[1],
    )
    fields = [FieldMapping(unique_key="a", document_field="A")]
    # Text-only pages still get line refs, but no geometry/confidence legend.
    pipeline.extract_from_pages(
        [{"page_index": 0, "text": "hello", "lines": []}], fields, "1042",
    )
    assert "L001 | hello" in captured["user"]
    assert "x=" not in captured["user"]
    assert "conf=" not in captured["user"]
    # Fully empty pages fall back to the plain-text prompt.
    pipeline.extract_from_pages(
        [{"page_index": 0, "text": "", "lines": []}], fields, "1042",
    )
    assert "DOCUMENT TEXT (OCR, top-to-bottom):" in captured["user"]


def test_route_trace_contains_positions_and_confidence(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ocr_service, "predict_path",
        lambda *a, **k: [{
            "rec_texts": ["Tax Invoice No", "INV-1"],
            "rec_scores": [0.99, 0.62],
            "rec_polys": [
                [[1452, 118], [1764, 118], [1764, 162], [1452, 162]],
                [[1452, 170], [1700, 170], [1700, 210], [1452, 210]],
            ],
            "rec_boxes": [],
        }],
    )
    monkeypatch.setattr(
        llm_client, "complete_json",
        lambda messages, schema: {"headers": {"a": "x"}, "line_items": []},
    )
    monkeypatch.setenv("IDP_DEBUG_DIR", str(tmp_path / "debug"))
    monkeypatch.setenv("IDP_DEBUG_TRACE", "1")
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/extract/invoice",
                files={"file": ("inv.png", b"fakepng", "image/png")},
                data={
                    "unique_ref_no": "1042",
                    "fields": json.dumps([{"unique_key": "a", "document_field": "A"}]),
                    "match": "false",
                },
            )
        assert r.status_code == 200, r.text
        trace_dir = next((tmp_path / "debug").iterdir())
        llm_request = json.loads((trace_dir / "03_llm_request.json").read_text(encoding="utf-8"))
        assert "x=1452 y=118" in llm_request["messages"][1]["content"]
        assert "conf=0.99" in llm_request["messages"][1]["content"]
        assert "[!]" in llm_request["messages"][1]["content"]
        ocr = json.loads((trace_dir / "02_ocr.json").read_text(encoding="utf-8"))
        assert ocr["pages"][0]["lines"][0]["box"] is not None
    finally:
        get_settings.cache_clear()
