"""Tests for per-request debug traces (offline, no OCR model / LLM key)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app import llm_client, ocr_service
from app.core.config import get_settings
from app.core.debug_trace import NULL_TRACE, DebugTrace, new_trace
from app.dtos.invoice import FieldMapping
from app.main import app
from app.services import pipeline


def test_trace_disabled_writes_nothing(tmp_path):
    trace = new_trace("1042", base_dir=str(tmp_path), enabled=False)
    assert trace.dir is None
    trace.save_json("01_request.json", {"a": 1})  # must not raise
    assert list(tmp_path.iterdir()) == []
    NULL_TRACE.save_text("x.txt", "hi")  # must not raise


def test_trace_writes_and_prunes(tmp_path):
    base = tmp_path / "debug"
    t1 = new_trace("A", base_dir=str(base), enabled=True, keep=2)
    t1.save_json("01_request.json", {"ref": "A"})
    t2 = new_trace("B", base_dir=str(base), enabled=True, keep=2)
    t2.save_text("note.txt", "hello")
    t3 = new_trace("C", base_dir=str(base), enabled=True, keep=2)
    t3.save_json("01_request.json", {"ref": "C"})
    remaining = sorted(d.name for d in base.iterdir() if d.is_dir())
    assert len(remaining) == 2
    assert any(n.startswith("C_") for n in remaining)
    assert (t3.dir / "01_request.json").exists()


def test_pipeline_trace_captures_llm_exchange(monkeypatch, tmp_path):
    monkeypatch.setattr(
        llm_client, "complete_json", lambda messages, schema: {
            "headers": {"a": "hi"}, "line_items": []
        }
    )
    trace = DebugTrace(enabled=True, dir=tmp_path / "t")
    trace.dir.mkdir()
    fields = [FieldMapping(unique_key="a", document_field="A")]
    pipeline.extract_from_ocr_text("some text", fields, trace=trace)
    request = json.loads((trace.dir / "03_llm_request.json").read_text(encoding="utf-8"))
    assert set(request) == {"model", "messages", "json_schema"}
    assert "some text" in request["messages"][1]["content"]
    assert request["json_schema"]["properties"]["headers"]["properties"].keys() == {"a"}
    response = json.loads((trace.dir / "04_llm_response.json").read_text(encoding="utf-8"))
    assert response["headers"] == {"a": "hi"}
    extracted = json.loads((trace.dir / "05_extracted.json").read_text(encoding="utf-8"))
    assert extracted["headers"] == {"a": "hi"}


def test_pipeline_trace_records_llm_error(monkeypatch, tmp_path):
    def _raise(messages, schema):
        raise llm_client.LLMError("LLM did not return valid JSON: bad", raw="not json{")

    monkeypatch.setattr(llm_client, "complete_json", _raise)
    trace = DebugTrace(enabled=True, dir=tmp_path / "t")
    trace.dir.mkdir()
    fields = [FieldMapping(unique_key="a", document_field="A")]
    try:
        pipeline.extract_from_ocr_text("text", fields, trace=trace)
    except llm_client.LLMError:
        pass
    else:
        raise AssertionError("expected LLMError")
    error = json.loads((trace.dir / "04_llm_error.json").read_text(encoding="utf-8"))
    assert error["raw"] == "not json{"


def test_llm_error_carries_raw(monkeypatch):
    from types import SimpleNamespace

    calls: list[dict] = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not json{"))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    try:
        llm_client.complete_json([], {})
    except llm_client.LLMError as exc:
        assert exc.raw == "not json{"
    else:
        raise AssertionError("expected LLMError")


def test_route_writes_trace_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ocr_service, "predict_path",
        lambda *a, **k: [{"rec_texts": ["Hello"], "rec_scores": [1.0],
                          "rec_polys": [], "rec_boxes": []}],
    )
    monkeypatch.setattr(
        llm_client, "complete_json",
        lambda messages, schema: {"headers": {"a": "hello"}, "line_items": []},
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
        debug_root = tmp_path / "debug"
        trace_dirs = [d for d in debug_root.iterdir() if d.is_dir()]
        assert len(trace_dirs) == 1
        names = {p.name for p in trace_dirs[0].iterdir()}
        for expected in ("01_request.json", "02_ocr.json", "02_ocr.txt",
                         "03_llm_request.json", "04_llm_response.json",
                         "05_extracted.json", "06_match.json",
                         "07_response.json", "08_meta.json"):
            assert expected in names, f"missing {expected}"
        request = json.loads((trace_dirs[0] / "01_request.json").read_text(encoding="utf-8"))
        assert request["unique_ref_no"] == "1042"
    finally:
        get_settings.cache_clear()
