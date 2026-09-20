"""Offline tests for guided extraction. No LLM key, no OCR model needed.

LLM and OCR boundaries are stubbed with monkeypatch; pure modules
(fields/coerce/prompt) are tested directly.
"""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import coerce, extract_service, llm_client, prompt
from app.fields import ExtractSpec, FieldSpec


def _spec() -> ExtractSpec:
    return ExtractSpec(
        headers=[
            {"key": "inv_no", "field": "Invoice Number", "type": "string"},
            {"key": "total", "field": "Total", "type": "number"},
            {
                "key": "due_date",
                "field": "Due Date",
                "type": "date",
                "hint": "Payment Terms 30 days means Invoice Date + 30 days",
            },
        ],
        line_items=[
            {"key": "desc", "field": "Description", "type": "string"},
            {"key": "qty", "field": "Quantity", "type": "number"},
        ],
    )


# --- spec validation --------------------------------------------------------


def test_spec_rejects_duplicate_keys():
    with pytest.raises(ValidationError):
        ExtractSpec(headers=[
            {"key": "a", "field": "A"},
            {"key": "a", "field": "A again"},
        ])


def test_spec_rejects_bad_key_and_type():
    with pytest.raises(ValidationError):
        FieldSpec(key="1 bad", field="X")
    with pytest.raises(ValidationError):
        ExtractSpec(headers=[{"key": "ok", "field": "X", "type": "money"}])


def test_spec_rejects_empty_and_too_many():
    with pytest.raises(ValidationError):
        ExtractSpec()
    with pytest.raises(ValidationError):
        ExtractSpec(headers=[{"key": f"k{i}", "field": "F"} for i in range(51)])


# --- coercion ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$1,299.50", 1299.5),
        ("(7.25)", -7.25),
        ("18%", 18.0),
        ("22,440.00 SAR", 22440.0),
        ("3,366.00 sar", 3366.0),
        ("42", 42.0),
        (12, 12.0),
        ("N/A", None),
        ("-", None),
        (True, None),
        ("abc", None),
    ],
)
def test_to_number(raw, expected):
    assert coerce.to_number(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12 Aug 2026", "2026-08-12"),
        ("2026-09-11", "2026-09-11"),
        ("2026-Jun-28", "2026-06-28"),
        ("28-Jun-2026", "2026-06-28"),
        ("05/06/2026", "2026-06-05"),  # day-first, documented
        ("August 12, 2026", "2026-08-12"),
        ("not a date", None),
        ("", None),
    ],
)
def test_to_date(raw, expected):
    assert coerce.to_date(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("yes", True), ("NO", False), ("Paid", True), ("1", True), ("maybe", None), (0, False)],
)
def test_to_boolean(raw, expected):
    assert coerce.to_boolean(raw) == expected


# --- prompt -----------------------------------------------------------------


def test_prompt_contains_keys_hints_and_json_shape():
    spec = _spec()
    messages = prompt.build_messages(spec, "Invoice Number INV-1\nTotal $10")
    assert messages[0]["role"] == "system"
    full_text = messages[0]["content"] + "\n" + messages[1]["content"]
    for token in ('"inv_no"', '"due_date"', '"qty"', "Payment Terms 30 days",
                  '{"headers": {...}, "line_items": [...]}', "JSON ONLY"):
        assert token in full_text
    schema = prompt.build_json_schema(spec)
    assert set(schema["properties"]) == {"headers", "line_items"}
    assert set(schema["properties"]["headers"]["properties"]) == {"inv_no", "total", "due_date"}


def test_prompt_truncates_huge_documents():
    spec = ExtractSpec(headers=[{"key": "a", "field": "A"}])
    user = prompt.build_user_message(spec, "x" * (prompt.MAX_OCR_CHARS + 100))
    assert "truncated" in user


def test_prompt_drops_blank_ocr_lines():
    spec = ExtractSpec(headers=[{"key": "a", "field": "A"}])
    user = prompt.build_user_message(spec, "Invoice Number\n\n\nJPP26-01533\n\n")
    doc = user.split("DOCUMENT TEXT (OCR, top-to-bottom):")[1]
    assert doc.strip().splitlines() == ["Invoice Number", "JPP26-01533"]


# --- service with stubbed LLM -----------------------------------------------


def _stub_llm(monkeypatch, payload: dict):
    monkeypatch.setattr(
        llm_client, "complete_json", lambda messages, schema: payload
    )


def test_service_coerces_and_nulls_missing(monkeypatch):
    _stub_llm(monkeypatch, {
        "headers": {"inv_no": " INV-2041 ", "total": "$1,299.50"},
        "line_items": [{"desc": "Widget", "qty": "2"}, {"desc": None, "qty": "x"}],
        "extra_key": "ignored",
    })
    out = extract_service.extract_from_pages(
        [{"page_index": 0, "text": "Invoice Number INV-2041", "lines": []}], _spec()
    )
    assert out["headers"] == {"inv_no": "INV-2041", "total": 1299.5, "due_date": None}
    assert out["line_items"] == [
        {"desc": "Widget", "qty": 2.0},
        {"desc": None, "qty": None},
    ]
    assert "INV-2041" in out["ocr_text"]


def test_llm_not_configured_without_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(llm_client.LLMNotConfigured):
        llm_client.complete_json([], {})


# --- routes (OCR + LLM stubbed) ---------------------------------------------


def _client_with_stubs(monkeypatch, llm_payload: dict | Exception):
    from app import ocr_service
    from app.main import app

    monkeypatch.setattr(
        ocr_service, "predict_path",
        lambda *a, **k: [{"rec_texts": ["Hello"], "rec_scores": [1.0],
                          "rec_polys": [], "rec_boxes": []}],
    )
    if isinstance(llm_payload, Exception):
        def _raise(messages, schema):
            raise llm_payload
        monkeypatch.setattr(llm_client, "complete_json", _raise)
    else:
        _stub_llm(monkeypatch, llm_payload)
    return TestClient(app)


def test_extract_route_503_without_llm_key(monkeypatch):
    client = _client_with_stubs(monkeypatch, llm_client.LLMNotConfigured("no key"))
    r = client.post(
        "/api/extract",
        files={"file": ("inv.png", b"fakepng", "image/png")},
        data={"spec": json.dumps({"headers": [{"key": "a", "field": "A"}]})},
    )
    assert r.status_code == 503


def test_extract_route_happy_path(monkeypatch):
    client = _client_with_stubs(monkeypatch, {
        "headers": {"a": "hello"}, "line_items": [],
    })
    r = client.post(
        "/api/extract",
        files={"file": ("inv.png", b"fakepng", "image/png")},
        data={"spec": json.dumps({"headers": [{"key": "a", "field": "A"}]})},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["headers"] == {"a": "hello"}
    assert body["line_items"] == []


def test_extract_route_rejects_bad_spec(monkeypatch):
    client = _client_with_stubs(monkeypatch, {"headers": {}, "line_items": []})
    r = client.post(
        "/api/extract",
        files={"file": ("inv.png", b"fakepng", "image/png")},
        data={"spec": "not-json{"},
    )
    assert r.status_code == 400
    r = client.post(
        "/api/extract",
        files={"file": ("inv.png", b"fakepng", "image/png")},
        data={"spec": json.dumps({"headers": []})},
    )
    assert r.status_code == 400


# --- llm_client wire behavior (SDK stubbed, no network) ----------------------


def _fake_openai(monkeypatch, behaviors):
    """Replace llm_client.OpenAI. behaviors: list of ("ok", raw_content) or ("raise", exc)."""
    from types import SimpleNamespace

    calls: list[dict] = []

    def _resp(content):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            action, value = behaviors.pop(0)
            if action == "raise":
                raise value
            return _resp(value)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    return calls


def _bad_request_error():
    import httpx
    from openai import BadRequestError

    req = httpx.Request("POST", "http://test/v1/chat/completions")
    resp = httpx.Response(400, request=req, json={"error": {"message": "no strict"}})
    return BadRequestError("unsupported", response=resp, body=None)


def test_llm_sends_strict_schema_first(monkeypatch):
    calls = _fake_openai(monkeypatch, [("ok", '{"headers": {}, "line_items": []}')])
    schema = {"type": "object"}
    out = llm_client.complete_json([{"role": "user", "content": "hi"}], schema)
    assert out == {"headers": {}, "line_items": []}
    assert len(calls) == 1
    assert calls[0]["model"] == "test-model"
    assert calls[0]["temperature"] == 0
    fmt = calls[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == schema


def test_llm_falls_back_to_json_object_on_400(monkeypatch):
    calls = _fake_openai(monkeypatch, [
        ("raise", _bad_request_error()),
        ("ok", '{"headers": {"a": "x"}, "line_items": []}'),
    ])
    out = llm_client.complete_json([], {})
    assert out["headers"] == {"a": "x"}
    assert len(calls) == 2
    assert calls[1]["response_format"] == {"type": "json_object"}


def test_llm_does_not_retry_other_errors(monkeypatch):
    calls = _fake_openai(monkeypatch, [("raise", RuntimeError("auth exploded"))])
    with pytest.raises(llm_client.LLMError, match="auth exploded"):
        llm_client.complete_json([], {})
    assert len(calls) == 1


def test_llm_rejects_empty_and_non_json(monkeypatch):
    # Persistently empty (strict + plain fallback) -> helpful error naming the cause.
    _fake_openai(monkeypatch, [("ok", None), ("ok", "  ")])
    with pytest.raises(llm_client.LLMError, match="empty response"):
        llm_client.complete_json([], {})
    _fake_openai(monkeypatch, [("ok", "not json{")])
    with pytest.raises(llm_client.LLMError, match="valid JSON"):
        llm_client.complete_json([], {})
    _fake_openai(monkeypatch, [("ok", "[1, 2]")])
    with pytest.raises(llm_client.LLMError, match="not a JSON object"):
        llm_client.complete_json([], {})


def test_llm_falls_back_to_json_object_on_empty_strict(monkeypatch):
    calls = _fake_openai(monkeypatch, [
        ("ok", ""),
        ("ok", '{"headers": {"a": "x"}, "line_items": []}'),
    ])
    out = llm_client.complete_json([], {})
    assert out["headers"] == {"a": "x"}
    assert len(calls) == 2
    assert calls[1]["response_format"] == {"type": "json_object"}


def test_dotenv_key_reaches_client(monkeypatch, tmp_path):
    import os

    from dotenv import load_dotenv

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    old = os.environ.get("LLM_API_KEY")
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API_KEY=file-key-123\n", encoding="utf-8")
    try:
        load_dotenv(dotenv_path=env_file)
        assert os.environ.get("LLM_API_KEY") == "file-key-123"
        llm_client._client()  # must not raise LLMNotConfigured
    finally:
        if old is None:
            os.environ.pop("LLM_API_KEY", None)
        else:
            os.environ["LLM_API_KEY"] = old


# --- logging ----------------------------------------------------------------


def test_extract_logs_each_stage(monkeypatch, caplog):
    import logging

    client = _client_with_stubs(monkeypatch, {
        "headers": {"a": "hello"}, "line_items": [],
    })
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        r = client.post(
            "/api/extract",
            files={"file": ("inv.png", b"fakepng", "image/png")},
            data={"spec": json.dumps({"headers": [{"key": "a", "field": "A"}]})},
        )
    assert r.status_code == 200
    messages = [rec.getMessage() for rec in caplog.records]
    for stage in ("[extract] request", "[extract] ocr starting", "[extract] ocr done",
                  "[extract] llm starting", "[extract] llm done", "[extract] request done"):
        assert any(stage in m for m in messages), f"missing log: {stage}"
