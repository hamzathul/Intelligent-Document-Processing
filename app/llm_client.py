"""Thin OpenAI-compatible chat client. Only job: JSON-mode chat calls.

Point it anywhere speaking the OpenAI protocol via env (.env is loaded by
:mod:`app`, so a local `.env` file works):
  LLM_BASE_URL   (default https://api.openai.com/v1, e.g. Ollama http://localhost:11434/v1)
  LLM_API_KEY    (required — never logged)
  LLM_MODEL      (default gpt-4o-mini)
  LLM_TIMEOUT_S  (default 60)
  LLM_TEMPERATURE (default 0 — extraction must be deterministic)
"""

from __future__ import annotations

import json
import logging
import os
import time

from openai import BadRequestError, OpenAI, UnprocessableEntityError

log = logging.getLogger("uvicorn.error")

# Providers that don't understand strict json_schema answer 400/422 —
# only then do we retry in plain json_object mode.
_NEEDS_PLAIN_JSON = (BadRequestError, UnprocessableEntityError)


class LLMNotConfigured(Exception):
    """Raised when no LLM_API_KEY is set. Route maps this to HTTP 503."""


class LLMError(Exception):
    """Raised when the LLM call fails or returns unusable output.

    ``raw`` carries the model output when the failure is about the content
    (bad JSON / non-object) so debug traces can persist it.
    """

    def __init__(self, message: str = "", *, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


def model_name() -> str:
    return os.getenv("LLM_MODEL", "gpt-4o-mini")


def _temperature() -> float:
    try:
        return float(os.getenv("LLM_TEMPERATURE", "0"))
    except ValueError:
        return 0.0


def _client() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        raise LLMNotConfigured(
            "LLM_API_KEY is not set. Copy .env.example to .env and set "
            "LLM_BASE_URL/LLM_API_KEY/LLM_MODEL to enable guided extraction."
        )
    return OpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=api_key,
        timeout=float(os.getenv("LLM_TIMEOUT_S", "60")),
        max_retries=1,
    )


class _EmptyResponse(Exception):
    """Internal: provider answered 200 but with no content.

    Common with free-tier routers (e.g. ``openrouter/free``): the randomly
    selected model may ignore ``response_format`` or emit nothing. Retriable
    once in plain ``json_object`` mode before surfacing an error.
    """

    def __init__(self, finish_reason: str | None = None) -> None:
        super().__init__(f"finish_reason={finish_reason}")
        self.finish_reason = finish_reason


def _chat(client: OpenAI, messages: list[dict], response_format: dict) -> tuple[str, str | None]:
    """One chat call. Returns (content, finish_reason); raises _EmptyResponse on blank."""
    resp = client.chat.completions.create(
        model=model_name(),
        messages=messages,  # type: ignore[arg-type]
        temperature=_temperature(),
        response_format=response_format,  # type: ignore[arg-type]
    )
    choice = resp.choices[0] if resp.choices else None
    content = (choice.message.content if choice and choice.message else None) or ""
    finish = getattr(choice, "finish_reason", None)
    if not content.strip():
        raise _EmptyResponse(finish)
    return content, finish


def _chat_plain(client: OpenAI, messages: list[dict]) -> tuple[str, str | None]:
    """Plain ``json_object`` attempt. Maps every failure to LLMError."""
    try:
        return _chat(client, messages, {"type": "json_object"})
    except _EmptyResponse as exc:
        raise LLMError(
            f"LLM returned an empty response (model={model_name()}, "
            f"finish_reason={exc.finish_reason}). The model ignored the request — "
            "retry, or switch LLM_MODEL to one with structured-output support."
        ) from exc
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"LLM request failed: {exc}") from exc


def complete_json(messages: list[dict], schema: dict) -> dict:
    """Chat and return the parsed JSON object. Raises LLMNotConfigured/LLMError."""
    client = _client()  # raises LLMNotConfigured before any network use
    log.info(
        "[llm] request starting model=%s base=%s...",
        model_name(),
        os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    t0 = time.perf_counter()
    strict_format = {
        "type": "json_schema",
        "json_schema": {"name": "extraction", "strict": True, "schema": schema},
    }
    try:
        content, finish = _chat(client, messages, strict_format)
        mode = "strict"
    except _NEEDS_PLAIN_JSON as exc:
        log.info("[llm] strict json_schema unsupported (%s), retrying json_object mode", exc)
        content, finish = _chat_plain(client, messages)
        mode = "plain"
    except _EmptyResponse as exc:
        log.info("[llm] strict mode returned empty (%s), retrying json_object mode", exc)
        content, finish = _chat_plain(client, messages)
        mode = "plain"
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 — auth, timeout, network: fail, don't retry
        raise LLMError(f"LLM request failed: {exc}") from exc
    log.info(
        "[llm] response received in %.1fs (mode=%s finish=%s chars=%d)",
        time.perf_counter() - t0,
        mode,
        finish,
        len(content),
    )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM did not return valid JSON: {exc}", raw=content) from exc
    if not isinstance(parsed, dict):
        raise LLMError("LLM response was not a JSON object", raw=content)
    return parsed
