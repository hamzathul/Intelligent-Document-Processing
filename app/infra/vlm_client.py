"""Direct-vision chat client for the VLM invoice route.

Same resilience pattern as :mod:`app.infra.llm_client` (strict ``json_schema``
first, plain ``json_object`` fallback on 400/422 or empty content), but sends
OpenAI-style **vision messages** — text plus ``image_url`` data-URL parts —
to a vision-capable model. Defaults to Google AI Studio (Gemini Developer
API) via its OpenAI-compatible endpoint. No OCR involved.

Config (``app.core.config.Settings``)::

  VLM_BASE_URL  (default https://generativelanguage.googleapis.com/v1beta/openai/)
  GOOGLE_API_KEY (or VLM_API_KEY override; required — never logged)
  GEMINI_MODEL  (or VLM_MODEL override; default gemini-2.5-flash)
  VLM_TIMEOUT_S (default 120 — vision calls are slower than text-only)
"""

from __future__ import annotations

import json
import logging
import time

from openai import BadRequestError, OpenAI, UnprocessableEntityError

from app.core.config import get_settings
from app.infra.llm_client import LLMError, LLMNotConfigured, _EmptyResponse

log = logging.getLogger("uvicorn.error")

_NEEDS_PLAIN_JSON = (BadRequestError, UnprocessableEntityError)

__all__ = ["LLMError", "LLMNotConfigured", "model_name", "complete_vision"]


def model_name() -> str:
    return get_settings().vlm_model


def _client() -> OpenAI:
    settings = get_settings()
    if not settings.vlm_api_key:
        raise LLMNotConfigured(
            "No Gemini API key is set (VLM_API_KEY or GOOGLE_API_KEY). Copy "
            ".env.example to .env and set GOOGLE_API_KEY/GEMINI_MODEL to "
            "enable direct-VLM extraction."
        )
    return OpenAI(
        base_url=settings.vlm_base_url,
        api_key=settings.vlm_api_key,
        timeout=settings.vlm_timeout_s,
        max_retries=1,
    )


def _chat(client: OpenAI, messages: list[dict], response_format: dict) -> tuple[str, str | None]:
    """One vision chat call. Returns (content, finish_reason)."""
    resp = client.chat.completions.create(
        model=model_name(),
        messages=messages,  # type: ignore[arg-type]
        temperature=0.0,  # extraction must be deterministic
        response_format=response_format,  # type: ignore[arg-type]
    )
    choice = resp.choices[0] if resp.choices else None
    content = (choice.message.content if choice and choice.message else None) or ""
    finish = getattr(choice, "finish_reason", None)
    if not content.strip():
        raise _EmptyResponse(finish)
    return content, finish


def _chat_plain(client: OpenAI, messages: list[dict]) -> tuple[str, str | None]:
    try:
        return _chat(client, messages, {"type": "json_object"})
    except _EmptyResponse as exc:
        raise LLMError(
            f"VLM returned an empty response (model={model_name()}, "
            f"finish_reason={exc.finish_reason}). The model ignored the request — "
            "retry, or switch VLM_MODEL to one with structured-output support."
        ) from exc
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"VLM request failed: {exc}") from exc


def complete_vision(messages: list[dict], schema: dict) -> dict:
    """Vision chat, returning the parsed JSON object.

    Raises LLMNotConfigured / LLMError (same mapping as the text client:
    503 for missing key, 502 for upstream/bad-output failures).
    """
    client = _client()  # raises LLMNotConfigured before any network use
    settings = get_settings()
    n_images = sum(
        1
        for m in messages
        if isinstance(m.get("content"), list)
        for part in m["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    )
    log.info(
        "[vlm] request starting model=%s base=%s images=%d...",
        model_name(),
        settings.vlm_base_url,
        n_images,
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
        log.info("[vlm] strict json_schema unsupported (%s), retrying json_object mode", exc)
        content, finish = _chat_plain(client, messages)
        mode = "plain"
    except _EmptyResponse as exc:
        log.info("[vlm] strict mode returned empty (%s), retrying json_object mode", exc)
        content, finish = _chat_plain(client, messages)
        mode = "plain"
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 — auth, timeout, network: fail, don't retry
        raise LLMError(f"VLM request failed: {exc}") from exc
    log.info(
        "[vlm] response received in %.1fs (mode=%s finish=%s chars=%d)",
        time.perf_counter() - t0,
        mode,
        finish,
        len(content),
    )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMError(f"VLM did not return valid JSON: {exc}", raw=content) from exc
    if not isinstance(parsed, dict):
        raise LLMError("VLM response was not a JSON object", raw=content)
    return parsed
