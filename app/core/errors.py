"""Domain errors mapped to HTTP responses.

Routes/services raise these instead of building ``HTTPException`` inline,
so error semantics stay consistent. Registered on the app via
:func:`register_exception_handlers`.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    status_code: int = 500
    detail: str = "Internal server error"

    def __init__(self, detail: str | None = None) -> None:
        if detail:
            self.detail = detail
        super().__init__(self.detail)


class BadRequestError(AppError):
    status_code = 400


class PayloadTooLargeError(AppError):
    status_code = 413


class OcrFailedError(AppError):
    status_code = 500


class LlmNotConfiguredError(AppError):
    status_code = 503


class LlmUpstreamError(AppError):
    status_code = 502


async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)  # type: ignore[arg-type]
