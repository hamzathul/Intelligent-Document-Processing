"""idp-ocr package. Loads `.env` first so local config (LLM keys, OCR
tuning) works no matter how the app is launched (`uv run`, plain uvicorn,
tests, scripts). Shell-exported variables always win over `.env` values."""

from dotenv import load_dotenv

load_dotenv()
