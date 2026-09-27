# IDP OCR — Intelligent Document Processing

Local OCR + LLM-guided invoice extraction API.

Pipeline: **PaddleOCR PP-OCRv6 (local inference)** → **OpenAI-compatible LLM (extraction)** → **PO matching (local)**. Served via FastAPI, synchronously — no callbacks, no queues.

## Features

- `POST /api/ocr` — raw OCR (per-line text + confidence + bounding boxes).
- `POST /api/extract` — guided extraction with a caller-defined spec (legacy/generic).
- `POST /extract/invoice` (alias: `POST /api/v1/extract/invoice`) — sync invoice contract: OCR → LLM extract → PO match, returned inline.
- `POST /extract/invoice/vlm` (alias: `POST /api/v1/extract/invoice/vlm`) — sync direct-vision contract: page images → VLM extract → PO match, no OCR. Same request/response shape as `/extract/invoice`, minus `page_ranges` (all pages always read).
- PO matching with `exact / partial / fuzzy / numeric` conditions and priority resolution.
- Per-request debug traces under `output/debug/`.
- Interactive docs at `GET /docs`.

## Requirements

- Python `>= 3.12`
- [uv](https://docs.astral.sh/uv/) `>= 0.11` (dependency + run manager)
- Windows, macOS, or Linux, CPU is fine (default `OCR_DEVICE=cpu`)
- An OpenAI-compatible LLM endpoint + API key (only needed for `/api/extract` and `/extract/invoice`; plain `/api/ocr` works without it). Local Ollama works too.

## Quick start

```powershell
# 1. Install uv (Windows PowerShell), then restart the shell
irm https://astral.sh/uv/install.ps1 | iex

# 2. Configure
Copy-Item .env.example .env
# edit .env -> set LLM_API_KEY (and LLM_BASE_URL / LLM_MODEL if needed)

# 3. Run (double-clickable alternatives: start.bat / start.ps1)
uv run python -m app
```

Open:

- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/health`

First run downloads the PP-OCRv6 model (~hundreds of MB) and warms it up at startup, so the first request is slow. Subsequent requests are fast.

### Run options

```powershell
uv run python -m app --host 127.0.0.1 --port 8000
# or via env
$env:HOST="0.0.0.0"; $env:PORT="8000"; $env:RELOAD="1"; uv run python -m app
```

| Var | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind address (`0.0.0.0` for LAN/docker) |
| `PORT` | `8000` | Port |
| `RELOAD` | off | `1` enables uvicorn auto-reload (dev only) |

## Configuration (`.env`)

Copy `.env.example` → `.env`. All settings have defaults; `LLM_API_KEY` is required for OCR-path extraction (`/api/extract`, `/extract/invoice`), `GOOGLE_API_KEY` for direct-VLM extraction (`/extract/invoice/vlm`).

```dotenv
OCR_MODEL=PP-OCRv6_medium
OCR_ENGINE=paddle
OCR_DEVICE=cpu
OCR_CPU_THREADS=4
OCR_MAX_MB=15

LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT_S=60

# Direct-VLM extraction (POST /extract/invoice/vlm) — vision model, no OCR.
# Google AI Studio (Gemini Developer API). Get a key at https://aistudio.google.com/apikey
# VLM_BASE_URL / VLM_API_KEY / VLM_MODEL override the GOOGLE_* / GEMINI_* names when set.
VLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
VLM_TIMEOUT_S=120
VLM_MAX_PAGES=8
VLM_MAX_SIDE_PX=1568
VLM_JPEG_QUALITY=85

IDP_DEBUG_TRACE=1
IDP_DEBUG_DIR=output/debug
IDP_DEBUG_KEEP=100
# IDP_DEBUG_SAVE_UPLOADS=1  # also store uploaded file as 00_upload.* (off by default)
```

Local Ollama example:

```dotenv
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:7b-instruct
```

Notes:

- Upload limits: max `OCR_MAX_MB` (default 15 MB). Allowed extensions: `.png .jpg .jpeg .bmp .tiff .tif .webp .pdf`.
- `PaddleOCR` is installed from the CPU index pinned in `pyproject.toml` (`paddleocr==3.7.0`, `paddlepaddle==3.2.0`). No manual wheel handling needed — `uv sync` / `uv run` resolves it.
- Never commit `.env` (already in `.gitignore`).

## API

### `GET /api/health`

```json
{"status":"ok","model":"PP-OCRv6_medium","engine":"paddle","device":"cpu","ocr_loaded":true}
```

### `POST /api/ocr` — raw OCR

Multipart `file`, optional query `page_ranges` (e.g. `?page_ranges=1-3,5`, PDF only).

```powershell
curl -X POST "http://127.0.0.1:8000/api/ocr" `
  -F "file=@samples/invoices/synthetic_invoice.png"
```

Response: `{filename, model, engine, pages: [{page_index, text, lines: [{text, score, box}], mean_score}], full_text, time_s}`.

### `POST /api/extract` — guided extraction (generic spec)

Multipart `file` + `spec` (JSON string), optional `?page_ranges`. See `samples/invoices/spec.example.json` and `tegan_spec.json`.

Spec shape:

```json
{
  "headers": [{"key": "total", "field": "Total", "type": "number", "hint": "..."}],
  "line_items": [{"key": "qty", "field": "QTY", "type": "number"}]
}
```

- `key`: `[A-Za-z][A-Za-z0-9_]{0,63}`, unique within `headers` / `line_items`; max 50 fields total.
- `type`: `string | number | integer | date | boolean`.
- `hint`: free-text help for the LLM (handles mislabeled OCR, computed dates, multi-line joins).

```powershell
$spec = Get-Content samples/invoices/spec.example.json -Raw
curl -X POST "http://127.0.0.1:8000/api/extract" `
  -F "file=@samples/invoices/synthetic_invoice.png" `
  -F "spec=$spec"
```

Requires `LLM_API_KEY` (else HTTP `503`).

### `POST /extract/invoice` — invoice contract (main endpoint)

Also at `POST /api/v1/extract/invoice`. Synchronous: returns the processed payload directly. `callback_url` is accepted but ignored.

Multipart form fields:

| Field | Required | Description |
|---|---|---|
| `file` | yes | Invoice PDF/image |
| `unique_ref_no` | yes | Invoice id (string, non-blank) |
| `fields` | yes | JSON array of field mappings (see below) |
| `match` | yes | `"true"` / `"false"` (also accepts `1/0/yes/no/on/off`) |
| `po_items` | if `match=true` | JSON array of PO objects |
| `matching_conditions` | if `match=true` | JSON array; each `invoice_field_key` must exist in `fields[].unique_key` |
| `document_type` | no | Passed through to the LLM prompt |
| `generic_extraction_hint` | no | Global hint appended to the LLM prompt |
| `extraction_json` | no | Prior extraction object (must be a JSON object); used as context |
| `callback_url` | no | Ignored (compat only) |
| `?page_ranges` | no | Query param, e.g. `1-3,5` |

Field mapping entry (`samples/invoices/invoice_fields.full.json`):

```json
{
  "unique_key": "invoiceNumber",
  "display_name": "Invoice Number",
  "extract_hint": "Top right, may be labelled 'Tax Invoice No'",
  "invoice_field": "Invoice No",
  "document_field": "Invoice No",
  "data_type": "string",
  "category": "header"
}
```

- `data_type`: `string | number | date | boolean | integer`. `category`: `header | line_item`.
- `unique_key` must be unique *within* its category (same key may appear once as a header total and once as a line value, e.g. GST).
- `transform_rule` accepted but ignored.
- Matching condition: `{"priority": 1, "po_field": "material_number", "invoice_field_key": "itemCode", "match_type": "exact"}` with `match_type ∈ exact | partial | fuzzy | numeric`. Lowest `priority` number wins per row.

Examples:

```powershell
# match=false (extract only)
curl -X POST "http://127.0.0.1:8000/extract/invoice" `
  -F "file=@samples/invoices/synthetic_invoice.png" `
  -F "unique_ref_no=1042" `
  -F "fields=@samples/invoices/invoice_fields.full.json;type=application/json" `
  -F "match=false"

# match=true (extract + PO match)
curl -X POST "http://127.0.0.1:8000/extract/invoice" `
  -F "file=@invoice.pdf" `
  -F "unique_ref_no=1042" `
  -F "fields=@fields.json;type=application/json" `
  -F "match=true" `
  -F "po_items=@po.json;type=application/json" `
  -F "matching_conditions=@conditions.json;type=application/json"
```

Response shape:

```json
{
  "unique_ref_no": "1042",
  "status": "success",
  "key_value_pairs": [{"key": "total", "value": "4777.5", "confidence": null}],
  "tables": [[{"key": "description", "value": "Steel Pipe"}, {"key": "quantity", "value": "100.0"}]],
  "match_results": [{"score": 5.0, "best_match": "...", "po_field": "...", "invoice_field": "...", "match_type": "fuzzy", "matching_method": "...", "priority": 1, "po_usage_count": 1}],
  "intermediate_data": {"ocr_chars": 1234, "model": "gpt-4o-mini"}
}
```

Header/table `value`s are strings (numbers stringified, e.g. `"4777.5"`).

### `POST /extract/invoice/vlm` — direct-vision invoice contract (no OCR)

Also at `POST /api/v1/extract/invoice/vlm`. Same multipart contract and same
response shape as `POST /extract/invoice`, except:

- No PaddleOCR step — the VLM reads page images directly.
- No `?page_ranges` — **all** pages are always rendered and sent (PDFs via
  `pypdfium2`, downscaled to `VLM_MAX_SIDE_PX` JPEG). Documents with more than
  `VLM_MAX_PAGES` pages are rejected with `400` (split the file).
- `intermediate_data` carries `{"model": ..., "mode": "vlm-direct", "pages": N}`.

```powershell
curl -X POST "http://127.0.0.1:8000/extract/invoice/vlm" `
  -F "file=@samples/invoices/synthetic_invoice.png" `
  -F "unique_ref_no=1042" `
  -F "fields=@samples/invoices/invoice_fields.full.json;type=application/json" `
  -F "match=false"
```

Requires `GOOGLE_API_KEY` (AI Studio key, else HTTP `503`). Default model
`GEMINI_MODEL=gemini-2.5-flash` via Google's OpenAI-compatible endpoint.

### Errors

| Code | Meaning |
|---|---|
| `400` | Bad file type / empty file, invalid JSON, spec/fields validation (dupe keys, unknown `invoice_field_key`, `match=true` missing `po_items`), blank `unique_ref_no`, bad `match` flag |
| `413` | File larger than `OCR_MAX_MB` |
| `500` | OCR inference failed |
| `502` | LLM request / bad JSON output |
| `503` | `LLM_API_KEY` not set |

## Project structure

```
app/
  main.py            # FastAPI factory, legacy /api/ocr + /api/extract routes
  __main__.py        # `uv run python -m app` entrypoint
  infra/             # external-service clients (canonical implementations)
    llm_client.py    # OpenAI-compatible text client (strict json_schema -> json_object fallback)
    vlm_client.py    # OpenAI-compatible vision client (page images -> JSON)
    ocr_service.py   # PaddleOCR singleton (PP-OCRv6), parse_result normalizer
  llm_client.py      # alias -> app.infra.llm_client (backward compat)
  ocr_service.py     # alias -> app.infra.ocr_service (backward compat)
  extract_service.py # generic spec orchestrator (OCR pages -> prompt -> LLM -> coerce)
  prompt.py  coerce.py  fields.py
  api/deps.py        # upload validation (type + size)
  api/v1/forms.py    # shared multipart validators (both invoice routes)
  api/v1/invoice.py  # POST /extract/invoice (OCR -> LLM -> match)
  api/v1/invoice_vlm.py  # POST /extract/invoice/vlm (page images -> VLM -> match, no OCR)
  api/v1/router.py
  dtos/invoice.py    # FieldMapping / MatchingCondition / ProcessedInvoiceResponse
  services/pipeline.py       # invoice extract pipeline (OCR text path)
  services/vlm_pipeline.py   # direct-VLM extract pipeline (page images path)
  services/images.py         # upload bytes -> vision-ready JPEG pages (PDF via pypdfium2)
  services/vlm_prompt.py     # vision message builder (spec + page images)
  services/matcher.py        # PO matching (exact/partial/fuzzy/numeric + priority)
  services/field_adapter.py
  core/config.py     # Settings (env -> defaults, incl. VLM_*)
  core/debug_trace.py  core/errors.py  core/logging.py
  utils/files.py
samples/invoices/    # spec.example.json, tegan_spec.json, invoice_fields.full.json, synthetic_invoice.png
tests/               # pytest suite (OCR/LLM stubbed where needed)
output/debug/        # per-request traces (gitignored)
```

Debug trace per request (`output/debug/<ref>_<timestamp>/`): `01_request.json`, `02_ocr.json/.txt`, `03_llm_request`, `04_llm_response`, `05_extracted`, `06_match`, `07_response`, `08_meta`, `error.json` on failure. Latest `IDP_DEBUG_KEEP` (default 100) retained.

## Development & tests

```powershell
uv sync            # install deps
uv run pytest -q   # full suite (offline tests stub OCR/LLM; no key needed)
uv run pytest tests/test_invoice.py -q
```

## Troubleshooting

- `503 LLM_API_KEY is not set` → copy `.env.example` to `.env` and set the key; restart the server (settings are read at startup).
- First request slow / model downloading → expected; PP-OCRv6 weights download once, then warm up on startup (`PP-OCRv6 warmed up` in logs).
- `413 File too large` → raise `OCR_MAX_MB` in `.env`.
- `400 Unsupported file type` → use png/jpg/bmp/tiff/webp/pdf.
- LLM `400/422` on strict schema → handled automatically (falls back to `json_object` mode); empty responses from free-tier routers surface as `502` with a hint to switch `LLM_MODEL`.
- Ollama not reached → set `LLM_BASE_URL=http://localhost:11434/v1`, ensure `ollama serve` is running.
- Stale settings in tests → `get_settings.cache_clear()` after changing env.
