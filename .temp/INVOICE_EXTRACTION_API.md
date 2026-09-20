# Invoice Extraction — Request & Response

Standard (normal) invoice extraction contract between **core** (`servers/core`) and the **AI extraction service**.

Source: `servers/core/src/services/AIProcessor.service.ts` → `extractInvoice()` / `processInvoice()`,
callback handled by `invoiceController.processedDataCallback`.

---

## Flow

```
core ──(1) POST {AI_SERVICE_PROCESS_INVOICE_BASE_URL}/extract/invoice ──▶ AI service
     ◀──(2) sync ack (job accepted) ──────────────────────────────────────
                                   ... async extraction ...
     ◀──(3) POST {BASE_URL}/api/v1/invoices/processed-data ──────────────
core ──(4) 200 ack ────────────────────────────────────────────────────▶
```

- `unique_ref_no` is the **invoice id** as a string. It correlates request ↔ callback.
- Core schedules a 10-minute extraction timeout; if no callback arrives, the invoice is marked `failed`.

---

## 1. Request — `POST /extract/invoice`

`Content-Type: multipart/form-data`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file` | binary | yes | The invoice document (PDF/image) |
| `unique_ref_no` | string | yes | Invoice id as string |
| `fields` | JSON string | yes | Field mappings telling the AI what to extract |
| `callback_url` | string | yes | `{BASE_URL}/api/v1/invoices/processed-data` |
| `document_type` | string | no | `invoice`, `credit-note`, `freight`, … |
| `match` | JSON bool string | yes | `false` for non-matched document types, or when no matching conditions are enabled |
| `po_items` | JSON string | only when `match=true` | Reference document line items |
| `matching_conditions` | JSON string | only when `match=true` | Global enabled matching rules, priority ordered |
| `extraction_json` | JSON string | no | Previous extraction, for re-extract |
| `generic_extraction_hint` | string | no | Vendor-level hint, else system-level hint for the document type |

### `fields[]` — built by `AIProcessorService.getFields()` (disabled mappings filtered out)

```json
[
  {
    "unique_key": "invoiceNumber",
    "display_name": "Invoice Number",
    "extract_hint": "Top right of the document, may be labelled 'Tax Invoice No'",
    "transform_rule": "trim",
    "invoice_field": "Invoice No",
    "document_field": "Invoice No",
    "data_type": "string",
    "category": "header"
  },
  {
    "unique_key": "invoiceDate",
    "display_name": "Invoice Date",
    "extract_hint": "Date printed next to the invoice number",
    "transform_rule": "date:YYYY-MM-DD",
    "invoice_field": "Invoice Date",
    "document_field": "Invoice Date",
    "data_type": "date",
    "category": "header"
  },
  {
    "unique_key": "description",
    "display_name": "Description",
    "extract_hint": "Line item description column",
    "transform_rule": null,
    "invoice_field": "Description",
    "document_field": "Description",
    "data_type": "string",
    "category": "line_item"
  },
  {
    "unique_key": "quantity",
    "display_name": "Quantity",
    "extract_hint": "Qty column",
    "transform_rule": null,
    "invoice_field": "Qty",
    "document_field": "Qty",
    "data_type": "number",
    "category": "line_item"
  }
]
```

`data_type`: `string | number | date | boolean`. `category` comes from the mapping's `group` (`header` / `line_item`).

### `po_items[]` — `IPOItem[]`

```json
[
  {
    "item_number": "00010",
    "material_number": "MAT-1001",
    "material_description": "Steel Pipe 2in x 6m",
    "quantity": 100,
    "unit_price": 45.5,
    "total_price": 4550,
    "unit_of_measure": "EA",
    "delivery_date": "2026-09-01T00:00:00.000Z",
    "net_value": 4550,
    "material_group": "PIPES",
    "tax_amount": 227.5,
    "goods_delivered_quantity": 100,
    "open_invoice_quantity": 100,
    "po_number": "4500001234"
  }
]
```

### `matching_conditions[]`

```json
[
  {
    "priority": 1,
    "po_field": "material_number",
    "invoice_field_key": "itemCode",
    "match_type": "exact",
    "description": "Match PO material number against invoice item code"
  },
  {
    "priority": 2,
    "po_field": "material_description",
    "invoice_field_key": "description",
    "match_type": "fuzzy",
    "description": "Fallback description similarity match"
  }
]
```

`match_type`: `exact | partial | fuzzy | numeric`.
`invoice_field_key` must equal a `fields[].unique_key`, otherwise the rule can never fire.

### Full example (multipart, values shown as JSON)

```
file:                    invoice_4500001234.pdf   (binary)
unique_ref_no:           "1042"
document_type:           "invoice"
callback_url:            "https://api.example.com/api/v1/invoices/processed-data"
match:                   true
fields:                  [ ...see above... ]
po_items:                [ ...see above... ]
matching_conditions:     [ ...see above... ]
generic_extraction_hint: "Amounts are in SAR. Ignore the delivery address block."
```

---

## 2. Sync response

Returned by `axios.post(...)` and logged via `extractionLog.response("invoice", …)`. Core only acks the job here — it does **not** read extracted data from this response.

```json
{
  "status": "accepted",
  "unique_ref_no": "1042",
  "message": "Extraction queued"
}
```

Non-2xx → `extractionLog.error` and the error propagates; the invoice goes to `failed`.

---

## 3. Callback — `POST /api/v1/invoices/processed-data`

Body type: `ProcessedInvoiceResponseType` (`packages/types/src/schema/ai-tool/processedInvoiceResponse.ts`).
Stored verbatim as `invoices.extracted_data`.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `unique_ref_no` | string | yes | Invoice id; non-numeric → 400 |
| `status` | string | yes | e.g. `success` / `failed` |
| `key_value_pairs` | `IKeyValuePair[]` | yes | Header fields |
| `tables` | `ITableKeyValue[][]` | yes | Line items — array of rows, each row an array of `{key, value}` |
| `match_results` | `IMatchResult[]` | yes | Empty when `match=false` |
| `intermediate_data` | object | no | Only when the source document needed translation |

```json
{
  "unique_ref_no": "1042",
  "status": "success",
  "key_value_pairs": [
    { "key": "invoiceNumber", "value": "INV-2026-0091", "confidence": 0.98 },
    { "key": "invoiceDate",   "value": "2026-09-05",    "confidence": 0.95 },
    { "key": "dueDate",       "value": "2026-10-05",    "confidence": 0.91 },
    { "key": "poNumber",      "value": "4500001234",    "confidence": 0.99 },
    { "key": "vendorName",    "value": "Acme Trading LLC", "confidence": 0.93 },
    { "key": "currency",      "value": "SAR",           "confidence": 0.97 },
    { "key": "subtotal",      "value": "4550.00",       "confidence": 0.96 },
    { "key": "tax",           "value": "227.50",        "confidence": 0.96 },
    { "key": "total",         "value": "4777.50",       "confidence": 0.98 },
    { "key": "deliveryNoteNumber", "value": null,       "confidence": null }
  ],
  "tables": [
    [
      { "key": "itemCode",    "value": "MAT-1001" },
      { "key": "description", "value": "Steel Pipe 2in x 6m" },
      { "key": "quantity",    "value": "100" },
      { "key": "unit",        "value": "EA" },
      { "key": "unitPrice",   "value": "45.50" },
      { "key": "amount",      "value": "4550.00" },
      { "key": "vat",         "value": "5" }
    ],
    [
      { "key": "itemCode",    "value": "MAT-2002" },
      { "key": "description", "value": "Pipe Clamp 2in" },
      { "key": "quantity",    "value": "50" },
      { "key": "unit",        "value": "EA" },
      { "key": "unitPrice",   "value": "3.20" },
      { "key": "amount",      "value": "160.00" },
      { "key": "vat",         "value": "5" }
    ]
  ],
  "match_results": [
    {
      "score": 4.8,
      "best_match": "Steel Pipe 2in x 6m",
      "invoice_item": "Steel Pipe 2in x 6m",
      "po_usage_count": 1,
      "po_field": "material_description",
      "invoice_field": "description",
      "invoice_document_field": "Description",
      "match_type": "fuzzy",
      "matching_method": "crossencoder",
      "priority": 2
    },
    {
      "score": 5,
      "best_match": "MAT-2002",
      "invoice_item": "Pipe Clamp 2in",
      "po_usage_count": 1,
      "po_field": "material_number",
      "invoice_field": "itemCode",
      "invoice_document_field": "Item Code",
      "match_type": "exact",
      "matching_method": "exact",
      "priority": 1
    }
  ]
}
```

Notes:
- `key_value_pairs[].key` and `tables[][].key` are the `fields[].unique_key` values sent in the request.
- `value` is always a string (or `null` for header fields the AI could not find); confidence `0..1` or `null`.
- `score` in `match_results` is out of 5.
- `intermediate_data` (optional) carries the bilingual raw extraction:

```json
{
  "intermediate_data": {
    "document_type": "invoice",
    "currency": "SAR",
    "key_value_pairs": [
      {
        "key": "رقم الفاتورة",
        "value": "INV-2026-0091",
        "translated_key": "Invoice Number",
        "translated_value": "INV-2026-0091",
        "confidence": "0.98"
      }
    ],
    "tables": [
      {
        "table_name": "Line Items",
        "table_data": { "headers": ["الوصف", "الكمية"], "rows": [["أنبوب فولاذي", "100"]] },
        "translated_table_data": { "headers": ["Description", "Quantity"], "rows": [["Steel Pipe", "100"]] }
      }
    ]
  }
}
```

---

## 4. Core's response to the callback

```json
{
  "version": "v1",
  "success": true,
  "data": {
    "success": true,
    "message": "Invoice data processed successfully",
    "invoice_id": 1042
  }
}
```

Errors (`BadRequestError`, HTTP 400):
- `unique_ref_no is required`
- `Invalid unique_ref_no`

---

## Related endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST {AI}/extract/invoice` | Standard invoice extraction (this doc) |
| `POST {AI}/extract/expense` | Expense invoice — vendor-less, callback `/api/v1/invoices/expense-processed-data` |
| `POST {AI}/extract/po` · `/extract/grn` · `/extract` | PO / GRN / generic document extraction |
| `POST {AI}/extract/vendor` | Vendor identification, callback `/api/v1/invoices/vendor-extraction-callback` |
| `POST {AI}/match` | Standalone matching pass, callback `/api/v1/invoices/match-callback` |
