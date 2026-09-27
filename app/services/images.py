"""Upload bytes → vision-ready page images.

Pure functions (no FastAPI, no network) so they stay unit-testable. The VLM
route sends **all** pages — PDFs are rasterized with ``pypdfium2`` (already a
project dependency), single images pass through — then each page is
downscaled with Pillow and encoded as a ``data:`` URL for OpenAI-style
``image_url`` message parts.

No ``page_ranges`` support by design: the VLM route always reads the whole
document (up to ``VLM_MAX_PAGES``).
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass


@dataclass(frozen=True)
class PageImage:
    """One vision-ready page."""

    page_index: int
    mime: str  # always image/jpeg after normalization
    data_url: str  # data:image/jpeg;base64,...
    width: int
    height: int
    source_bytes: int = 0  # encoded JPEG size, for debug traces


class TooManyPagesError(ValueError):
    """Raised when a document exceeds the configured page cap."""


class ImageRenderError(ValueError):
    """Raised when bytes cannot be decoded as an image or PDF."""


def _downscale_jpeg(raw: bytes, *, max_side_px: int, quality: int) -> tuple[bytes, int, int]:
    """Decode any image bytes, fit longest side to ``max_side_px``, return JPEG bytes + dims."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(raw)) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "PA"):
            canvas = Image.new("RGB", img.size, (255, 255, 255))
            alpha = img.split()[-1] if img.mode != "P" else None
            canvas.paste(img.convert("RGB"), mask=alpha)
            img = canvas
        else:
            img = img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > max_side_px and longest > 0:
            scale = max_side_px / longest
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue(), img.size[0], img.size[1]


def _render_pdf_pages(blob: bytes, *, scale: float = 2.0) -> list[bytes]:
    """Rasterize every PDF page to PNG bytes (caller downscales to JPEG)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(blob)
    try:
        pages: list[bytes] = []
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                bitmap = page.render(scale=scale)
                pil = bitmap.to_pil()
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                pages.append(buf.getvalue())
            finally:
                page.close()
        return pages
    finally:
        pdf.close()


def _is_pdf(suffix: str, blob: bytes) -> bool:
    if suffix.lower() == ".pdf":
        return True
    return blob[:5] == b"%PDF-"


def render_pages(
    blob: bytes,
    suffix: str,
    *,
    max_pages: int,
    max_side_px: int,
    jpeg_quality: int,
) -> list[PageImage]:
    """Normalize an upload into vision-ready JPEG pages.

    Raises :class:`TooManyPagesError` (caller maps to HTTP 400) and
    :class:`ImageRenderError` (caller maps to HTTP 500).
    """
    if not blob:
        raise ImageRenderError("Empty file")
    try:
        if _is_pdf(suffix, blob):
            raws = _render_pdf_pages(blob)
        else:
            raws = [blob]
    except TooManyPagesError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ImageRenderError(f"Could not decode upload as image/PDF: {exc}") from exc

    if len(raws) > max_pages:
        raise TooManyPagesError(
            f"Document has {len(raws)} pages, max {max_pages} for the VLM route — "
            "split the file into smaller documents."
        )
    pages: list[PageImage] = []
    for idx, raw in enumerate(raws):
        try:
            jpeg, w, h = _downscale_jpeg(raw, max_side_px=max_side_px, quality=jpeg_quality)
        except Exception as exc:  # noqa: BLE001
            raise ImageRenderError(f"Could not decode page {idx + 1}: {exc}") from exc
        b64 = base64.b64encode(jpeg).decode("ascii")
        pages.append(
            PageImage(
                page_index=idx,
                mime="image/jpeg",
                data_url=f"data:image/jpeg;base64,{b64}",
                width=w,
                height=h,
                source_bytes=len(jpeg),
            )
        )
    if not pages:
        raise ImageRenderError("No pages found in upload")
    return pages


def describe_for_trace(pages: list[PageImage]) -> list[dict]:
    """Trace-safe summary — metadata only, never full base64 payloads."""
    return [
        {
            "page_index": p.page_index,
            "mime": p.mime,
            "width": p.width,
            "height": p.height,
            "jpeg_bytes": p.source_bytes,
            "data_url_preview": p.data_url[:120] + "...",
        }
        for p in pages
    ]
