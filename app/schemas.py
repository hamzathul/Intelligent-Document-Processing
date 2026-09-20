from pydantic import BaseModel, Field


class OcrLine(BaseModel):
    text: str
    score: float = 0.0
    box: list[list[float]] | None = None


class OcrPage(BaseModel):
    page_index: int = 0
    text: str = ""
    lines: list[OcrLine] = Field(default_factory=list)
    mean_score: float = 0.0


class OcrResponse(BaseModel):
    filename: str
    model: str = "PP-OCRv6_medium"
    engine: str = "paddle"
    pages: list[OcrPage] = Field(default_factory=list)
    full_text: str = ""
    time_s: float = 0.0


class HealthResponse(BaseModel):
    status: str
    model: str = "PP-OCRv6_medium"
    engine: str = "paddle"
    device: str = "cpu"
    ocr_loaded: bool = False
    detail: str | None = None


ExtractedValue = str | float | int | bool | None


class ExtractResponse(BaseModel):
    filename: str
    headers: dict[str, ExtractedValue] = Field(default_factory=dict)
    line_items: list[dict[str, ExtractedValue]] = Field(default_factory=list)
    ocr_text: str = ""
    model: str = ""
    time_s: float = 0.0
