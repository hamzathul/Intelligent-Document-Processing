"""Versioned router aggregation."""

from fastapi import APIRouter

from app.api.v1.invoice import router as invoice_router
from app.api.v1.invoice_vlm import router as invoice_vlm_router

api_router = APIRouter()
api_router.include_router(invoice_router)
api_router.include_router(invoice_vlm_router)
