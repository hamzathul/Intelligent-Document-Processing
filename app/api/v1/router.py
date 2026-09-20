"""Versioned router aggregation."""

from fastapi import APIRouter

from app.api.v1.invoice import router as invoice_router

api_router = APIRouter()
api_router.include_router(invoice_router)
