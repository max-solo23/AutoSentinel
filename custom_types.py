"""Typed response objects shared by FastAPI and tests."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlateBBox(BaseModel):
    """Axis-aligned bounding box describing a detected plate."""

    x1: float = Field(..., description="Left coordinate.")
    y1: float = Field(..., description="Top coordinate.")
    x2: float = Field(..., description="Right coordinate.")
    y2: float = Field(..., description="Bottom coordinate.")


class PlateResult(BaseModel):
    """Structured output returned by the pipeline."""

    status: str
    plate_text: str
    confidence: float
    bbox: PlateBBox


__all__ = ["PlateBBox", "PlateResult"]
