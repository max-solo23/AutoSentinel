from pydantic import BaseModel, Field
from typing import List


class PlateBBox(BaseModel):
    x1: float = Field(..., description="left")
    y1: float = Field(..., description="top")
    x2: float = Field(..., description="right")
    y2: float = Field(..., description="bottom")


class PlateResult(BaseModel):
    status: str
    plate_text: str
    confidence: float
    bbox: PlateBBox
