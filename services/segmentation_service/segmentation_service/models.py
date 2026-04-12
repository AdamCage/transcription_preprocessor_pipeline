"""Pydantic request / response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SegmentItem(BaseModel):
    start: float = Field(..., ge=0.0, description="Segment start in seconds")
    end: float = Field(..., ge=0.0, description="Segment end in seconds")
    label: Literal["speech", "non_speech"] = "speech"


class SegmentResponse(BaseModel):
    segments: list[SegmentItem]
    duration_sec: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
