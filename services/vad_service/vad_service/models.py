"""Pydantic request / response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TimeSpanIn(BaseModel):
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)


class RefineRequest(BaseModel):
    spans: list[TimeSpanIn]
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(250, ge=0)
    min_silence_duration_ms: int = Field(200, ge=0)
    speech_pad_ms: int = Field(200, ge=0)
    merge_gap_seconds: float = Field(0.5, ge=0.0)


class TimeSpanOut(BaseModel):
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)


class RefineResponse(BaseModel):
    spans: list[TimeSpanOut]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
