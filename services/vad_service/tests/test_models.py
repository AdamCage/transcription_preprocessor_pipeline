"""Unit tests for VAD service Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vad_service.models import (
    HealthResponse,
    RefineRequest,
    RefineResponse,
    TimeSpanIn,
    TimeSpanOut,
)


class TestTimeSpanIn:
    def test_valid(self):
        ts = TimeSpanIn(start=0.5, end=2.0)
        assert ts.start == 0.5
        assert ts.end == 2.0

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            TimeSpanIn(start=-1.0, end=1.0)


class TestRefineRequest:
    def test_defaults(self):
        r = RefineRequest(spans=[TimeSpanIn(start=0.0, end=1.0)])
        assert r.threshold == 0.5
        assert r.min_speech_duration_ms == 250
        assert r.min_silence_duration_ms == 200
        assert r.speech_pad_ms == 200
        assert r.merge_gap_seconds == 0.5

    def test_empty_spans(self):
        r = RefineRequest(spans=[])
        assert r.spans == []

    def test_invalid_threshold(self):
        with pytest.raises(ValidationError):
            RefineRequest(spans=[], threshold=1.5)

    def test_negative_min_speech_duration(self):
        with pytest.raises(ValidationError):
            RefineRequest(spans=[], min_speech_duration_ms=-1)


class TestRefineResponse:
    def test_valid(self):
        r = RefineResponse(spans=[TimeSpanOut(start=0.1, end=0.9)])
        assert len(r.spans) == 1

    def test_empty(self):
        r = RefineResponse(spans=[])
        assert r.spans == []


class TestHealthResponse:
    def test_minimal(self):
        h = HealthResponse(status="ok", model="silero_vad", device="cpu")
        assert h.gpu_memory_used_mb is None
