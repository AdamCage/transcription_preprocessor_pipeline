"""Unit tests for segmentation service Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from segmentation_service.models import HealthResponse, SegmentItem, SegmentResponse


class TestSegmentItem:
    def test_valid_speech(self):
        s = SegmentItem(start=0.0, end=1.5, label="speech")
        assert s.start == 0.0
        assert s.end == 1.5
        assert s.label == "speech"

    def test_valid_non_speech(self):
        s = SegmentItem(start=1.5, end=3.0, label="non_speech")
        assert s.label == "non_speech"

    def test_default_label(self):
        s = SegmentItem(start=0.0, end=1.0)
        assert s.label == "speech"

    def test_negative_start_rejected(self):
        with pytest.raises(ValidationError):
            SegmentItem(start=-1.0, end=1.0, label="speech")

    def test_invalid_label_rejected(self):
        with pytest.raises(ValidationError):
            SegmentItem(start=0.0, end=1.0, label="music")


class TestSegmentResponse:
    def test_valid(self):
        r = SegmentResponse(
            segments=[SegmentItem(start=0.0, end=1.0)],
            duration_sec=1.0,
        )
        assert len(r.segments) == 1
        assert r.duration_sec == 1.0

    def test_empty_segments(self):
        r = SegmentResponse(segments=[], duration_sec=0.0)
        assert r.segments == []

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError):
            SegmentResponse(segments=[], duration_sec=-1.0)


class TestHealthResponse:
    def test_minimal(self):
        h = HealthResponse(status="ok", model="test", device="cpu")
        assert h.gpu_memory_used_mb is None

    def test_with_gpu(self):
        h = HealthResponse(
            status="ok",
            model="test",
            device="cuda:0",
            gpu_memory_used_mb=512.0,
            gpu_memory_total_mb=16384.0,
        )
        assert h.gpu_memory_total_mb == 16384.0
