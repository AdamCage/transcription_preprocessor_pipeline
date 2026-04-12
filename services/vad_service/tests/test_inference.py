"""Unit tests for SileroVADGPU inference logic."""

from __future__ import annotations

import pytest

from vad_service.inference import SileroVADGPU, _merge_nearby_spans
from vad_service.models import TimeSpanIn, TimeSpanOut


class TestMergeNearbySpans:
    def test_empty(self):
        assert _merge_nearby_spans([], 0.5) == []

    def test_no_merge(self):
        spans = [TimeSpanOut(start=0.0, end=1.0), TimeSpanOut(start=2.0, end=3.0)]
        result = _merge_nearby_spans(spans, 0.5)
        assert len(result) == 2

    def test_merge_close(self):
        spans = [TimeSpanOut(start=0.0, end=1.0), TimeSpanOut(start=1.3, end=2.0)]
        result = _merge_nearby_spans(spans, 0.5)
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 2.0

    def test_merge_chain(self):
        spans = [
            TimeSpanOut(start=0.0, end=1.0),
            TimeSpanOut(start=1.2, end=2.0),
            TimeSpanOut(start=2.3, end=3.0),
        ]
        result = _merge_nearby_spans(spans, 0.5)
        assert len(result) == 1

    def test_single_span(self):
        spans = [TimeSpanOut(start=1.0, end=2.0)]
        result = _merge_nearby_spans(spans, 0.5)
        assert len(result) == 1


class TestSileroVADGPU:
    def test_not_loaded_raises(self, sample_wav_bytes):
        vad = SileroVADGPU(device="cpu", executor_workers=1)
        with pytest.raises(RuntimeError, match="not loaded"):
            vad.refine(
                sample_wav_bytes,
                [TimeSpanIn(start=0.0, end=1.0)],
            )

    def test_refine_basic(self, sample_wav_bytes, fake_silero_model):
        vad = SileroVADGPU(device="cpu", executor_workers=1)
        vad.load()
        result = vad.refine(
            sample_wav_bytes,
            [TimeSpanIn(start=0.0, end=1.0)],
        )
        assert len(result) >= 1
        for sp in result:
            assert sp.start < sp.end
            assert sp.start >= 0.0

    def test_refine_empty_spans(self, sample_wav_bytes, fake_silero_model):
        vad = SileroVADGPU(device="cpu", executor_workers=1)
        vad.load()
        result = vad.refine(sample_wav_bytes, [])
        assert result == []

    def test_refine_multiple_spans(self, sample_wav_bytes, fake_silero_model):
        vad = SileroVADGPU(device="cpu", executor_workers=1)
        vad.load()
        spans = [
            TimeSpanIn(start=0.0, end=0.5),
            TimeSpanIn(start=0.5, end=1.0),
        ]
        result = vad.refine(sample_wav_bytes, spans)
        assert len(result) >= 1

    def test_shutdown(self, fake_silero_model):
        vad = SileroVADGPU(device="cpu", executor_workers=1)
        vad.load()
        vad.shutdown()
        assert vad._model is None
        assert vad._get_speech_timestamps is None
