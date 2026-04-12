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

    def test_model_pool_returns_replica(self, fake_silero_model):
        """After refine(), the model replica is returned to the pool."""
        vad = SileroVADGPU(device="cpu", executor_workers=2)
        vad.load()
        assert vad._model_pool.qsize() == 2

    def test_reset_states_called(self, sample_wav_bytes, fake_silero_model):
        """reset_states() is called before every inference."""
        vad = SileroVADGPU(device="cpu", executor_workers=1)
        vad.load()
        vad.refine(sample_wav_bytes, [TimeSpanIn(start=0.0, end=1.0)])
        model, _ = vad._model_pool.get()
        model.reset_states.assert_called()

    def test_parallel_spans(self, sample_wav_bytes, fake_silero_model):
        """With executor_workers > 1, multiple spans are processed in parallel."""
        vad = SileroVADGPU(device="cpu", executor_workers=2)
        vad.load()
        spans = [
            TimeSpanIn(start=0.0, end=0.3),
            TimeSpanIn(start=0.3, end=0.6),
            TimeSpanIn(start=0.6, end=1.0),
        ]
        result = vad.refine(sample_wav_bytes, spans)
        assert len(result) >= 1
        assert vad._model_pool.qsize() == 2

    def test_shutdown(self, fake_silero_model):
        vad = SileroVADGPU(device="cpu", executor_workers=1)
        vad.load()
        vad.shutdown()
        assert vad._model_pool.empty()
