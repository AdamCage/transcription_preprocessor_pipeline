"""Unit tests for PyannoteSegmenter inference logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from segmentation_service.inference import PyannoteSegmenter


class TestPyannoteSegmenter:
    def test_segment_not_loaded_raises(self, sample_wav_bytes):
        seg = PyannoteSegmenter(
            model_name="test", device="cpu", hf_token="", executor_workers=1
        )
        with pytest.raises(RuntimeError, match="not loaded"):
            seg.segment(sample_wav_bytes)

    def test_segment_with_fake_pipeline(self, sample_wav_bytes, fake_pyannote_pipeline):
        _FakePipeline, tracks = fake_pyannote_pipeline
        seg = PyannoteSegmenter(
            model_name="test", device="cpu", hf_token="", executor_workers=1
        )
        seg.load()
        segments, duration = seg.segment(sample_wav_bytes)

        assert duration > 0.9
        speech_segs = [s for s in segments if s.label == "speech"]
        assert len(speech_segs) == len(tracks)
        for s in speech_segs:
            assert s.start < s.end

    def test_segment_fills_gaps_with_non_speech(self, sample_wav_bytes, fake_pyannote_pipeline):
        seg = PyannoteSegmenter(
            model_name="test", device="cpu", hf_token="", executor_workers=1
        )
        seg.load()
        segments, duration = seg.segment(sample_wav_bytes)

        non_speech = [s for s in segments if s.label == "non_speech"]
        assert len(non_speech) >= 1

    def test_shutdown_clears_pipeline(self, fake_pyannote_pipeline):
        seg = PyannoteSegmenter(
            model_name="test", device="cpu", hf_token="", executor_workers=1
        )
        seg.load()
        seg.shutdown()
        assert seg._pipeline is None

    def test_empty_audio_returns_full_speech(self, empty_wav_bytes, fake_pyannote_pipeline):
        """When pyannote returns no tracks, fall back to full-file speech."""
        from tests.conftest import FakePyannoteOutput

        _FakePipeline, _ = fake_pyannote_pipeline
        _FakePipeline.__call__ = lambda self, x: FakePyannoteOutput([])

        seg = PyannoteSegmenter(
            model_name="test", device="cpu", hf_token="", executor_workers=1
        )
        seg.load()
        segments, _ = seg.segment(empty_wav_bytes)
        assert any(s.label == "speech" for s in segments)
