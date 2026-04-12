"""Shared fixtures for segmentation service tests."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture()
def sample_wav_bytes() -> bytes:
    """1-second 16kHz sine-wave WAV."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    signal = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture()
def short_wav_bytes() -> bytes:
    """100ms silence WAV."""
    sr = 16000
    signal = np.zeros(sr // 10, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture()
def empty_wav_bytes() -> bytes:
    """WAV with zero samples."""
    sr = 16000
    signal = np.zeros(0, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class FakePyannoteOutput:
    """Mimics pyannote Annotation .itertracks() output."""

    def __init__(self, tracks: list[tuple[float, float]]) -> None:
        self._tracks = tracks

    def itertracks(self):
        for start, end in self._tracks:
            seg = MagicMock()
            seg.start = start
            seg.end = end
            yield (seg,)


@pytest.fixture()
def fake_pyannote_pipeline(monkeypatch):
    """Monkeypatch pyannote Model + VoiceActivityDetection so no real model is loaded."""
    tracks = [(0.1, 0.5), (0.8, 1.0)]
    output = FakePyannoteOutput(tracks)

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, *a):
            return self

    class _FakeVADPipeline:
        def __init__(self, segmentation=None, **kw):
            pass

        def instantiate(self, params):
            pass

        def __call__(self, audio_input):
            return output

    import sys

    fake_audio = MagicMock()
    fake_audio.Model = _FakeModel
    fake_pipelines = MagicMock()
    fake_pipelines.VoiceActivityDetection = _FakeVADPipeline
    fake_audio.pipelines = fake_pipelines
    fake_audio.pipelines.VoiceActivityDetection = _FakeVADPipeline

    monkeypatch.setitem(sys.modules, "pyannote", MagicMock())
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)
    monkeypatch.setitem(sys.modules, "pyannote.audio.pipelines", fake_pipelines)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core", MagicMock())
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.pipeline", MagicMock())

    return _FakeVADPipeline, tracks
