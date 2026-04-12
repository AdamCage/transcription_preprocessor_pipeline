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
    """Mimics pyannote Timeline/Annotation .itertracks() output."""

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
    """Monkeypatch pyannote Pipeline so no real model is loaded."""
    tracks = [(0.1, 0.5), (0.8, 1.0)]
    output = FakePyannoteOutput(tracks)

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        def __call__(self, audio_input):
            return output

    monkeypatch.setattr(
        "segmentation_service.inference.Pipeline",
        FakePipeline,
        raising=False,
    )
    import sys

    fake_module = MagicMock()
    fake_module.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)
    return FakePipeline, tracks
