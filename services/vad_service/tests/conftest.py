"""Shared fixtures for VAD service tests."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
import torch


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


def _make_fake_silero_pair():
    """Build a (fake_model, fake_get_speech_timestamps) pair."""

    def fake_get_speech_timestamps(wav_tensor, model, **kwargs):
        n_samples = wav_tensor.shape[0]
        if n_samples < 100:
            return []
        return [
            {"start": 100, "end": int(n_samples * 0.4)},
            {"start": int(n_samples * 0.6), "end": n_samples - 100},
        ]

    fake_model = MagicMock()
    fake_model.to = MagicMock(return_value=fake_model)
    fake_model.eval = MagicMock(return_value=fake_model)
    fake_model.reset_states = MagicMock()

    return fake_model, fake_get_speech_timestamps


@pytest.fixture()
def fake_silero_model(monkeypatch):
    """Monkeypatch torch.hub.load to return a fake Silero VAD model.

    Each call to ``torch.hub.load`` returns a fresh pair so that the
    model-pool in ``SileroVADGPU`` gets independent replicas.
    """
    def fake_hub_load(*args, **kwargs):
        model, get_ts = _make_fake_silero_pair()
        return model, [get_ts]

    monkeypatch.setattr("torch.hub.load", fake_hub_load)
    pair = _make_fake_silero_pair()
    return pair[0], pair[1]
