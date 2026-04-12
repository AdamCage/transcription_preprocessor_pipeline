"""E2E endpoint tests for the segmentation service."""

from __future__ import annotations

import asyncio
import io
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from segmentation_service.config import ServiceConfig
from segmentation_service.models import SegmentItem


def _make_wav(duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    signal = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture()
def client(fake_pyannote_pipeline):
    config = ServiceConfig(
        model_name="test",
        hf_token="",
        device="cpu",
        max_concurrency=4,
    )
    from segmentation_service.app import _build_app

    app = _build_app(config)
    with TestClient(app) as c:
        yield c


class TestSegmentEndpoint:
    def test_valid_wav(self, client):
        wav = _make_wav(1.0)
        resp = client.post("/segment", files={"audio": ("test.wav", wav, "audio/wav")})
        assert resp.status_code == 200
        data = resp.json()
        assert "segments" in data
        assert data["duration_sec"] > 0
        for seg in data["segments"]:
            assert seg["label"] in ("speech", "non_speech")
            assert seg["start"] < seg["end"]

    def test_empty_payload(self, client):
        resp = client.post("/segment", files={"audio": ("test.wav", b"", "audio/wav")})
        assert resp.status_code == 422

    def test_invalid_content_type(self, client):
        resp = client.post(
            "/segment",
            files={"audio": ("test.txt", b"not audio", "text/plain")},
        )
        assert resp.status_code == 422

    def test_short_audio(self, client):
        wav = _make_wav(0.1)
        resp = client.post("/segment", files={"audio": ("short.wav", wav, "audio/wav")})
        assert resp.status_code == 200
        assert len(resp.json()["segments"]) >= 1

    def test_multiple_requests_sequential(self, client):
        wav = _make_wav(0.5)
        for _ in range(5):
            resp = client.post("/segment", files={"audio": ("t.wav", wav, "audio/wav")})
            assert resp.status_code == 200


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "test"
        assert data["device"] == "cpu"
