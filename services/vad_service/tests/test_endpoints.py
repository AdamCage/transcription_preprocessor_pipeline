"""E2E endpoint tests for the VAD service."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from vad_service.config import ServiceConfig


def _make_wav(duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    signal = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture()
def client(fake_silero_model):
    config = ServiceConfig(
        device="cpu",
        max_concurrency=4,
        executor_workers=1,
        inference_timeout_sec=30.0,
    )
    from vad_service.app import _build_app

    app = _build_app(config)
    with TestClient(app) as c:
        yield c


class TestRefineEndpoint:
    def test_valid_request(self, client):
        wav = _make_wav(1.0)
        request_json = json.dumps(
            {
                "spans": [{"start": 0.0, "end": 1.0}],
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 200,
                "speech_pad_ms": 200,
                "merge_gap_seconds": 0.5,
            }
        )
        resp = client.post(
            "/refine",
            files={"audio": ("test.wav", wav, "audio/wav")},
            data={"request": request_json},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "spans" in data
        for sp in data["spans"]:
            assert sp["start"] < sp["end"]

    def test_empty_spans(self, client):
        wav = _make_wav(1.0)
        request_json = json.dumps({"spans": []})
        resp = client.post(
            "/refine",
            files={"audio": ("test.wav", wav, "audio/wav")},
            data={"request": request_json},
        )
        assert resp.status_code == 200
        assert resp.json()["spans"] == []

    def test_empty_audio_payload(self, client):
        request_json = json.dumps({"spans": [{"start": 0.0, "end": 1.0}]})
        resp = client.post(
            "/refine",
            files={"audio": ("test.wav", b"", "audio/wav")},
            data={"request": request_json},
        )
        assert resp.status_code == 422

    def test_invalid_request_json(self, client):
        wav = _make_wav(0.5)
        resp = client.post(
            "/refine",
            files={"audio": ("test.wav", wav, "audio/wav")},
            data={"request": "not valid json{"},
        )
        assert resp.status_code == 422

    def test_invalid_threshold(self, client):
        wav = _make_wav(0.5)
        request_json = json.dumps({"spans": [{"start": 0.0, "end": 0.5}], "threshold": 5.0})
        resp = client.post(
            "/refine",
            files={"audio": ("test.wav", wav, "audio/wav")},
            data={"request": request_json},
        )
        assert resp.status_code == 422

    def test_multiple_spans(self, client):
        wav = _make_wav(2.0)
        request_json = json.dumps(
            {
                "spans": [
                    {"start": 0.0, "end": 0.8},
                    {"start": 1.0, "end": 2.0},
                ],
            }
        )
        resp = client.post(
            "/refine",
            files={"audio": ("test.wav", wav, "audio/wav")},
            data={"request": request_json},
        )
        assert resp.status_code == 200
        assert len(resp.json()["spans"]) >= 1

    def test_payload_too_large(self, fake_silero_model):
        """Audio larger than max_audio_size_mb must be rejected with 413."""
        config = ServiceConfig(
            device="cpu",
            max_concurrency=4,
            executor_workers=1,
            max_audio_size_mb=0.001,
        )
        from vad_service.app import _build_app

        app = _build_app(config)
        with TestClient(app) as c:
            wav = _make_wav(1.0)
            request_json = json.dumps({"spans": [{"start": 0.0, "end": 1.0}]})
            resp = c.post(
                "/refine",
                files={"audio": ("test.wav", wav, "audio/wav")},
                data={"request": request_json},
            )
            assert resp.status_code == 413


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "silero_vad"
        assert data["device"] == "cpu"


class TestMetricsEndpoint:
    def test_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.text
        assert "vad_requests_total" in body
        assert "vad_request_duration_seconds" in body
