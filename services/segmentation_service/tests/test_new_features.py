"""Tests for production hardening features: size limits, timeouts, probes, metrics."""

from __future__ import annotations

import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from segmentation_service.config import ServiceConfig
from segmentation_service.inference import PyannoteSegmenter


def _make_wav(duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    signal = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture()
def strict_config() -> ServiceConfig:
    """Config with tight limits for testing."""
    return ServiceConfig(
        model_name="test",
        hf_token="",
        device="cpu",
        max_concurrency=2,
        inference_timeout_sec=2.0,
        max_audio_bytes=1024,
        max_audio_duration_sec=0.5,
    )


@pytest.fixture()
def strict_client(fake_pyannote_pipeline, strict_config):
    from segmentation_service.app import _build_app

    app = _build_app(strict_config)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def normal_client(fake_pyannote_pipeline):
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


class TestAudioSizeLimit:
    def test_oversized_payload_rejected(self, strict_client):
        wav = _make_wav(1.0)
        assert len(wav) > 1024
        resp = strict_client.post(
            "/segment", files={"audio": ("big.wav", wav, "audio/wav")}
        )
        assert resp.status_code == 413
        assert "exceeds" in resp.json()["detail"].lower()

    def test_small_payload_accepted(self, strict_client):
        wav = _make_wav(0.01)
        if len(wav) > 1024:
            pytest.skip("Even tiny WAV exceeds test limit")
        resp = strict_client.post(
            "/segment", files={"audio": ("tiny.wav", wav, "audio/wav")}
        )
        assert resp.status_code == 200


class TestAudioDurationLimit:
    def test_long_audio_rejected(self, fake_pyannote_pipeline):
        config = ServiceConfig(
            model_name="test",
            hf_token="",
            device="cpu",
            max_concurrency=2,
            max_audio_bytes=200 * 1024 * 1024,
            max_audio_duration_sec=0.5,
        )
        from segmentation_service.app import _build_app

        app = _build_app(config)
        with TestClient(app) as client:
            wav = _make_wav(1.0)
            resp = client.post(
                "/segment", files={"audio": ("long.wav", wav, "audio/wav")}
            )
            assert resp.status_code == 422
            assert "exceeds limit" in resp.json()["detail"].lower()


class TestInferenceTimeout:
    def test_timeout_returns_504(self, fake_pyannote_pipeline):
        _FakeVAD, _ = fake_pyannote_pipeline

        original_call = _FakeVAD.__call__

        def slow_call(self, audio_input):
            time.sleep(5)
            return original_call(self, audio_input)

        _FakeVAD.__call__ = slow_call

        config = ServiceConfig(
            model_name="test",
            hf_token="",
            device="cpu",
            max_concurrency=2,
            inference_timeout_sec=0.5,
        )
        from segmentation_service.app import _build_app

        app = _build_app(config)
        with TestClient(app) as client:
            wav = _make_wav(0.1)
            resp = client.post(
                "/segment", files={"audio": ("t.wav", wav, "audio/wav")}
            )
            assert resp.status_code == 504
            assert "timeout" in resp.json()["detail"].lower()

        _FakeVAD.__call__ = original_call


class TestSegmentAsync:
    @pytest.mark.asyncio
    async def test_segment_async_returns_result(
        self, sample_wav_bytes, fake_pyannote_pipeline
    ):
        seg = PyannoteSegmenter(model_name="test", device="cpu", hf_token="")
        seg.load()
        segments, duration = await seg.segment_async(sample_wav_bytes)
        assert duration > 0.9
        assert len(segments) >= 1
        seg.shutdown()

    @pytest.mark.asyncio
    async def test_segment_async_duration_limit(
        self, sample_wav_bytes, fake_pyannote_pipeline
    ):
        seg = PyannoteSegmenter(model_name="test", device="cpu", hf_token="")
        seg.load()
        with pytest.raises(ValueError, match="exceeds limit"):
            await seg.segment_async(
                sample_wav_bytes, max_duration_sec=0.1
            )
        seg.shutdown()


class TestConcurrentRequests:
    def test_multiple_concurrent_requests(self, normal_client):
        """Multiple requests should all succeed (sequentially via TestClient)."""
        wav = _make_wav(0.5)
        results = []
        for _ in range(6):
            resp = normal_client.post(
                "/segment", files={"audio": ("t.wav", wav, "audio/wav")}
            )
            results.append(resp.status_code)
        assert all(s == 200 for s in results)


class TestProbes:
    def test_healthz_always_200(self, normal_client):
        resp = normal_client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_readyz_200_after_startup(self, normal_client):
        resp = normal_client.get("/readyz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_health_backward_compat(self, normal_client):
        resp = normal_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, normal_client):
        normal_client.post(
            "/segment",
            files={"audio": ("t.wav", _make_wav(0.5), "audio/wav")},
        )
        resp = normal_client.get("/metrics")
        assert resp.status_code == 200
        body = resp.text
        assert "segmentation_requests_total" in body
        assert "segmentation_inference_duration_seconds" in body


class TestRequestID:
    def test_request_id_propagated(self, normal_client):
        wav = _make_wav(0.5)
        resp = normal_client.post(
            "/segment",
            files={"audio": ("t.wav", wav, "audio/wav")},
            headers={"x-request-id": "test-id-42"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-request-id") == "test-id-42"

    def test_request_id_generated(self, normal_client):
        wav = _make_wav(0.5)
        resp = normal_client.post(
            "/segment", files={"audio": ("t.wav", wav, "audio/wav")}
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-request-id")


class TestWarmup:
    def test_warmup_runs_without_error(self, fake_pyannote_pipeline):
        seg = PyannoteSegmenter(model_name="test", device="cpu", hf_token="")
        seg.load()
        seg.warmup()
        seg.shutdown()
