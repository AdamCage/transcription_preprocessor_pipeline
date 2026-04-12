"""Live endpoint integration test -- exercises all HTTP endpoints with a mock model."""

from __future__ import annotations

import io
import sys
import time
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf


class FakeSeg:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class FakeOutput:
    def itertracks(self):
        yield (FakeSeg(0.1, 0.5),)
        yield (FakeSeg(0.6, 0.9),)


class FakeModel:
    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

    def to(self, *a):
        return self


class FakeVADPipeline:
    def __init__(self, segmentation=None, **kw):
        pass

    def instantiate(self, params):
        pass

    def __call__(self, x):
        return FakeOutput()


# Patch pyannote before any import of our app
fake_audio = MagicMock()
fake_audio.Model = FakeModel
fake_pipelines = MagicMock()
fake_pipelines.VoiceActivityDetection = FakeVADPipeline
fake_audio.pipelines = fake_pipelines
fake_audio.pipelines.VoiceActivityDetection = FakeVADPipeline

sys.modules["pyannote"] = MagicMock()
sys.modules["pyannote.audio"] = fake_audio
sys.modules["pyannote.audio.pipelines"] = fake_pipelines
sys.modules["pyannote.audio.core"] = MagicMock()
sys.modules["pyannote.audio.core.pipeline"] = MagicMock()

from fastapi.testclient import TestClient

from segmentation_service.app import _build_app
from segmentation_service.config import ServiceConfig


def make_wav(dur: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
    sig = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def main() -> None:
    config = ServiceConfig(
        model_name="test",
        hf_token="",
        device="cpu",
        max_concurrency=4,
        inference_timeout_sec=10.0,
        max_audio_bytes=1024 * 1024,
        max_audio_duration_sec=60.0,
    )
    app = _build_app(config)
    ok = 0
    fail = 0

    with TestClient(app) as client:
        # 1. POST /segment (valid WAV)
        print("=== 1. POST /segment (valid WAV) ===")
        wav = make_wav(1.0)
        resp = client.post("/segment", files={"audio": ("test.wav", wav, "audio/wav")})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        seg_count = len(data["segments"])
        dur = data["duration_sec"]
        print(f"  Status: {resp.status_code}")
        print(f"  Segments: {seg_count}, duration: {dur}")
        for s in data["segments"]:
            print(f"    {s['label']}: {s['start']:.4f} - {s['end']:.4f}")
        rid = resp.headers.get("x-request-id")
        print(f"  X-Request-ID: {rid}")
        assert seg_count >= 1
        assert dur > 0.9
        assert rid
        ok += 1

        # 2. POST /segment with custom X-Request-ID
        print("\n=== 2. POST /segment with custom X-Request-ID ===")
        resp = client.post(
            "/segment",
            files={"audio": ("test.wav", wav, "audio/wav")},
            headers={"x-request-id": "my-trace-42"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-request-id") == "my-trace-42"
        print(f"  Status: {resp.status_code}, X-Request-ID: {resp.headers['x-request-id']}")
        ok += 1

        # 3. POST /segment (empty payload => 422)
        print("\n=== 3. POST /segment (empty payload => 422) ===")
        resp = client.post("/segment", files={"audio": ("test.wav", b"", "audio/wav")})
        assert resp.status_code == 422
        print(f"  Status: {resp.status_code}, detail: {resp.json()['detail']}")
        ok += 1

        # 4. POST /segment (bad content type => 422)
        print("\n=== 4. POST /segment (bad content type => 422) ===")
        resp = client.post(
            "/segment", files={"audio": ("test.txt", b"hello", "text/plain")}
        )
        assert resp.status_code == 422
        print(f"  Status: {resp.status_code}, detail: {resp.json()['detail']}")
        ok += 1

        # 5. POST /segment (oversized => 413)
        print("\n=== 5. POST /segment (oversized => 413) ===")
        big = b"x" * (1024 * 1024 + 1)
        resp = client.post("/segment", files={"audio": ("big.wav", big, "audio/wav")})
        assert resp.status_code == 413
        print(f"  Status: {resp.status_code}, detail: {resp.json()['detail']}")
        ok += 1

        # 6. GET /health
        print("\n=== 6. GET /health ===")
        resp = client.get("/health")
        assert resp.status_code == 200
        hdata = resp.json()
        print(f"  Status: {resp.status_code}")
        print(f"  Body: {hdata}")
        assert hdata["status"] == "ok"
        assert hdata["model"] == "test"
        ok += 1

        # 7. GET /healthz (liveness)
        print("\n=== 7. GET /healthz (liveness) ===")
        resp = client.get("/healthz")
        assert resp.status_code == 200
        print(f"  Status: {resp.status_code}, body: {resp.json()}")
        assert resp.json()["status"] == "alive"
        ok += 1

        # 8. GET /readyz (readiness)
        print("\n=== 8. GET /readyz (readiness) ===")
        resp = client.get("/readyz")
        assert resp.status_code == 200
        print(f"  Status: {resp.status_code}, body: {resp.json()}")
        assert resp.json()["status"] == "ready"
        ok += 1

        # 9. GET /metrics (Prometheus)
        print("\n=== 9. GET /metrics (Prometheus) ===")
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.text
        assert "segmentation_requests_total" in body
        assert "segmentation_inference_duration_seconds" in body
        assert "segmentation_audio_duration_seconds" in body
        print(f"  Status: {resp.status_code}")
        for line in body.splitlines():
            if "segmentation_" in line and not line.startswith("#"):
                print(f"  {line}")
        ok += 1

        # 10. Sequential throughput (5 requests)
        print("\n=== 10. Sequential throughput (5 requests) ===")
        t0 = time.perf_counter()
        for i in range(5):
            r = client.post(
                "/segment", files={"audio": (f"{i}.wav", make_wav(0.5), "audio/wav")}
            )
            assert r.status_code == 200
        elapsed = time.perf_counter() - t0
        print(f"  5 requests in {elapsed:.2f}s ({elapsed / 5:.3f}s avg)")
        ok += 1

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {ok} passed, {fail} failed")
    if fail:
        sys.exit(1)
    print("ALL ENDPOINT TESTS PASSED")


if __name__ == "__main__":
    main()
