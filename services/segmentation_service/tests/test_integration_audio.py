"""Integration tests against real audio files from test_audio_mad/.

Writes performance metrics to tests/results/metrics_<timestamp>.json.
Run with:  uv run python -m pytest tests/test_integration_audio.py -v -s
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from segmentation_service.app import _build_app
from segmentation_service.config import ServiceConfig

AUDIO_DIR = Path(__file__).resolve().parents[3] / "test_audio_mad"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

_collected_results: list[dict] = []
_session_config: dict = {}


def _wav_files() -> list[Path]:
    if not AUDIO_DIR.is_dir():
        return []
    return sorted(AUDIO_DIR.glob("*.wav"))


def _make_ids(paths: list[Path]) -> list[str]:
    return [p.stem for p in paths]


@pytest.fixture(scope="module")
def live_client():
    """Build the app with real model from .env (or mocked if unavailable)."""
    config = ServiceConfig()
    app = _build_app(config)
    _session_config.update({
        "model": config.model_name,
        "model_path": config.model_path,
        "device": config.device,
        "dtype": config.dtype,
    })
    with TestClient(app) as c:
        yield c


_files = _wav_files()


@pytest.mark.skipif(not _files, reason="No .wav files in test_audio_mad/")
@pytest.mark.parametrize("wav_path", _files, ids=_make_ids(_files))
def test_segment_audio_file(wav_path: Path, live_client: TestClient) -> None:
    raw = wav_path.read_bytes()
    file_size = len(raw)

    t0 = time.perf_counter()
    resp = live_client.post(
        "/segment",
        files={"audio": (wav_path.name, raw, "audio/wav")},
    )
    wall_sec = time.perf_counter() - t0

    assert resp.status_code == 200, f"{wav_path.name}: HTTP {resp.status_code} - {resp.text}"
    data = resp.json()

    duration_sec = data["duration_sec"]
    segments = data["segments"]
    rtf = wall_sec / duration_sec if duration_sec > 0 else 0.0

    speech_segs = [s for s in segments if s["label"] == "speech"]
    non_speech_segs = [s for s in segments if s["label"] == "non_speech"]

    _collected_results.append({
        "file": wav_path.name,
        "size_bytes": file_size,
        "duration_sec": round(duration_sec, 4),
        "num_segments": len(segments),
        "num_speech": len(speech_segs),
        "num_non_speech": len(non_speech_segs),
        "inference_wall_sec": round(wall_sec, 4),
        "rtf": round(rtf, 6),
        "segments": segments,
    })

    assert len(segments) >= 1
    assert duration_sec > 0


@pytest.fixture(scope="module", autouse=True)
def write_metrics_json():
    """Write collected metrics to JSON after all tests in this module complete."""
    yield
    if not _collected_results:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"metrics_{ts}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **_session_config,
        "results": _collected_results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Metrics written to {out_path}")
