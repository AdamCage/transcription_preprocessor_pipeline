"""Unit tests for remote segmentation and VAD clients."""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from audio_asr_pipeline.models import LabeledSegment, TimeSpan
from audio_asr_pipeline.remote_clients import (
    RemoteSegmentationClient,
    RemoteVADClient,
    _map_remote_label,
    _samples_to_wav_bytes,
)


def _make_samples(duration_sec: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


def _wav_from_samples(samples: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestSamplesToWavBytes:
    def test_produces_valid_wav(self):
        samples = _make_samples(0.5)
        wav = _samples_to_wav_bytes(samples, 16000)
        assert len(wav) > 44  # WAV header + data
        data, sr = sf.read(io.BytesIO(wav), dtype="float32")
        assert sr == 16000
        assert len(data) > 0

    def test_normalizes_loud_signal(self):
        loud = np.ones(1600, dtype=np.float32) * 5.0
        wav = _samples_to_wav_bytes(loud, 16000)
        data, _ = sf.read(io.BytesIO(wav), dtype="float32")
        assert np.max(np.abs(data)) <= 1.01


class TestMapRemoteLabel:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("speech", "speech"),
            ("non_speech", "silence"),
            ("nonspeech", "silence"),
            ("non-speech", "silence"),
            ("SPEECH", "speech"),
            ("music_segment", "music"),
            ("noise", "noise"),
            ("unknown", "silence"),
        ],
    )
    def test_mapping(self, raw, expected):
        assert _map_remote_label(raw) == expected


class TestRemoteSegmentationClient:
    @pytest.mark.asyncio
    async def test_segment_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "segments": [
                {"start": 0.0, "end": 0.5, "label": "speech"},
                {"start": 0.5, "end": 1.0, "label": "non_speech"},
            ],
            "duration_sec": 1.0,
        }

        client = RemoteSegmentationClient("http://fake:8001")
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        labeled, dur = await client.segment(b"fake_wav")
        assert dur == 1.0
        assert len(labeled) == 2
        assert labeled[0].label == "speech"
        assert labeled[1].label == "silence"

    @pytest.mark.asyncio
    async def test_segment_empty_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"segments": [], "duration_sec": 2.0}

        client = RemoteSegmentationClient("http://fake:8001")
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        labeled, dur = await client.segment(b"fake_wav", duration_sec=2.0)
        assert dur == 2.0
        assert len(labeled) == 1
        assert labeled[0].label == "speech"
        assert labeled[0].end == 2.0

    @pytest.mark.asyncio
    async def test_segment_from_samples(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "segments": [{"start": 0.0, "end": 0.5, "label": "speech"}],
            "duration_sec": 0.5,
        }

        client = RemoteSegmentationClient("http://fake:8001")
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        samples = _make_samples(0.5)
        labeled, dur = await client.segment_from_samples(samples, 16000)
        assert len(labeled) == 1

    @pytest.mark.asyncio
    async def test_aclose(self):
        client = RemoteSegmentationClient("http://fake:8001")
        mock_http = AsyncMock()
        client._client = mock_http
        await client.aclose()
        mock_http.aclose.assert_awaited_once()
        assert client._client is None


class TestRemoteVADClient:
    @pytest.mark.asyncio
    async def test_refine_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "spans": [{"start": 0.1, "end": 0.4}, {"start": 0.6, "end": 0.9}]
        }

        client = RemoteVADClient("http://fake:8002")
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        spans = await client.refine(
            b"fake_wav",
            [TimeSpan(0.0, 1.0)],
        )
        assert len(spans) == 2
        assert spans[0].start == 0.1
        assert spans[1].end == 0.9

    @pytest.mark.asyncio
    async def test_refine_empty_returns_original(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"spans": []}

        client = RemoteVADClient("http://fake:8002")
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        original = [TimeSpan(0.0, 1.0)]
        spans = await client.refine(b"fake_wav", original)
        assert len(spans) == 1
        assert spans[0].start == 0.0

    @pytest.mark.asyncio
    async def test_aclose(self):
        client = RemoteVADClient("http://fake:8002")
        mock_http = AsyncMock()
        client._client = mock_http
        await client.aclose()
        mock_http.aclose.assert_awaited_once()
