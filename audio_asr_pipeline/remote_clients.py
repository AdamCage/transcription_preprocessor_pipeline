"""HTTP clients for remote GPU segmentation and VAD services."""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import httpx
import numpy as np
import soundfile as sf

from audio_asr_pipeline.errors import SegmentationError, VADProcessingError
from audio_asr_pipeline.models import LabeledSegment, TimeSpan

log = logging.getLogger(__name__)


def _samples_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono waveform as WAV PCM_16 in memory."""
    seg = np.asarray(samples, dtype=np.float32).ravel()
    peak = float(np.max(np.abs(seg))) if seg.size else 0.0
    if peak > 1.0:
        seg = seg / peak
    pcm = (np.clip(seg, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _map_remote_label(label: str) -> str:
    """Map service label to library SegmentLabel."""
    low = label.lower()
    if low == "speech":
        return "speech"
    if low in ("non_speech", "nonspeech", "non-speech"):
        return "silence"
    if "music" in low:
        return "music"
    if "noise" in low:
        return "noise"
    return "silence"


class RemoteSegmentationClient:
    """Async HTTP client for the segmentation GPU service."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_sec: float = 120.0,
        connect_timeout_sec: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(request_timeout_sec, connect=connect_timeout_sec)
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client

    async def segment(
        self,
        wav_bytes: bytes,
        *,
        duration_sec: float = 0.0,
    ) -> tuple[list[LabeledSegment], float]:
        """POST audio to /segment, return (labeled_segments, duration_sec)."""
        client = await self._ensure_client()
        try:
            resp = await client.post(
                "/segment",
                files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise SegmentationError(
                f"Remote segmentation HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SegmentationError(f"Remote segmentation request failed: {exc}") from exc

        data = resp.json()
        dur = float(data.get("duration_sec", duration_sec))
        segments: list[LabeledSegment] = []
        for seg in data.get("segments", []):
            segments.append(
                LabeledSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    label=_map_remote_label(seg.get("label", "speech")),
                )
            )

        if not segments:
            segments.append(LabeledSegment(start=0.0, end=dur, label="speech"))

        return segments, dur

    async def segment_from_samples(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> tuple[list[LabeledSegment], float]:
        wav_bytes = _samples_to_wav_bytes(samples, sample_rate)
        duration_sec = float(len(samples)) / sample_rate
        return await self.segment(wav_bytes, duration_sec=duration_sec)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class RemoteVADClient:
    """Async HTTP client for the VAD GPU service."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_sec: float = 120.0,
        connect_timeout_sec: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(request_timeout_sec, connect=connect_timeout_sec)
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client

    async def refine(
        self,
        wav_bytes: bytes,
        speech_spans: list[TimeSpan],
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 200,
        speech_pad_ms: int = 200,
        merge_gap_seconds: float = 0.5,
    ) -> list[TimeSpan]:
        """POST audio + spans to /refine, return refined speech spans."""
        client = await self._ensure_client()
        request_body: dict[str, Any] = {
            "spans": [{"start": s.start, "end": s.end} for s in speech_spans],
            "threshold": threshold,
            "min_speech_duration_ms": min_speech_duration_ms,
            "min_silence_duration_ms": min_silence_duration_ms,
            "speech_pad_ms": speech_pad_ms,
            "merge_gap_seconds": merge_gap_seconds,
        }
        try:
            resp = await client.post(
                "/refine",
                files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
                data={"request": json.dumps(request_body)},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VADProcessingError(
                f"Remote VAD HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise VADProcessingError(f"Remote VAD request failed: {exc}") from exc

        data = resp.json()
        spans: list[TimeSpan] = []
        for sp in data.get("spans", []):
            spans.append(TimeSpan(start=float(sp["start"]), end=float(sp["end"])))

        if not spans:
            return list(speech_spans)

        return spans

    async def refine_from_samples(
        self,
        samples: np.ndarray,
        sample_rate: int,
        speech_spans: list[TimeSpan],
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 200,
        speech_pad_ms: int = 200,
        merge_gap_seconds: float = 0.5,
    ) -> list[TimeSpan]:
        wav_bytes = _samples_to_wav_bytes(samples, sample_rate)
        return await self.refine(
            wav_bytes,
            speech_spans,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            merge_gap_seconds=merge_gap_seconds,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
