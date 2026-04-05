"""OpenAI-compatible async STT client."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

from audio_asr_pipeline.config import VLLMTranscribeConfig
from audio_asr_pipeline.errors import TranscriptionRequestError
from audio_asr_pipeline.models import AudioChunk

logger = logging.getLogger(__name__)


def _retry_after_seconds(response: httpx.Response, cap: float) -> float:
    raw = response.headers.get("Retry-After")
    if not raw:
        return min(1.0, cap)
    try:
        return min(float(raw.strip()), cap)
    except ValueError:
        return min(1.0, cap)


class VLLMTranscriptionClient:
    def __init__(self, config: VLLMTranscribeConfig) -> None:
        self._cfg = config

    async def transcribe_chunk(
        self,
        chunk: AudioChunk,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        if not chunk.audio_bytes:
            raise TranscriptionRequestError(f"Chunk {chunk.chunk_id} has no audio_bytes")
        base = self._cfg.base_url.rstrip("/")
        logger.debug(
            "STT POST %s/v1/audio/transcriptions chunk=%s bytes=%d",
            base,
            chunk.chunk_id,
            len(chunk.audio_bytes),
        )
        return await self._transcribe_bytes(
            chunk.audio_bytes,
            filename=f"{chunk.chunk_id}.wav",
            client=client,
        )

    async def _transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        filename: str,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        cfg = self._cfg
        last_err: Exception | None = None
        own_client = client is None
        if client is None:
            client = httpx.AsyncClient(
                base_url=cfg.base_url.rstrip("/"),
                timeout=httpx.Timeout(
                    cfg.request_timeout_sec,
                    connect=cfg.connect_timeout_sec,
                ),
                trust_env=cfg.trust_env,
            )

        try:
            for attempt in range(cfg.max_retries + 1):
                try:
                    # Must be a Mapping (dict), not list[tuple]. httpx treats non-Mapping
                    # `data` as raw body → SyncByteStream only → AsyncClient RuntimeError.
                    gran: list[str] = ["segment"]
                    if cfg.include_word_timestamps:
                        gran.append("word")
                    data_map: dict[str, str | list[str]] = {
                        "model": cfg.model,
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": gran,
                    }
                    if cfg.language:
                        data_map["language"] = cfg.language
                    if cfg.temperature is not None:
                        data_map["temperature"] = str(cfg.temperature)
                    if cfg.prompt:
                        data_map["prompt"] = cfg.prompt

                    files = {"file": (filename, audio_bytes, "audio/wav")}
                    resp = await client.post(
                        "/v1/audio/transcriptions",
                        data=data_map,
                        files=files,
                    )
                    if resp.status_code in (429, 503) and attempt < cfg.max_retries:
                        ra = _retry_after_seconds(resp, cfg.retry_after_cap_sec)
                        backoff = cfg.retry_backoff_sec * (2**attempt)
                        delay = max(backoff, ra)
                        logger.warning(
                            "STT %s | file=%s attempt=%d/%d retry_after=%.1fs",
                            resp.status_code,
                            filename,
                            attempt + 1,
                            cfg.max_retries,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()
                    return resp.json()
                except (httpx.HTTPError, ValueError) as e:
                    last_err = e
                    if attempt < cfg.max_retries:
                        logger.warning(
                            "STT retry | file=%s attempt=%d/%d err=%s",
                            filename,
                            attempt + 1,
                            cfg.max_retries,
                            e,
                        )
                        await asyncio.sleep(cfg.retry_backoff_sec * (2**attempt))
            assert last_err is not None
            raise TranscriptionRequestError(f"STT failed after retries: {last_err}") from last_err
        finally:
            if own_client:
                await client.aclose()


def write_chunk_to_temp(chunk: AudioChunk, directory: Path) -> Path:
    """Optional helper when keeping intermediate WAV chunks on disk."""
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{chunk.chunk_id}.wav"
    out.write_bytes(chunk.audio_bytes or b"")
    return out
