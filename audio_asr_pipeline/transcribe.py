"""OpenAI-compatible async STT clients (httpx raw and openai library)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Protocol

import httpx
from openai import AsyncOpenAI

from audio_asr_pipeline.config import VLLMTranscribeConfig
from audio_asr_pipeline.errors import TranscriptionRequestError
from audio_asr_pipeline.models import AudioChunk

logger = logging.getLogger(__name__)


class STTClient(Protocol):
    """Common interface for STT backends."""

    async def transcribe_chunk(
        self,
        chunk: AudioChunk,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


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


    async def aclose(self) -> None:
        pass


class OpenAITranscriptionClient:
    """STT client using the ``openai`` library's ``AsyncOpenAI`` with native
    retry, connection-pool, and auth-header handling."""

    def __init__(self, config: VLLMTranscribeConfig) -> None:
        self._cfg = config
        self._client: AsyncOpenAI | None = None

    def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            base = self._cfg.base_url.rstrip("/")
            if not base.endswith("/v1"):
                base = base + "/v1"
            self._client = AsyncOpenAI(
                api_key=self._cfg.api_key or "EMPTY",
                base_url=base,
                timeout=self._cfg.request_timeout_sec,
                max_retries=self._cfg.max_retries,
            )
        return self._client

    async def transcribe_chunk(
        self,
        chunk: AudioChunk,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not chunk.audio_bytes:
            raise TranscriptionRequestError(f"Chunk {chunk.chunk_id} has no audio_bytes")
        client = self._ensure_client()
        cfg = self._cfg
        logger.debug(
            "STT openai POST chunk=%s bytes=%d model=%s",
            chunk.chunk_id,
            len(chunk.audio_bytes),
            cfg.model,
        )
        granularities: list[str] = ["segment"]
        if cfg.include_word_timestamps:
            granularities.append("word")

        extra_body: dict[str, Any] = {}
        if cfg.prompt:
            extra_body["prompt"] = cfg.prompt

        try:
            kwargs_create: dict[str, Any] = {
                "model": cfg.model,
                "file": (f"{chunk.chunk_id}.wav", chunk.audio_bytes, "audio/wav"),
                "response_format": "verbose_json",
                "timestamp_granularities": granularities,
            }
            if cfg.language:
                kwargs_create["language"] = cfg.language
            if cfg.temperature is not None:
                kwargs_create["temperature"] = cfg.temperature
            if extra_body:
                kwargs_create["extra_body"] = extra_body
            resp = await client.audio.transcriptions.create(**kwargs_create)
        except Exception as exc:
            raise TranscriptionRequestError(
                f"STT openai failed for {chunk.chunk_id}: {exc}"
            ) from exc

        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        return dict(resp)  # type: ignore[arg-type]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def create_stt_client(config: VLLMTranscribeConfig) -> VLLMTranscriptionClient | OpenAITranscriptionClient:
    """Factory: choose STT client based on ``config.stt_backend``."""
    if config.stt_backend == "openai":
        return OpenAITranscriptionClient(config)
    return VLLMTranscriptionClient(config)


def write_chunk_to_temp(chunk: AudioChunk, directory: Path) -> Path:
    """Optional helper when keeping intermediate WAV chunks on disk."""
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{chunk.chunk_id}.wav"
    out.write_bytes(chunk.audio_bytes or b"")
    return out
