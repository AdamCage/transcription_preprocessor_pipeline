"""OpenAI-compatible async STT clients (httpx raw and openai library) and Gemma chat ASR."""

from __future__ import annotations

import asyncio
import base64
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


_GEMMA_ASR_PROMPT_TEMPLATE = (
    "Transcribe the following speech segment in {lang} into {lang} text.\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, "
    "i.e. write 1.7 and not one point seven, and write 3 instead of three."
)

_GEMMA_ASR_PROMPT_AUTO = (
    "Transcribe the following speech segment in its original language.\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, "
    "i.e. write 1.7 and not one point seven, and write 3 instead of three."
)

_LANG_NAMES: dict[str, str] = {
    "ru": "Russian",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "tr": "Turkish",
    "pl": "Polish",
    "uk": "Ukrainian",
    "nl": "Dutch",
}


def _gemma_asr_prompt(config: VLLMTranscribeConfig) -> str:
    if config.gemma_asr_prompt:
        return config.gemma_asr_prompt
    if config.language:
        lang = _LANG_NAMES.get(config.language, config.language)
        return _GEMMA_ASR_PROMPT_TEMPLATE.format(lang=lang)
    return _GEMMA_ASR_PROMPT_AUTO


class GemmaTranscriptionClient:
    """Chat-based ASR client for Gemma-4-E4B-it via Ollama or OpenAI-compatible chat API."""

    def __init__(self, config: VLLMTranscribeConfig) -> None:
        self._cfg = config
        self._http: httpx.AsyncClient | None = None
        self._openai: AsyncOpenAI | None = None

    def _ensure_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._cfg.base_url.rstrip("/"),
                timeout=httpx.Timeout(
                    self._cfg.request_timeout_sec,
                    connect=self._cfg.connect_timeout_sec,
                ),
                trust_env=self._cfg.trust_env,
            )
        return self._http

    def _ensure_openai(self) -> AsyncOpenAI:
        if self._openai is None:
            base = self._cfg.base_url.rstrip("/")
            if not base.endswith("/v1"):
                base = base + "/v1"
            self._openai = AsyncOpenAI(
                api_key=self._cfg.api_key or "EMPTY",
                base_url=base,
                timeout=self._cfg.request_timeout_sec,
                max_retries=self._cfg.max_retries,
            )
        return self._openai

    async def transcribe_chunk(
        self,
        chunk: AudioChunk,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not chunk.audio_bytes:
            raise TranscriptionRequestError(f"Chunk {chunk.chunk_id} has no audio_bytes")
        cfg = self._cfg
        audio_b64 = base64.b64encode(chunk.audio_bytes).decode("ascii")
        prompt = _gemma_asr_prompt(cfg)
        chunk_duration = max(0.0, chunk.end - chunk.start)

        logger.debug(
            "Gemma ASR | api_style=%s chunk=%s bytes=%d model=%s",
            cfg.gemma_api_style,
            chunk.chunk_id,
            len(chunk.audio_bytes),
            cfg.model,
        )

        text = await self._call_with_retries(audio_b64, prompt)

        return {
            "text": text.strip(),
            "language": cfg.language,
            "segments": [
                {"start": 0.0, "end": chunk_duration, "text": text.strip()},
            ],
            "words": [],
        }

    async def _call_with_retries(self, audio_b64: str, prompt: str) -> str:
        cfg = self._cfg
        last_err: Exception | None = None
        for attempt in range(cfg.max_retries + 1):
            try:
                if cfg.gemma_api_style == "ollama_native":
                    return await self._ollama_native(audio_b64, prompt)
                else:
                    return await self._openai_chat(audio_b64, prompt)
            except TranscriptionRequestError:
                raise
            except Exception as e:
                last_err = e
                if attempt < cfg.max_retries:
                    delay = cfg.retry_backoff_sec * (2 ** attempt)
                    logger.warning(
                        "Gemma retry | attempt=%d/%d err=%s delay=%.1fs",
                        attempt + 1,
                        cfg.max_retries,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
        assert last_err is not None
        raise TranscriptionRequestError(
            f"Gemma ASR failed after {cfg.max_retries + 1} attempts: {last_err}"
        ) from last_err

    async def _ollama_native(self, audio_b64: str, prompt: str) -> str:
        """POST /api/chat with audio in ``images`` field (Ollama convention)."""
        client = self._ensure_http()
        body: dict[str, Any] = {
            "model": self._cfg.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [audio_b64],
                },
            ],
        }
        if self._cfg.gemma_max_tokens:
            body["options"] = {"num_predict": self._cfg.gemma_max_tokens}
        if self._cfg.gemma_thinking:
            body["think"] = True

        resp = await client.post("/api/chat", json=body)
        if resp.status_code in (429, 503):
            ra = _retry_after_seconds(resp, self._cfg.retry_after_cap_sec)
            raise httpx.HTTPStatusError(
                f"Ollama {resp.status_code}",
                request=resp.request,
                response=resp,
            )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    async def _openai_chat(self, audio_b64: str, prompt: str) -> str:
        """POST /v1/chat/completions with audio content blocks (vLLM / OpenAI-compatible)."""
        client = self._ensure_openai()
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        extra: dict[str, Any] = {}
        if self._cfg.gemma_max_tokens:
            extra["max_tokens"] = self._cfg.gemma_max_tokens

        resp = await client.chat.completions.create(
            model=self._cfg.model,
            messages=messages,  # type: ignore[arg-type]
            **extra,
        )
        return resp.choices[0].message.content or ""

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        if self._openai is not None:
            await self._openai.close()
            self._openai = None


def create_stt_client(
    config: VLLMTranscribeConfig,
) -> VLLMTranscriptionClient | OpenAITranscriptionClient | GemmaTranscriptionClient:
    """Factory: choose STT client based on ``config.stt_backend``."""
    if config.stt_backend == "gemma":
        return GemmaTranscriptionClient(config)
    if config.stt_backend == "openai":
        return OpenAITranscriptionClient(config)
    return VLLMTranscriptionClient(config)


def write_chunk_to_temp(chunk: AudioChunk, directory: Path) -> Path:
    """Optional helper when keeping intermediate WAV chunks on disk."""
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{chunk.chunk_id}.wav"
    out.write_bytes(chunk.audio_bytes or b"")
    return out
