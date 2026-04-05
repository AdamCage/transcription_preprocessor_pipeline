"""Ensure multipart STT requests use an async-compatible stream (regression for httpx)."""

from __future__ import annotations

import asyncio

import httpx
from httpx._types import AsyncByteStream


def test_multipart_build_request_uses_async_byte_stream():
    """list[tuple] as `data` would force sync IteratorByteStream and break AsyncClient."""

    async def inner() -> None:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:1", trust_env=False) as client:
            req = client.build_request(
                "POST",
                "/v1/audio/transcriptions",
                data={
                    "model": "large-v3-turbo",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": ["segment", "word"],
                },
                files={"file": ("chunk.wav", b"RIFFfake", "audio/wav")},
            )
            assert isinstance(req.stream, AsyncByteStream), type(req.stream)

    asyncio.run(inner())
