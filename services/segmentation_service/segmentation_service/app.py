"""FastAPI application for the segmentation service."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
from fastapi import FastAPI, HTTPException, UploadFile

from segmentation_service.config import ServiceConfig, get_config
from segmentation_service.inference import PyannoteSegmenter
from segmentation_service.models import HealthResponse, SegmentResponse

log = logging.getLogger(__name__)

_AUDIO_CONTENT_TYPES = frozenset(
    {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/mpeg",
        "audio/ogg",
        "audio/flac",
        "application/octet-stream",
    }
)


def _build_app(config: ServiceConfig | None = None) -> FastAPI:
    config = config or get_config()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        segmenter = PyannoteSegmenter(
            model_name=config.model_name,
            device=config.device,
            hf_token=config.hf_token,
            executor_workers=config.executor_workers,
        )
        segmenter.load()
        _app.state.segmenter = segmenter
        _app.state.sem = asyncio.Semaphore(config.max_concurrency)
        _app.state.config = config
        log.info(
            "Segmentation service ready | model=%s | device=%s | max_concurrency=%d",
            config.model_name,
            segmenter.device,
            config.max_concurrency,
        )
        yield
        segmenter.shutdown()
        log.info("Segmentation service shut down")

    app = FastAPI(
        title="Segmentation Service",
        description="GPU-accelerated speech segmentation via pyannote.audio",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.post("/segment", response_model=SegmentResponse)
    async def segment_audio(audio: UploadFile) -> SegmentResponse:
        ct = (audio.content_type or "").lower()
        if ct and ct not in _AUDIO_CONTENT_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported content type: {ct}. Send audio/wav or similar.",
            )
        raw = await audio.read()
        if not raw:
            raise HTTPException(status_code=422, detail="Empty audio payload")

        sem: asyncio.Semaphore = app.state.sem
        try:
            async with sem:
                segmenter: PyannoteSegmenter = app.state.segmenter
                segments, duration = await segmenter.segment_async(raw)
        except RuntimeError as exc:
            log.exception("Segmentation inference failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("Unexpected segmentation error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return SegmentResponse(segments=segments, duration_sec=round(duration, 4))

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        segmenter: PyannoteSegmenter = app.state.segmenter
        gpu_used: float | None = None
        gpu_total: float | None = None
        if torch.cuda.is_available():
            try:
                idx = int(segmenter.device.split(":")[-1]) if ":" in segmenter.device else 0
                gpu_used = round(torch.cuda.memory_allocated(idx) / 1024**2, 1)
                gpu_total = round(torch.cuda.get_device_properties(idx).total_mem / 1024**2, 1)
            except Exception:  # noqa: BLE001
                pass
        return HealthResponse(
            status="ok",
            model=segmenter.model_name,
            device=segmenter.device,
            gpu_memory_used_mb=gpu_used,
            gpu_memory_total_mb=gpu_total,
        )

    return app


app = _build_app()
