"""FastAPI application for the VAD service."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
from fastapi import FastAPI, Form, HTTPException, UploadFile

from vad_service.config import ServiceConfig, get_config
from vad_service.inference import SileroVADGPU
from vad_service.models import HealthResponse, RefineRequest, RefineResponse

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
        vad = SileroVADGPU(
            device=config.device,
            executor_workers=config.executor_workers,
        )
        vad.load()
        _app.state.vad = vad
        _app.state.sem = asyncio.Semaphore(config.max_concurrency)
        _app.state.config = config
        log.info(
            "VAD service ready | device=%s | max_concurrency=%d",
            vad.device,
            config.max_concurrency,
        )
        yield
        vad.shutdown()
        log.info("VAD service shut down")

    app = FastAPI(
        title="VAD Service",
        description="GPU-accelerated Voice Activity Detection via Silero VAD",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.post("/refine", response_model=RefineResponse)
    async def refine_spans(audio: UploadFile, request: str = Form(...)) -> RefineResponse:
        """Refine coarse speech spans using Silero VAD.

        Accepts multipart form: ``audio`` file + ``request`` JSON string.
        """
        ct = (audio.content_type or "").lower()
        if ct and ct not in _AUDIO_CONTENT_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported content type: {ct}. Send audio/wav or similar.",
            )
        raw = await audio.read()
        if not raw:
            raise HTTPException(status_code=422, detail="Empty audio payload")

        try:
            req = RefineRequest.model_validate(json.loads(request))
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid request JSON: {exc}",
            ) from exc

        if not req.spans:
            return RefineResponse(spans=[])

        sem: asyncio.Semaphore = app.state.sem
        try:
            async with sem:
                vad: SileroVADGPU = app.state.vad
                refined = await vad.refine_async(
                    raw,
                    req.spans,
                    threshold=req.threshold,
                    min_speech_duration_ms=req.min_speech_duration_ms,
                    min_silence_duration_ms=req.min_silence_duration_ms,
                    speech_pad_ms=req.speech_pad_ms,
                    merge_gap_seconds=req.merge_gap_seconds,
                )
        except RuntimeError as exc:
            log.exception("VAD inference failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("Unexpected VAD error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return RefineResponse(spans=refined)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        vad: SileroVADGPU = app.state.vad
        gpu_used: float | None = None
        gpu_total: float | None = None
        if torch.cuda.is_available():
            try:
                idx = int(vad.device.split(":")[-1]) if ":" in vad.device else 0
                gpu_used = round(torch.cuda.memory_allocated(idx) / 1024**2, 1)
                gpu_total = round(torch.cuda.get_device_properties(idx).total_mem / 1024**2, 1)
            except Exception:  # noqa: BLE001
                pass
        return HealthResponse(
            status="ok",
            model="silero_vad",
            device=vad.device,
            gpu_memory_used_mb=gpu_used,
            gpu_memory_total_mb=gpu_total,
        )

    return app


app = _build_app()
