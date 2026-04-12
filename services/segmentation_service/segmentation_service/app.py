"""FastAPI application for the segmentation service."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
import torch
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from prometheus_client import generate_latest
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import PlainTextResponse

from segmentation_service.config import ServiceConfig, get_config
from segmentation_service.inference import PyannoteSegmenter
from segmentation_service.logging_config import configure_logging
from segmentation_service.metrics import (
    AUDIO_DURATION,
    GPU_MEMORY_USED,
    INFERENCE_DURATION,
    REQUESTS_TOTAL,
    SEMAPHORE_WAITERS,
)
from segmentation_service.models import HealthResponse, SegmentResponse

log = structlog.stdlib.get_logger(__name__)

_AUDIO_CONTENT_TYPES = frozenset(
    {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/mpeg",
        "audio/ogg",
        "audio/flac",
    }
)


class _RequestIDMiddleware(BaseHTTPMiddleware):
    """Propagate or generate X-Request-ID and bind to structlog context."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response


def _build_app(config: ServiceConfig | None = None) -> FastAPI:
    config = config or get_config()
    configure_logging(config.log_level, config.log_dir, config.log_retention_days)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        _app.state.ready = False
        segmenter = PyannoteSegmenter(
            model_name=config.model_name,
            device=config.device,
            hf_token=config.hf_token,
            dtype=config.dtype,
            model_path=config.model_path,
            min_duration_on=config.min_duration_on,
            min_duration_off=config.min_duration_off,
        )
        segmenter.load()
        segmenter.warmup()
        _app.state.segmenter = segmenter
        _app.state.sem = asyncio.Semaphore(config.max_concurrency)
        _app.state.config = config
        _app.state.ready = True
        log.info(
            "Segmentation service ready",
            model=config.model_name,
            device=segmenter.device,
            max_concurrency=config.max_concurrency,
        )
        yield
        _app.state.ready = False
        segmenter.shutdown()
        log.info("Segmentation service shut down")

    app = FastAPI(
        title="Segmentation Service",
        description="GPU-accelerated speech segmentation via pyannote.audio",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(_RequestIDMiddleware)

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

        cfg: ServiceConfig = app.state.config
        if len(raw) > cfg.max_audio_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio payload {len(raw)} bytes exceeds "
                    f"limit {cfg.max_audio_bytes} bytes"
                ),
            )

        sem: asyncio.Semaphore = app.state.sem
        SEMAPHORE_WAITERS.set(cfg.max_concurrency - sem._value)
        t0 = time.perf_counter()
        try:
            async with sem:
                segmenter: PyannoteSegmenter = app.state.segmenter
                segments, duration = await asyncio.wait_for(
                    segmenter.segment_async(
                        raw, max_duration_sec=cfg.max_audio_duration_sec
                    ),
                    timeout=cfg.inference_timeout_sec,
                )
        except asyncio.TimeoutError:
            REQUESTS_TOTAL.labels(status="timeout").inc()
            log.error(
                "Inference timed out",
                timeout_sec=cfg.inference_timeout_sec,
            )
            raise HTTPException(
                status_code=504, detail="Inference timeout"
            )
        except ValueError as exc:
            REQUESTS_TOTAL.labels(status="rejected").inc()
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            REQUESTS_TOTAL.labels(status="error").inc()
            log.exception("Segmentation inference failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            REQUESTS_TOTAL.labels(status="error").inc()
            log.exception("Unexpected segmentation error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        INFERENCE_DURATION.observe(time.perf_counter() - t0)
        AUDIO_DURATION.observe(duration)
        REQUESTS_TOTAL.labels(status="ok").inc()
        if torch.cuda.is_available():
            GPU_MEMORY_USED.set(torch.cuda.memory_allocated())

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

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness probe -- returns 200 if the process is alive."""
        return {"status": "alive"}

    @app.get("/readyz")
    async def readyz() -> Response:
        """Readiness probe -- returns 200 only after model is loaded and warmed."""
        if not getattr(app.state, "ready", False):
            return Response(
                content='{"status":"not_ready"}',
                status_code=503,
                media_type="application/json",
            )
        return Response(
            content='{"status":"ready"}',
            status_code=200,
            media_type="application/json",
        )

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        """Prometheus metrics endpoint."""
        return PlainTextResponse(
            generate_latest(), media_type="text/plain; version=0.0.4"
        )

    return app


app = _build_app()
