"""FastAPI application for the VAD service."""

from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import numpy as np
import soundfile as sf
import structlog
import torch
from fastapi import FastAPI, Form, HTTPException, Request, Response, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from vad_service.config import ServiceConfig, get_config
from vad_service.inference import SileroVADGPU
from vad_service.logging_config import configure_logging
from vad_service.metrics import (
    GPU_MEM_BYTES,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    SPANS_IN,
    SPANS_OUT,
)
from vad_service.models import (
    HealthResponse,
    RefineRequest,
    RefineResponse,
    TimeSpanIn,
)

log = structlog.stdlib.get_logger(__name__)

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
    configure_logging(config.log_level, config.log_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        vad = SileroVADGPU(
            device=config.device,
            executor_workers=config.executor_workers,
            model_path=config.model_path,
        )
        vad.load()

        # Warm-up: force JIT compilation and CUDA context init
        warmup_t0 = time.perf_counter()
        dummy = np.zeros(16000, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, dummy, 16000, format="WAV")
        vad.refine(buf.getvalue(), [TimeSpanIn(start=0.0, end=1.0)])
        log.info("Warm-up complete", warmup_sec=round(time.perf_counter() - warmup_t0, 2))

        _app.state.vad = vad
        _app.state.sem = asyncio.Semaphore(config.max_concurrency)
        _app.state.config = config
        log.info(
            "VAD service ready",
            device=vad.device,
            workers=config.executor_workers,
            max_concurrency=config.max_concurrency,
        )
        yield

        # Graceful shutdown: drain in-flight requests by acquiring all permits
        sem: asyncio.Semaphore = _app.state.sem
        for _ in range(config.max_concurrency):
            await sem.acquire()
        vad.shutdown()
        log.info("VAD service shut down")

    app = FastAPI(
        title="VAD Service",
        description="GPU-accelerated Voice Activity Detection via Silero VAD",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(_RequestIDMiddleware)

    @app.post("/refine", response_model=RefineResponse)
    async def refine_spans(audio: UploadFile, request: str = Form(...)) -> RefineResponse:
        """Refine coarse speech spans using Silero VAD.

        Accepts multipart form: ``audio`` file + ``request`` JSON string.
        """
        t0 = time.perf_counter()
        status_label = "ok"

        ct = (audio.content_type or "").lower()
        if ct and ct not in _AUDIO_CONTENT_TYPES:
            REQUEST_COUNT.labels(status="invalid_content_type").inc()
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported content type: {ct}. Send audio/wav or similar.",
            )
        raw = await audio.read()
        if not raw:
            REQUEST_COUNT.labels(status="empty_payload").inc()
            raise HTTPException(status_code=422, detail="Empty audio payload")

        max_bytes = int(config.max_audio_size_mb * 1024 * 1024)
        if len(raw) > max_bytes:
            REQUEST_COUNT.labels(status="payload_too_large").inc()
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio payload too large: {len(raw) / 1024**2:.1f} MB "
                    f"exceeds limit of {config.max_audio_size_mb} MB"
                ),
            )

        try:
            req = RefineRequest.model_validate(json.loads(request))
        except Exception as exc:
            REQUEST_COUNT.labels(status="invalid_json").inc()
            raise HTTPException(
                status_code=422,
                detail=f"Invalid request JSON: {exc}",
            ) from exc

        if not req.spans:
            REQUEST_COUNT.labels(status="ok").inc()
            return RefineResponse(spans=[])

        SPANS_IN.inc(len(req.spans))
        log.info(
            "Refine request received",
            audio_bytes=len(raw),
            spans_in=len(req.spans),
            threshold=req.threshold,
        )

        sem: asyncio.Semaphore = app.state.sem
        try:
            async with sem:
                vad: SileroVADGPU = app.state.vad
                refined = await asyncio.wait_for(
                    vad.refine_async(
                        raw,
                        req.spans,
                        threshold=req.threshold,
                        min_speech_duration_ms=req.min_speech_duration_ms,
                        min_silence_duration_ms=req.min_silence_duration_ms,
                        speech_pad_ms=req.speech_pad_ms,
                        merge_gap_seconds=req.merge_gap_seconds,
                    ),
                    timeout=config.inference_timeout_sec,
                )
        except asyncio.TimeoutError:
            status_label = "timeout"
            log.error(
                "VAD inference timed out",
                timeout_sec=config.inference_timeout_sec,
            )
            raise HTTPException(
                status_code=504,
                detail=f"Inference timed out after {config.inference_timeout_sec}s",
            )
        except RuntimeError as exc:
            status_label = "error"
            log.exception("VAD inference failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            status_label = "error"
            log.exception("Unexpected VAD error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            REQUEST_COUNT.labels(status=status_label).inc()
            REQUEST_LATENCY.observe(time.perf_counter() - t0)

        SPANS_OUT.inc(len(refined))
        latency = time.perf_counter() - t0
        log.info(
            "Refine request complete",
            spans_out=len(refined),
            latency_sec=round(latency, 3),
            status=status_label,
        )
        return RefineResponse(spans=refined)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        vad: SileroVADGPU = app.state.vad
        gpu_used: float | None = None
        gpu_total: float | None = None
        if torch.cuda.is_available():
            try:
                idx = int(vad.device.split(":")[-1]) if ":" in vad.device else 0
                mem_alloc = torch.cuda.memory_allocated(idx)
                GPU_MEM_BYTES.set(mem_alloc)
                gpu_used = round(mem_alloc / 1024**2, 1)
                gpu_total = round(
                    torch.cuda.get_device_properties(idx).total_mem / 1024**2, 1,
                )
            except Exception:  # noqa: BLE001
                pass
        return HealthResponse(
            status="ok",
            model="silero_vad",
            device=vad.device,
            gpu_memory_used_mb=gpu_used,
            gpu_memory_total_mb=gpu_total,
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


app = _build_app()
