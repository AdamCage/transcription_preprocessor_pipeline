---
name: Segmentation service review
overview: Deep review of the segmentation_service draft, checking alignment with README.md and ARCHITECTURE.md, and planning a refactoring for production deployment on L40/RTX 6000 GPUs. Results written to segmentation_service_review.md.
todos:
  - id: write-review
    content: Write segmentation_service_review.md with all findings, alignment analysis, and refactoring plan
    status: completed
isProject: false
---

# Segmentation Service: Deep Review and Refactoring Plan

## Review Scope

Deep audit of `services/segmentation_service/` against the contracts defined in [README.md](README.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The service consists of 5 Python modules (~350 LOC), a pyproject.toml, and a test suite (~300 LOC). No Dockerfile or deployment manifests exist.

## Current State Summary

The service is a well-structured FastAPI application with clean separation (config, models, inference, app), Pydantic validation, async/sync bridge via `ThreadPoolExecutor`, semaphore concurrency control, and good test coverage. It correctly fulfills the `POST /segment` and `GET /health` API contract expected by `RemoteSegmentationClient` in the main library.

## Key Findings

### Positive (no changes needed)

- API contract (`POST /segment`, `GET /health`) fully matches `RemoteSegmentationClient` expectations: multipart `audio` field, JSON response with `segments[].{start, end, label}` and `duration_sec`
- Label mapping (`speech` / `non_speech`) aligns with `_map_remote_label()` in `remote_clients.py`
- Port 8001 and `SEG_*` env prefix match ARCHITECTURE.md
- Pydantic models with `ge=0.0` constraints and `Literal` labels
- Gap-filling logic in `inference.py` (non_speech between speech segments, trailing non_speech)
- CUDA fallback to CPU when GPU unavailable

### Issues Found (grouped by severity)

**Critical for production (L40/RTX 6000)**

1. **No containerization** -- no Dockerfile, no docker-compose. Cannot deploy reliably on GPU VMs
2. **No audio size / duration limit** -- arbitrarily large files can trigger CUDA OOM on a shared GPU
3. **No inference timeout** -- hung pyannote call holds semaphore slot forever, cascading to deadlock under load
4. **Thread safety unclear** -- pyannote Pipeline object shared across `ThreadPoolExecutor` threads; pyannote docs do not guarantee thread safety. On L40 (48 GB) with 16-32 recommended workers, this is a real risk
5. **`use_auth_token` deprecated** -- pyannote >= 3.3 uses `token=` parameter; `use_auth_token` triggers deprecation warning and may break in future versions

**Important for reliability**

6. **No model warmup** -- first request pays cold JIT penalty (~2-5s extra latency)
7. **No structured logging** -- plain `logging` module; no JSON format, no request IDs, hard to parse in production log aggregators
8. **No observability** -- no Prometheus metrics (latency, throughput, GPU util, queue depth)
9. **No request ID propagation** -- cannot correlate with client-side `RemoteSegmentationClient` logs
10. **`uvicorn --workers N` scaling model** -- each worker loads a separate model copy. For pyannote VAD at ~300 MB, 32 workers = ~9.6 GB; manageable on L40 but wasteful. Single-process + semaphore + more executor threads is more VRAM-efficient

**Minor / quality**

11. **No CORS configuration** -- not needed now but blocks future web-based health dashboards
12. **`application/octet-stream` accepted** -- too permissive; any binary content passes validation
13. **No readiness probe** -- `/health` always returns 200 even during model loading (before lifespan completes, this is fine with FastAPI, but no separate liveness vs readiness distinction)
14. **Empty `__init__.py`** in tests/ -- fine, but docstring would be consistent with package `__init__.py`

### Alignment with ARCHITECTURE.md

| Aspect | ARCHITECTURE.md expectation | Actual | Status |
|--------|----------------------------|--------|--------|
| Endpoint | `POST /segment` (multipart audio) | Matches | OK |
| Response | `{segments, duration_sec}` | Matches | OK |
| Labels | `speech` / `non_speech` | Matches | OK |
| Port | 8001 | Config default 8001 | OK |
| Model | pyannote.audio | `pyannote/voice-activity-detection` | OK (but see note) |
| Health | `GET /health` with GPU stats | Matches | OK |

**Note on model choice**: The architecture docs describe this as "coarse segmentation" while the default model `pyannote/voice-activity-detection` is a VAD pipeline, not a speaker segmentation or diarization pipeline. This is functionally correct (VAD output is exactly what the library needs for coarse speech detection), but the naming creates conceptual confusion with the separate `vad_service` which also does VAD (Silero). Consider documenting this distinction more clearly.

## Refactoring Plan for Production GPU (L40 / RTX 6000)

### Phase 1: Critical fixes (before any GPU deployment)

1. **Add Dockerfile** -- multi-stage build: NVIDIA CUDA 12.4 base, uv for deps, pre-download model in image layer, health check instruction
2. **Add docker-compose.yml** -- GPU resource reservation, env file, port mapping, restart policy
3. **Inference timeout** -- wrap `run_in_executor` call with `asyncio.wait_for(timeout)` in `app.py`
4. **Audio size limit** -- reject files > configurable max (e.g. `SEG_MAX_AUDIO_MB=100`, `SEG_MAX_DURATION_SEC=3600`)
5. **Fix `use_auth_token` -> `token`** -- update `inference.py` for pyannote >= 3.3
6. **Thread safety** -- switch from shared model + ThreadPoolExecutor to a model-per-thread pool pattern, or use a single inference thread with an async queue

### Phase 2: Production hardening

7. **Model warmup** -- run a dummy inference in `lifespan` after `load()` to JIT-compile all GPU kernels
8. **Structured JSON logging** -- `python-json-logger` or `structlog`, include request_id middleware
9. **Prometheus metrics** -- `prometheus-fastapi-instrumentator` or manual: inference_seconds histogram, requests_total counter, gpu_memory_bytes gauge, semaphore_queue_length gauge
10. **Request ID middleware** -- generate or propagate `X-Request-ID`, log in all inference messages
11. **Readiness vs liveness probes** -- `GET /healthz` (liveness, always 200 if process alive), `GET /readyz` (readiness, 200 only after model loaded and warmup complete)
12. **Gunicorn process manager** -- `gunicorn -k uvicorn.workers.UvicornWorker` for proper signal handling and worker lifecycle

### Phase 3: Performance optimization for L40/RTX 6000

13. **Batch inference endpoint** -- `POST /segment/batch` accepting multiple files, process as batch through pyannote for better GPU utilization
14. **Half-precision inference** -- test `pipeline.to(torch.float16)` or `torch.bfloat16` on L40; pyannote supports it with minor quality tradeoff
15. **Multi-GPU support** -- `SEG_DEVICES=cuda:0,cuda:1` with round-robin or least-loaded dispatch
16. **Adaptive concurrency** -- auto-tune `max_concurrency` based on GPU memory headroom
17. **Model caching in Docker layer** -- pre-download HF model during `docker build`, set `HF_HOME` to cached path; eliminates startup download on new containers
18. **Connection draining on shutdown** -- complete in-flight requests before exiting (SIGTERM handler)

### Estimated GPU resource profiles

| GPU | VRAM | Model footprint | Recommended config |
|-----|------|----------------|-------------------|
| L40 | 48 GB | ~300 MB (VAD) | `SEG_MAX_CONCURRENCY=24`, `SEG_EXECUTOR_WORKERS=8`, single process |
| RTX 6000 Pro | 96 GB | ~300 MB (VAD) | `SEG_MAX_CONCURRENCY=48`, `SEG_EXECUTOR_WORKERS=16`, single process |

Single-process model avoids duplicate VRAM allocation; more executor threads handle CPU I/O while GPU saturates.

## Deliverable

Write `services/segmentation_service/segmentation_service_review.md` with the full review results.
