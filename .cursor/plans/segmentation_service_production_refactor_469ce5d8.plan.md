---
name: Segmentation Service Production Refactor
overview: "Implement the three-phase production refactoring plan from segmentation_service_review.md: critical fixes (thread safety, timeouts, size limits, deprecated API, containerization), production hardening (warmup, logging, metrics, probes), and GPU performance optimization (half-precision, multi-GPU, batch endpoint)."
todos:
  - id: fix-use-auth-token
    content: "C5: Change `use_auth_token` to `token` in inference.py Pipeline.from_pretrained()"
    status: completed
  - id: thread-safety
    content: "C4: Single-threaded executor (max_workers=1), remove SEG_EXECUTOR_WORKERS config"
    status: completed
  - id: inference-timeout
    content: "C3: Add SEG_INFERENCE_TIMEOUT_SEC config + asyncio.wait_for() wrapper"
    status: completed
  - id: audio-size-limit
    content: "C2: Add SEG_MAX_AUDIO_BYTES + SEG_MAX_AUDIO_DURATION_SEC config, reject oversized uploads"
    status: completed
  - id: model-warmup
    content: "I1: Add warmup() method, call in lifespan after load()"
    status: completed
  - id: remove-octet-stream
    content: "M1: Remove application/octet-stream from _AUDIO_CONTENT_TYPES"
    status: completed
  - id: containerization
    content: "C1: Create Dockerfile, docker-compose.yml, .dockerignore"
    status: completed
  - id: readme-scaling
    content: "I5: Update README to recommend single-process model instead of --workers N"
    status: completed
  - id: structured-logging
    content: "I2+I4: Add structlog, JSON logging, request ID middleware"
    status: completed
  - id: prometheus-metrics
    content: "I3: Add prometheus-fastapi-instrumentator, custom GPU/inference metrics"
    status: completed
  - id: readiness-probes
    content: "M2: Add GET /healthz + GET /readyz endpoints"
    status: completed
  - id: tests-gaps
    content: "Add tests: concurrent requests, timeout, size limit, segment_async"
    status: completed
  - id: phase3-half-precision
    content: "Phase 3: Add SEG_DTYPE config for float16/bfloat16 inference"
    status: completed
isProject: false
---

# Segmentation Service: Production Refactoring Plan

Based on the review in [segmentation_service_review.md](services/segmentation_service/segmentation_service_review.md), validated against the actual source code. All cited issues are confirmed.

## Phase 1: Critical Fixes

### 1.1 Fix deprecated `use_auth_token` (C5) -- XS

In [inference.py](services/segmentation_service/segmentation_service/inference.py), line 56-58:

```python
pipeline = Pipeline.from_pretrained(
    self._model_name,
    use_auth_token=self._hf_token,  # deprecated in pyannote >= 3.3
)
```

Change to `token=self._hf_token`. Zero-risk, immediate fix.

### 1.2 Thread safety -- single-threaded executor (C4) -- S

The `ThreadPoolExecutor(max_workers=executor_workers)` with default `executor_workers=4` allows 4 concurrent calls to `self._pipeline(audio_input)` on a shared pipeline object. PyTorch GPU ops from multiple threads on the same model can race.

**Fix:** In [inference.py](services/segmentation_service/segmentation_service/inference.py):
- Hard-code executor to `max_workers=1` (serialize GPU calls)
- Remove `executor_workers` from constructor; remove `SEG_EXECUTOR_WORKERS` from [config.py](services/segmentation_service/segmentation_service/config.py)
- The `asyncio.Semaphore(max_concurrency)` in [app.py](services/segmentation_service/segmentation_service/app.py) already gates I/O concurrency -- requests queue at the semaphore while the single executor thread runs inference sequentially

### 1.3 Inference timeout (C3) -- S

In [inference.py](services/segmentation_service/segmentation_service/inference.py), `segment_async` has no timeout:

```python
async def segment_async(self, wav_bytes):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(self._executor, self.segment, wav_bytes)
```

**Fix:**
- Add `SEG_INFERENCE_TIMEOUT_SEC: float = 300.0` to `ServiceConfig` in [config.py](services/segmentation_service/segmentation_service/config.py)
- Wrap with `asyncio.wait_for()` in either `segment_async` or the endpoint in [app.py](services/segmentation_service/segmentation_service/app.py):

```python
async with sem:
    try:
        segments, duration = await asyncio.wait_for(
            segmenter.segment_async(raw),
            timeout=config.inference_timeout_sec,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Inference timeout")
```

### 1.4 Audio size limit (C2) -- S

[app.py](services/segmentation_service/segmentation_service/app.py) line 72 reads entire upload without size check.

**Fix:**
- Add to [config.py](services/segmentation_service/segmentation_service/config.py):
  - `max_audio_bytes: int = 200 * 1024 * 1024` (200 MB)
  - `max_audio_duration_sec: float = 3600.0`
- In the `/segment` endpoint, after `raw = await audio.read()`:
  - Check `len(raw) > config.max_audio_bytes` -> 413
  - After `sf.read()` in inference, check duration against limit -> 422
  - Or check duration at the endpoint level after inference returns (simpler, but burns GPU time on invalid input)
- Better approach: check raw size pre-inference in the endpoint, check duration post-decode but pre-GPU-inference inside `segment()` in [inference.py](services/segmentation_service/segmentation_service/inference.py)

### 1.5 Containerization (C1) -- M

New files under `services/segmentation_service/`:

- **Dockerfile**: Multi-stage build based on `nvidia/cuda:12.4.1-runtime-ubuntu22.04`, `uv` for deps, model pre-download in build layer, `HEALTHCHECK`. Follow the sketch in review Section 8.
- **docker-compose.yml**: GPU reservation via `deploy.resources.reservations.devices`, env_file, health check, restart policy. Follow Section 9.
- **.dockerignore**: Exclude tests, `.git`, `__pycache__`, `.env`, review docs, etc.

---

## Phase 2: Production Hardening

### 2.1 Model warmup (I1) -- S

In [app.py](services/segmentation_service/segmentation_service/app.py) lifespan, after `segmenter.load()`:
- Add a `warmup()` method to `PyannoteSegmenter` that runs inference on a 1-second silent WAV
- Call it before `yield` in lifespan
- This JIT-compiles CUDA kernels and warms caches

### 2.2 Structured logging + request ID (I2, I4) -- S

- Add `structlog` or `python-json-logger` to [pyproject.toml](services/segmentation_service/pyproject.toml)
- Configure JSON logging in lifespan
- Add middleware in [app.py](services/segmentation_service/segmentation_service/app.py) that:
  - Reads `X-Request-ID` header or generates `uuid4`
  - Binds it to structlog context
  - Passes through to inference logging

### 2.3 Prometheus metrics (I3) -- M

- Add `prometheus-fastapi-instrumentator` to deps
- Create `metrics.py` with custom metrics:
  - `segmentation_inference_duration_seconds` (histogram)
  - `segmentation_requests_total` (counter by status)
  - `segmentation_audio_duration_seconds` (histogram)
  - `gpu_memory_used_bytes` (gauge)
  - `segmentation_semaphore_waiters` (gauge)
- Wire into [app.py](services/segmentation_service/segmentation_service/app.py)

### 2.4 Readiness / liveness probes (M2) -- S

- Add `GET /healthz` (200 if process alive, even before model load)
- Add `GET /readyz` (200 only after model loaded + warmed)
- Keep `GET /health` for backward compat with `RemoteSegmentationClient`
- Track ready state via `app.state.ready: bool`

### 2.5 Update README scaling guidance (I5) -- S

Replace "scale with `--workers N`" in [README.md](services/segmentation_service/README.md) with single-process model guidance. Explain that each uvicorn worker duplicates the model in VRAM. Recommend `SEG_MAX_CONCURRENCY` tuning instead.

### 2.6 Remove `application/octet-stream` from allowed types (M1) -- XS

Remove from the `_AUDIO_CONTENT_TYPES` frozenset in [app.py](services/segmentation_service/segmentation_service/app.py). Optionally make it configurable.

---

## Phase 3: GPU Performance Optimization

### 3.1 Half-precision inference (P14) -- M

- Add `SEG_DTYPE` config field (`float32` / `float16` / `bfloat16`, default `float32`)
- In `inference.py`, cast model and input tensors to selected dtype
- bfloat16 is native on L40 (Ada Lovelace sm_89) and RTX 6000 Blackwell (sm_100)

### 3.2 Multi-GPU support (P15) -- L

- Add `SEG_DEVICES` config (comma-separated, e.g. `cuda:0,cuda:1`)
- Create per-device executor + pipeline instances
- Round-robin or least-loaded dispatch
- Largest scope change -- defer until single-GPU is proven in production

### 3.3 Batch inference endpoint (P16) -- M

- `POST /segment/batch` accepting multiple files
- Sequential GPU processing but single overhead
- Lower priority -- useful for Airflow DAGs with many short files

---

## Test Gaps to Fill

- Concurrent request stress test (multiple simultaneous `/segment` calls)
- Inference timeout test (mock hung pipeline)
- Audio size limit rejection test
- `segment_async` test (currently only sync `segment` tested)

---

## Implementation Order (recommended)

Priority follows review Section 10:

1. C5: `use_auth_token` -> `token` (5 min)
2. C4: Thread safety -- single executor (30 min)
3. C3: Inference timeout (30 min)
4. C2: Audio size limit (30 min)
5. I1: Model warmup (20 min)
6. M1: Remove `application/octet-stream` (5 min)
7. C1: Dockerfile + docker-compose (1-2h)
8. I5: README scaling guidance (20 min)
9. I2+I4: Structured logging + request ID (1-2h)
10. I3: Prometheus metrics (1-2h)
11. M2: Readiness/liveness probes (30 min)
12. Phase 3 items as needed based on profiling
