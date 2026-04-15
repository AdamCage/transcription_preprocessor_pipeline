# Segmentation Service: Code Review & Production Refactoring Plan

> **Date:** 2026-04-12
> **Scope:** `services/segmentation_service/` (5 modules, ~350 LOC app + ~300 LOC tests)
> **Reference docs:** `README.md` (root), `docs/ARCHITECTURE.md`
> **Target hardware:** NVIDIA L40 (48 GB), RTX 6000 Pro Blackwell (96 GB)

---

## 1. Executive Summary

The segmentation service is a well-structured FastAPI draft with clean module separation, correct API contracts, and solid test coverage. It is ready for local development and functional testing. However, it lacks several critical elements for production GPU deployment: containerization, inference safeguards (timeouts, size limits), observability, and optimizations for high-VRAM cards like L40/RTX 6000.

This document catalogues all findings and proposes a three-phase refactoring plan.

---

## 2. Architecture Overview

```
segmentation_service/
  __init__.py          # package marker
  config.py            # ServiceConfig (pydantic-settings, SEG_* env prefix)
  models.py            # SegmentItem, SegmentResponse, HealthResponse
  inference.py         # PyannoteSegmenter (load / segment / segment_async / shutdown)
  app.py               # FastAPI app factory, /segment, /health endpoints
tests/
  conftest.py          # FakePyannoteOutput, sample WAV fixtures
  test_models.py       # Pydantic schema validation
  test_inference.py    # PyannoteSegmenter unit tests
  test_endpoints.py    # E2E FastAPI TestClient tests
pyproject.toml         # hatchling build, pyannote.audio>=3.3,<3.5, torch cu121 index (2.5.1)
```

Request flow:

```
Client (RemoteSegmentationClient)
  |
  |  POST /segment  multipart audio=<WAV bytes>
  v
app.py: segment_audio()
  |-- content-type check (frozenset of audio MIME types)
  |-- await audio.read()
  |-- asyncio.Semaphore acquire
  |-- await segmenter.segment_async(raw)
  |       |
  |       v
  |   inference.py: segment()  [in ThreadPoolExecutor]
  |       |-- soundfile.read -> float32 mono
  |       |-- torch tensor -> GPU
  |       |-- pyannote Pipeline.__call__
  |       |-- itertracks() -> SegmentItem[] with gap-filling
  |       |
  |       v
  |   (segments, duration_sec)
  |
  v
SegmentResponse JSON
```

---

## 3. Alignment with README.md and ARCHITECTURE.md

### 3.1 API Contract

| Aspect | Expected (ARCHITECTURE.md / remote_clients.py) | Actual (app.py / models.py) | Verdict |
|--------|------------------------------------------------|----------------------------|---------|
| Endpoint | `POST /segment` | `POST /segment` | OK |
| Input | multipart `audio` field | `UploadFile` param named `audio` | OK |
| Response body | `{segments: [{start, end, label}], duration_sec}` | `SegmentResponse` with same schema | OK |
| Labels | `"speech"`, `"non_speech"` | `Literal["speech", "non_speech"]` | OK |
| Health | `GET /health` with `status`, `model`, `device`, GPU mem | `HealthResponse` matches | OK |
| Default port | 8001 | `ServiceConfig.port = 8001` | OK |

### 3.2 Client Compatibility

`RemoteSegmentationClient` (`audio_asr_pipeline/remote_clients.py:46-119`) sends:
```python
files={"audio": ("audio.wav", wav_bytes, "audio/wav")}
```
and expects JSON with `segments[].{start, end, label}` and `duration_sec`. The service returns exactly this. The client's `_map_remote_label()` (line 32-43) maps `"speech"` -> `"speech"` and `"non_speech"` -> `"silence"`, which is the correct pipeline semantic.

### 3.3 Model Naming Ambiguity

ARCHITECTURE.md Section 8 describes the service as "Segmentation Service / FastAPI + pyannote". The default model `pyannote/voice-activity-detection` is technically a VAD pipeline, not a segmentation/diarization pipeline. This is **functionally correct** (the library only needs speech/non-speech labels for coarse segmentation), but the naming overlaps conceptually with `vad_service` (Silero). The distinction is:

| Service | Model | Role in pipeline |
|---------|-------|-----------------|
| `segmentation_service` | pyannote VAD | **Coarse** speech detection (first pass) |
| `vad_service` | Silero VAD | **Refinement** of speech boundaries (second pass) |

**Recommendation:** Add a clarifying note in the segmentation service README and ARCHITECTURE.md to avoid confusion.

---

## 4. Code Review: Issues by Severity

### 4.1 Critical (blocks production GPU deployment)

#### C1. No containerization

No `Dockerfile`, `docker-compose.yml`, or deployment manifests exist. The README documents `uvicorn ...` as the only launch method. For production GPU VMs (L40/RTX 6000), a container image with pinned CUDA runtime, pre-downloaded model weights, and health check is essential.

**Files affected:** new files required.

#### C2. No audio size or duration limit

`app.py` line 72 reads the entire upload into memory without any size check:
```python
raw = await audio.read()
```
A 2-hour WAV at 16 kHz mono = ~230 MB; at 48 kHz stereo = ~1.3 GB. Loading this as a torch tensor on GPU can trigger CUDA OOM, crashing the worker and all in-flight requests.

**Files affected:** `app.py`, `config.py`.

#### C3. No inference timeout

`segment_async` (inference.py:106-110) runs in a `ThreadPoolExecutor` without any timeout:
```python
async def segment_async(self, wav_bytes):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(self._executor, self.segment, wav_bytes)
```
If pyannote hangs (GPU stall, driver bug), the semaphore slot is held forever. With `max_concurrency=4`, four hung requests deadlock the entire service.

**Files affected:** `inference.py` or `app.py`.

#### C4. Thread safety of pyannote Pipeline

A single `self._pipeline` object is shared across all `ThreadPoolExecutor` threads (inference.py:34, 78). The pyannote `Pipeline.__call__` method operates on internal state and PyTorch tensors. PyTorch GPU operations from multiple threads on the same model can race.

With the current default `executor_workers=4`, up to 4 threads call `self._pipeline(audio_input)` concurrently. On L40 with recommended `SEG_EXECUTOR_WORKERS=8-16`, this becomes a significant risk.

**Files affected:** `inference.py`.

#### C5. Deprecated `use_auth_token` parameter

inference.py:56-58:
```python
pipeline = Pipeline.from_pretrained(
    self._model_name,
    use_auth_token=self._hf_token,
)
```
pyannote.audio >= 3.3.0 (which `pyproject.toml` requires) deprecated `use_auth_token` in favor of `token`. The current code emits a deprecation warning on every startup and may break in pyannote 4.x.

**Files affected:** `inference.py`.

---

### 4.2 Important (reliability and operability)

#### I1. No model warmup

The first inference request after startup pays additional latency from PyTorch JIT compilation and CUDA kernel caching (typically 2-5s extra). In production with health-check-based load balancing, this cold request may timeout on the client side (`remote_request_timeout_sec=120` is generous, but warmup + long audio could stack).

**Files affected:** `app.py` (lifespan), `inference.py`.

#### I2. No structured logging

All logging uses plain `logging.getLogger()`. In production (Kubernetes, systemd journal, ELK), JSON-structured logs with request IDs, latency, and audio metadata are necessary for troubleshooting.

**Files affected:** `app.py`, `inference.py`, new middleware.

#### I3. No observability / metrics

No Prometheus metrics, no OpenTelemetry spans. For a GPU service handling production audio, essential metrics include:
- `segmentation_inference_duration_seconds` (histogram)
- `segmentation_requests_total` (counter by status)
- `segmentation_audio_duration_seconds` (histogram of input lengths)
- `gpu_memory_used_bytes` (gauge)
- `segmentation_semaphore_waiters` (gauge)

**Files affected:** new `metrics.py` or middleware in `app.py`.

#### I4. No request ID propagation

Neither `app.py` nor `inference.py` generate or propagate a request ID. The client `RemoteSegmentationClient` doesn't send one either. Cross-service log correlation is impossible.

**Files affected:** `app.py` (middleware), `inference.py`.

#### I5. Scaling model: `uvicorn --workers N` vs. single-process

The README suggests:
> Scale with: `uvicorn segmentation_service.app:app --workers N`

Each uvicorn worker is a separate process that loads its own model copy. For pyannote VAD at ~300 MB VRAM, 32 workers = ~9.6 GB of duplicated model state. A single-process architecture with higher `SEG_MAX_CONCURRENCY` and `SEG_EXECUTOR_WORKERS` is more VRAM-efficient and avoids inter-process GPU contention.

On L40 (48 GB), single-process with `max_concurrency=24` can saturate the GPU without wasting VRAM on duplicate models.

**Files affected:** README.md (guidance update), deployment config.

---

### 4.3 Minor / Code Quality

#### M1. `application/octet-stream` in allowed content types

app.py:27 includes `"application/octet-stream"` in `_AUDIO_CONTENT_TYPES`. This is a generic fallback that any binary file will match, bypassing the content type check entirely. Some HTTP clients send this by default.

**Recommendation:** Remove from the frozenset; if needed, make it configurable.

#### M2. No readiness vs. liveness probe distinction

`GET /health` serves both purposes. In Kubernetes, a separate `GET /readyz` (returns 503 until model is loaded and warmed up) and `GET /healthz` (returns 200 if process is alive) pattern is standard.

#### M3. `HealthResponse.gpu_memory_total_mb` reads `total_mem`

app.py:99:
```python
gpu_total = round(torch.cuda.get_device_properties(idx).total_mem / 1024**2, 1)
```
The attribute is `total_mem` (not `total_memory`). This works, but it returns total physical VRAM, not available VRAM. Adding `torch.cuda.mem_get_info()` would give both free and total.

#### M4. Stereo-to-mono averaging in inference

inference.py:70-71:
```python
if data.ndim > 1:
    data = np.mean(data, axis=1)
```
The `axis=1` assumes shape `(n_samples, n_channels)` which is soundfile's default. This is correct for `sf.read()`, but a comment clarifying the assumption would be valuable since the main library (`io.py`) handles both `(n, 2)` and `(2, n)` layouts.

#### M5. Gap-filling threshold hardcoded

inference.py:87:
```python
if seg_start > prev_end + 0.01:
```
The 10ms gap threshold for inserting `non_speech` segments is hardcoded. This is reasonable but not configurable.

---

## 5. Test Coverage Assessment

| Test file | Coverage | Notes |
|-----------|----------|-------|
| `test_models.py` | Pydantic schemas: valid, invalid, defaults, edge cases | Good |
| `test_inference.py` | Load/segment/shutdown, error paths, empty audio fallback | Good |
| `test_endpoints.py` | E2E with TestClient: valid WAV, empty, bad content type, sequential | Good |

**Gaps:**
- No concurrent request stress test (multiple simultaneous `/segment` calls)
- No test for very large audio (OOM behavior)
- No test for inference timeout (hung pipeline mock)
- No async `segment_async` test (only sync `segment` is tested directly)
- `conftest.py:FakePyannoteOutput.itertracks()` yields 1-tuples `(seg,)` to match the unpacking `speech_turn[0].start` in inference.py:85. This is correct but fragile if pyannote changes its iteration format.

---

## 6. Refactoring Plan: Production GPU (L40 / RTX 6000)

### Phase 1: Critical Fixes (before any production deployment)

| # | Task | Files | Effort |
|---|------|-------|--------|
| 1 | **Dockerfile** -- multi-stage build with `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base, `uv` for dependency installation, model pre-download in build layer, `HEALTHCHECK` instruction | new `Dockerfile`, `.dockerignore` | M |
| 2 | **docker-compose.yml** -- GPU resource reservation (`deploy.resources.reservations.devices`), env_file, port mapping, restart policy, volume for model cache | new `docker-compose.yml` | S |
| 3 | **Inference timeout** -- wrap `segment_async` with `asyncio.wait_for(timeout)`, add `SEG_INFERENCE_TIMEOUT_SEC` config field (default: 300s) | `inference.py`, `app.py`, `config.py` | S |
| 4 | **Audio size limit** -- add `SEG_MAX_AUDIO_BYTES` (default 200 MB) and `SEG_MAX_AUDIO_DURATION_SEC` (default 3600s) to config; reject in `segment_audio()` before inference | `config.py`, `app.py` | S |
| 5 | **Fix `use_auth_token` -> `token`** -- update `Pipeline.from_pretrained()` call | `inference.py` | XS |
| 6 | **Thread safety** -- serialize GPU inference through a single-threaded executor (`max_workers=1`) + higher semaphore concurrency for I/O overlap, OR use `threading.Lock` around `self._pipeline(...)` | `inference.py` | S |

### Phase 2: Production Hardening

| # | Task | Files | Effort |
|---|------|-------|--------|
| 7 | **Model warmup** -- after `load()`, run inference on a 1s silent WAV in `lifespan` to JIT-compile GPU kernels | `app.py`, `inference.py` | S |
| 8 | **Structured JSON logging** -- add `structlog` or `python-json-logger`, configure in lifespan, inject `request_id` into log context | `app.py`, `pyproject.toml` | S |
| 9 | **Request ID middleware** -- generate UUID4 or propagate `X-Request-ID` header; pass through to inference logging | `app.py` (new middleware) | S |
| 10 | **Prometheus metrics** -- add `prometheus-fastapi-instrumentator` or manual counters/histograms: `inference_seconds`, `requests_total`, `audio_duration_seconds`, `gpu_memory_bytes`, `semaphore_queue_length` | new `metrics.py`, `app.py`, `pyproject.toml` | M |
| 11 | **Readiness / liveness probes** -- `GET /healthz` (always 200 if process alive, before model load), `GET /readyz` (200 only after load + warmup), keep `GET /health` for backward compat | `app.py`, `models.py` | S |
| 12 | **Gunicorn wrapper** -- `gunicorn -k uvicorn.workers.UvicornWorker` for proper SIGTERM handling and worker lifecycle; alternative: update README to single-worker guidance for GPU | new `gunicorn.conf.py` or README update | S |
| 13 | **Update README scaling guidance** -- replace "scale via `--workers N`" with single-process model, explain VRAM implications | `README.md` | S |

### Phase 3: Performance Optimization for L40 / RTX 6000

| # | Task | Files | Effort |
|---|------|-------|--------|
| 14 | **Half-precision inference** -- add `SEG_DTYPE` config (`float32` / `float16` / `bfloat16`), cast model and input tensors; test quality impact on pyannote VAD. bfloat16 is native on L40 (Ada Lovelace / sm_89) | `inference.py`, `config.py` | M |
| 15 | **Multi-GPU support** -- `SEG_DEVICES=cuda:0,cuda:1` config, round-robin or least-loaded dispatch across device-specific executors, each with its own model copy | `inference.py`, `config.py`, `app.py` | L |
| 16 | **Batch inference endpoint** -- `POST /segment/batch` accepting multiple files, process sequentially on GPU but with single tensor transfer overhead; useful for Airflow batch DAGs | `app.py`, `models.py`, `inference.py` | M |
| 17 | **Model caching in Docker** -- `RUN python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('...', token='...')"` in Dockerfile to bake model weights into image layer; set `HF_HOME` to known path | `Dockerfile` | S |
| 18 | **Adaptive concurrency** -- monitor `torch.cuda.mem_get_info()` and auto-reduce semaphore if free VRAM drops below threshold | `app.py` or new `gpu_monitor.py` | M |
| 19 | **Connection draining** -- on SIGTERM, stop accepting new requests, wait for in-flight to complete (configurable grace period), then shutdown. Gunicorn's `graceful_timeout` handles part of this | `app.py` (lifespan enhancement) | S |

---

## 7. Recommended GPU Configuration Profiles

### L40 (48 GB VRAM, Ada Lovelace, sm_89)

```env
SEG_DEVICE=cuda:0
SEG_MAX_CONCURRENCY=24
SEG_EXECUTOR_WORKERS=1
SEG_INFERENCE_TIMEOUT_SEC=300
SEG_MAX_AUDIO_BYTES=209715200
SEG_DTYPE=bfloat16
```

Single-threaded executor serializes GPU calls (thread safety guaranteed). Semaphore at 24 allows 24 requests to be queued/awaiting I/O while 1 runs inference. pyannote VAD at ~300 MB leaves 47+ GB for tensor allocations. bfloat16 is natively supported on Ada Lovelace.

### RTX 6000 Pro Blackwell (96 GB VRAM, Blackwell, sm_100)

```env
SEG_DEVICE=cuda:0
SEG_MAX_CONCURRENCY=48
SEG_EXECUTOR_WORKERS=1
SEG_INFERENCE_TIMEOUT_SEC=300
SEG_MAX_AUDIO_BYTES=209715200
SEG_DTYPE=bfloat16
```

Same single-threaded model, higher semaphore. If two GPUs are available:
```env
SEG_DEVICES=cuda:0,cuda:1
SEG_MAX_CONCURRENCY=48
```

### Comparison: Current Defaults vs. Recommended

| Parameter | Current default | L40 recommended | RTX 6000 recommended |
|-----------|----------------|-----------------|---------------------|
| `SEG_MAX_CONCURRENCY` | 4 | 24 | 48 |
| `SEG_EXECUTOR_WORKERS` | 4 | 1 | 1 |
| Inference timeout | none | 300s | 300s |
| Audio size limit | none | 200 MB | 200 MB |
| dtype | float32 | bfloat16 | bfloat16 |
| Process model | multi-worker | single-process | single-process |

---

## 8. Dockerfile Sketch

```dockerfile
# -- Stage 1: build dependencies --
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf_cache \
    UV_CACHE_DIR=/opt/uv_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# -- Stage 2: pre-download model --
COPY segmentation_service/ segmentation_service/
ARG SEG_HF_TOKEN=""
RUN uv run python -c "\
from pyannote.audio import Pipeline; \
Pipeline.from_pretrained('pyannote/voice-activity-detection', token='${SEG_HF_TOKEN}')"

# -- Stage 3: runtime --
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["uv", "run", "uvicorn", "segmentation_service.app:app", \
     "--host", "0.0.0.0", "--port", "8001"]
```

---

## 9. docker-compose.yml Sketch

```yaml
services:
  segmentation:
    build:
      context: .
      args:
        SEG_HF_TOKEN: ${SEG_HF_TOKEN}
    ports:
      - "8001:8001"
    env_file: .env
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 5s
      retries: 3
```

---

## 10. Priority Ordering

For immediate production deployment on L40/RTX 6000, the recommended implementation order is:

1. **C5** Fix `use_auth_token` (5 min, zero risk)
2. **C6** Thread safety -- single-threaded executor (30 min)
3. **C3** Inference timeout (30 min)
4. **C2** Audio size limit (30 min)
5. **I1** Model warmup (20 min)
6. **C1** Dockerfile + docker-compose (1-2h)
7. **I5** Update README scaling guidance (20 min)
8. **I2-I4** Structured logging + request ID (1-2h)
9. **I3** Prometheus metrics (1-2h)
10. Phase 3 optimizations (as needed based on profiling)
