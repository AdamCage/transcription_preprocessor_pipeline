---
name: VAD Service Production Hardening
overview: "Implement all fixes from vad_service_review.md: thread safety, model vendoring, request limits, timeouts, warm-up, sample rate validation, Dockerfile, Prometheus metrics (OTel-ready), graceful shutdown, dependency cleanup, and config parity with segmentation_service."
todos:
  - id: thread-safety
    content: "Refactor inference.py: model replica pool, reset_states, executor_workers=1 default"
    status: completed
  - id: model-vendoring
    content: Add VAD_MODEL_PATH config + torch.jit.load local path support in inference.py
    status: completed
  - id: request-limit
    content: Add max_audio_size_mb config + 413 check in app.py
    status: completed
  - id: inference-timeout
    content: Add inference_timeout_sec config + asyncio.wait_for in app.py
    status: completed
  - id: warmup
    content: Add dummy inference warm-up in lifespan after load()
    status: completed
  - id: sample-rate
    content: Add sample rate validation/warning in inference.py refine()
    status: completed
  - id: deps-cleanup
    content: Remove torchaudio and packaging from pyproject.toml, add prometheus-client
    status: completed
  - id: metrics
    content: Create metrics.py with Prometheus counters/histograms, add /metrics endpoint, instrument /refine
    status: completed
  - id: graceful-shutdown
    content: Change shutdown(wait=True), drain semaphore in lifespan cleanup
    status: completed
  - id: config-parity
    content: Add env_file to model_config, create .env.example
    status: completed
  - id: type-annotations
    content: Improve type hints in inference.py (ScriptModule, Callable)
    status: completed
  - id: dockerfile
    content: Create Dockerfile with pytorch base, pre-downloaded Silero model
    status: completed
  - id: tests-update
    content: Update conftest/test_inference/test_endpoints for pool pattern, size limit, timeout, metrics
    status: completed
isProject: false
---

# VAD Service — Full Production Hardening

Based on [vad_service_review.md](services/vad_service/vad_service_review.md), all 11 issues (4.1-4.11) and refactoring plan (7.1-7.9) will be implemented.

---

## 1. Thread Safety (CRITICAL 4.1 / 7.1)

**File:** [inference.py](services/vad_service/vad_service/inference.py)

Current default `executor_workers=4` causes data races on shared JIT model + `reset_states()`. Fix:

- Change default `executor_workers` to `1` in both `config.py` and `inference.py`
- Implement a **model replica pool**: create N copies of the Silero model (one per worker), stored in a `queue.Queue`. Each `refine()` call pops a model, uses it, then returns it. Silero is ~2MB VRAM so even 8 replicas are trivial on L40
- Call `model.reset_states()` before each `refine()` invocation for safety
- Keep semaphore at `max_concurrency=8` (async-level) but `executor_workers` defaults to `1`; users who set workers > 1 automatically get that many replicas

```python
import queue

class SileroVADGPU:
    def __init__(self, device="cuda:0", executor_workers=1):
        self._model_pool: queue.Queue[tuple] = queue.Queue()
        ...

    def load(self):
        for _ in range(executor_workers):
            model, utils = self._load_single_model(device)
            self._model_pool.put((model, utils[0]))

    def refine(self, ...):
        model, get_ts = self._model_pool.get()
        try:
            model.reset_states()
            # ... inference ...
        finally:
            self._model_pool.put((model, get_ts))
```

---

## 2. Model Vendoring (HIGH 4.2 / 7.2)

**Files:** [config.py](services/vad_service/vad_service/config.py), [inference.py](services/vad_service/vad_service/inference.py)

- Add `model_path: str = ""` to `ServiceConfig` (env: `VAD_MODEL_PATH`)
- If `model_path` is set and file exists, load via `torch.jit.load(model_path)` + manually define `get_speech_timestamps` from `utils_vad` or import from vendored copy
- Fallback to `torch.hub.load(...)` if `model_path` is empty (current behavior)
- Dockerfile will pre-download the model into the image at build time

---

## 3. Request Size Limit (HIGH 4.3)

**Files:** [config.py](services/vad_service/vad_service/config.py), [app.py](services/vad_service/vad_service/app.py)

- Add `max_audio_size_mb: float = 200.0` to `ServiceConfig`
- After `raw = await audio.read()`, check `len(raw) > config.max_audio_size_mb * 1024**2` and return 413

---

## 4. Inference Timeout (MEDIUM 4.4)

**File:** [app.py](services/vad_service/vad_service/app.py), [config.py](services/vad_service/vad_service/config.py)

- Add `inference_timeout_sec: float = 120.0` to config
- Wrap the `refine_async` call in `asyncio.wait_for(..., timeout=config.inference_timeout_sec)`
- Catch `asyncio.TimeoutError` and return HTTP 504

---

## 5. Warm-up (MEDIUM 4.5 / 7.5)

**File:** [app.py](services/vad_service/vad_service/app.py)

After `vad.load()` in lifespan, run a dummy inference on 1s of zeros at 16kHz:

```python
import numpy as np, io, soundfile as sf
dummy = np.zeros(16000, dtype=np.float32)
buf = io.BytesIO(); sf.write(buf, dummy, 16000, format="WAV")
vad.refine(buf.getvalue(), [TimeSpanIn(start=0.0, end=1.0)])
log.info("Warm-up complete")
```

---

## 6. Sample Rate Validation (MEDIUM 4.7)

**File:** [inference.py](services/vad_service/vad_service/inference.py)

After `sf.read()`, check `sr`. If not 16000:
- Log a warning
- Resample using `numpy` linear interpolation (avoid adding `torchaudio`/`librosa` dep) or simply log warning and proceed (Silero internally handles some rates but quality degrades)

Minimal approach: log warning + pass actual `sr` to `get_speech_timestamps` (which already receives `sampling_rate=sr`). Add explicit guard if sr not in {8000, 16000}:

```python
if sr not in (8000, 16000):
    log.warning("Unexpected sample rate %d Hz; Silero expects 16kHz. Results may degrade.", sr)
```

---

## 7. Dependency Cleanup (MEDIUM 4.6 / 7.8)

**File:** [pyproject.toml](services/vad_service/pyproject.toml)

- Remove `torchaudio>=2.1.0` (saves ~600MB in Docker image; not imported anywhere)
- Remove `packaging>=23.0` (not imported)
- Remove `torchaudio` from `[tool.uv.sources]`
- Add `prometheus-client>=0.21.0` for metrics

---

## 8. Prometheus Metrics (7.6 — OTel-ready)

**New file:** `services/vad_service/vad_service/metrics.py`
**Modified:** [app.py](services/vad_service/vad_service/app.py)

Using `prometheus_client` (OTel collector scrapes `/metrics` natively):

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT   = Counter("vad_requests_total", "Total /refine requests", ["status"])
REQUEST_LATENCY = Histogram("vad_request_duration_seconds", "Refine latency", buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120])
SPANS_IN        = Counter("vad_spans_input_total", "Input spans received")
SPANS_OUT       = Counter("vad_spans_output_total", "Output spans returned")
GPU_MEM         = Gauge("vad_gpu_memory_bytes", "GPU memory allocated")
```

- Add `GET /metrics` endpoint returning `generate_latest()`
- Instrument `/refine`: increment counters, observe latency histogram
- Periodically update `GPU_MEM` in `/health` or via background task

---

## 9. Graceful Shutdown (LOW 4.10 / 7.7)

**File:** [inference.py](services/vad_service/vad_service/inference.py)

- Change `self._executor.shutdown(wait=False)` to `self._executor.shutdown(wait=True, cancel_futures=True)` (Python 3.11+)
- In lifespan `yield` cleanup: acquire all semaphore permits (drains in-flight) before calling `vad.shutdown()`

---

## 10. Config Parity (LOW 4.11 / 7.9)

**File:** [config.py](services/vad_service/vad_service/config.py)

- Add `env_file=".env"` and `env_file_encoding="utf-8"` to `model_config` (match segmentation_service)
- Create `.env.example` with sample values

---

## 11. Type Annotations (LOW 4.8)

**File:** [inference.py](services/vad_service/vad_service/inference.py)

```python
from typing import Callable
self._model_pool: queue.Queue[tuple[torch.jit.ScriptModule, Callable]] = queue.Queue()
```

---

## 12. Dockerfile (7.3)

**New file:** `services/vad_service/Dockerfile`

```dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY vad_service/ vad_service/
# Pre-download Silero model into torch hub cache
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=False, trust_repo=True)"
EXPOSE 8002
CMD ["uvicorn", "vad_service.app:app", "--host", "0.0.0.0", "--port", "8002"]
```

---

## 13. Tests

Update existing tests for new behavior:

- **test_inference.py**: update `SileroVADGPU` constructor calls to match new defaults; test model pool acquisition/release; test `reset_states` is called
- **test_endpoints.py**: test 413 on oversized payload; test 504 on timeout; test `/metrics` endpoint
- **conftest.py**: update `fake_silero_model` to support pool-based loading (mock `torch.jit.load` path too)

---

## File Change Summary

| File | Action |
|------|--------|
| `vad_service/config.py` | Add `model_path`, `max_audio_size_mb`, `inference_timeout_sec`; change `executor_workers` default to 1; add `env_file` |
| `vad_service/inference.py` | Model replica pool, `reset_states`, local model load, sample rate warning, better types, graceful shutdown |
| `vad_service/app.py` | Size limit check, timeout wrapper, warm-up, metrics instrumentation, `/metrics` endpoint |
| `vad_service/metrics.py` | New — Prometheus counters/histograms/gauge |
| `vad_service/models.py` | No changes needed |
| `pyproject.toml` | Remove `torchaudio`, `packaging`; add `prometheus-client` |
| `.env.example` | New — sample env vars |
| `Dockerfile` | New — GPU production image |
| `tests/conftest.py` | Update fake model for pool pattern |
| `tests/test_inference.py` | Adapt to pool, add reset_states test |
| `tests/test_endpoints.py` | Add size limit, timeout, metrics tests |
