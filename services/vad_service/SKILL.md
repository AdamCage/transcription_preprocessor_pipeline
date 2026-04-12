---
name: vad-service
description: >-
  Develops and debugs the GPU VAD service (FastAPI + Silero VAD).
  Refines coarse speech spans into precise boundaries on GPU.
  Use when working on vad_service/, Silero GPU inference, or the /refine endpoint.
---

# VAD Service

## Scope

FastAPI service under `services/vad_service/`. GPU-accelerated Voice Activity Detection via Silero VAD (JIT).

## Layout

- **`vad_service/app.py`** — FastAPI app, lifespan, `/refine` and `/health` endpoints.
- **`vad_service/inference.py`** — `SileroVADGPU` class: model load, sync `refine()`, async `refine_async()`, `_merge_nearby_spans` helper.
- **`vad_service/models.py`** — `TimeSpanIn`, `TimeSpanOut`, `RefineRequest`, `RefineResponse`, `HealthResponse`.
- **`vad_service/config.py`** — `ServiceConfig` (Pydantic Settings, env prefix `VAD_`).
- **`tests/`** — `conftest.py` (fake Silero model), `test_models.py`, `test_inference.py`, `test_endpoints.py`.

## Conventions

- Config via environment variables with `VAD_` prefix.
- Silero VAD loaded as JIT model (not ONNX) for GPU support: `torch.hub.load(..., onnx=False)`.
- Inference runs in `ThreadPoolExecutor`, guarded by `asyncio.Semaphore(max_concurrency)`.
- `/refine` accepts multipart form: `audio` file + `request` JSON string with spans and parameters.
- When VAD returns no speech, falls back to original input spans.

## Debugging

1. **torch.hub download fails** — check network access to github.com. Model caches in `~/.cache/torch/hub/`.
2. **CUDA errors** — verify `VAD_DEVICE` matches available GPU. Silero uses ~2MB VRAM.
3. **Empty refined spans** — check that input spans actually contain speech. Lower `threshold` if too aggressive.
