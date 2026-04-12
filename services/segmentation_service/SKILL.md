---
name: segmentation-service
description: >-
  Develops and debugs the GPU segmentation service (FastAPI + pyannote.audio).
  Classifies audio into speech/non-speech regions on GPU.
  Use when working on segmentation_service/, pyannote inference, or the /segment endpoint.
---

# Segmentation Service

## Scope

FastAPI service under `services/segmentation_service/`. GPU-accelerated speech segmentation via pyannote.audio.

## Layout

- **`segmentation_service/app.py`** — FastAPI app, lifespan, `/segment` and `/health` endpoints.
- **`segmentation_service/inference.py`** — `PyannoteSegmenter` class: model load, sync `segment()`, async `segment_async()` via executor.
- **`segmentation_service/models.py`** — `SegmentItem`, `SegmentResponse`, `HealthResponse` Pydantic schemas.
- **`segmentation_service/config.py`** — `ServiceConfig` (Pydantic Settings, env prefix `SEG_`).
- **`tests/`** — `conftest.py` (fake pyannote pipeline), `test_models.py`, `test_inference.py`, `test_endpoints.py`.

## Conventions

- Config via environment variables with `SEG_` prefix.
- Inference runs in `ThreadPoolExecutor` (pyannote is blocking), guarded by `asyncio.Semaphore(max_concurrency)`.
- Model loaded once at startup via lifespan; `shutdown()` frees GPU memory.
- Audio accepted as multipart `audio` file; WAV/MP3/FLAC/OGG.
- Response maps pyannote speech regions to `"speech"` labels and gaps to `"non_speech"`.

## Debugging

1. **Model download fails** — check `SEG_HF_TOKEN` and network access to huggingface.co.
2. **CUDA OOM** — reduce `SEG_MAX_CONCURRENCY` or switch to a smaller model.
3. **Slow inference** — ensure `SEG_DEVICE=cuda:0` (not cpu). Check GPU utilization with `nvidia-smi`.
