# Segmentation Service

GPU-accelerated speech segmentation service using **pyannote.audio**. Classifies audio regions as speech or non-speech and returns timestamped segments.

## Architecture

- **Model:** `pyannote/segmentation-3.0` loaded via `Model.from_pretrained` + `VoiceActivityDetection` pipeline (configurable via `SEG_MODEL_NAME` / `SEG_MODEL_PATH`)
- **Runtime:** FastAPI + uvicorn, GPU inference via PyTorch
- **Concurrency:** `asyncio.Semaphore` + `ThreadPoolExecutor` (pyannote is blocking)
- **Memory:** ~300MB VRAM per model instance

## Quick Start

```bash
cd services/segmentation_service

# Install dependencies
uv sync

# Set HuggingFace token (required for pyannote models)
export SEG_HF_TOKEN="hf_..."

# Or point to a local model snapshot
export SEG_MODEL_PATH="/path/to/models--pyannote--segmentation-3.0/snapshots/<hash>"

# Start the service
uvicorn segmentation_service.app:app --host 0.0.0.0 --port 8001
```

## Configuration

All settings are via environment variables (prefix `SEG_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SEG_MODEL_NAME` | `pyannote/segmentation-3.0` | pyannote model name (HF repo ID) |
| `SEG_MODEL_PATH` | `""` | Local path to model snapshot (overrides HF download) |
| `SEG_HF_TOKEN` | `""` | HuggingFace auth token |
| `SEG_DEVICE` | `cuda:0` | PyTorch device |
| `SEG_DTYPE` | `float32` | Inference precision: `float32`, `float16`, `bfloat16` |
| `SEG_MAX_CONCURRENCY` | `4` | Max queued/concurrent requests (semaphore) |
| `SEG_INFERENCE_TIMEOUT_SEC` | `300` | Per-request GPU inference timeout |
| `SEG_MAX_AUDIO_BYTES` | `209715200` (200 MB) | Reject uploads larger than this |
| `SEG_MAX_AUDIO_DURATION_SEC` | `3600` | Reject audio longer than this |
| `SEG_MIN_DURATION_ON` | `0.0` | VAD: remove speech regions shorter than this (seconds) |
| `SEG_MIN_DURATION_OFF` | `0.0` | VAD: fill non-speech gaps shorter than this (seconds) |
| `SEG_HOST` | `0.0.0.0` | Bind host |
| `SEG_PORT` | `8001` | Bind port |
| `SEG_LOG_LEVEL` | `info` | Log level |
| `SEG_LOG_DIR` | `logs` | Directory for rotated log files |
| `SEG_LOG_RETENTION_DAYS` | `30` | Number of daily log backups to keep |

## API Reference

### `POST /segment`

Segment audio into speech/non-speech regions.

**Request:** multipart form with `audio` file (WAV, MP3, FLAC, OGG).

**Response:**
```json
{
  "segments": [
    {"start": 0.0, "end": 0.15, "label": "non_speech"},
    {"start": 0.15, "end": 2.34, "label": "speech"},
    {"start": 2.34, "end": 3.0, "label": "non_speech"}
  ],
  "duration_sec": 3.0
}
```

**Errors:**
- `413` -- audio payload exceeds `SEG_MAX_AUDIO_BYTES`
- `422` -- invalid/empty audio, unsupported content type, or duration exceeds limit
- `504` -- inference timed out (`SEG_INFERENCE_TIMEOUT_SEC`)
- `500` -- inference failure

### `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model": "pyannote/segmentation-3.0",
  "device": "cuda:0",
  "gpu_memory_used_mb": 312.5,
  "gpu_memory_total_mb": 16384.0
}
```

## GPU Scaling

Run a **single uvicorn process** (no `--workers N`). Each uvicorn worker loads its own
model copy, wasting VRAM. Instead, tune `SEG_MAX_CONCURRENCY` to control how many
requests queue while the single-threaded GPU executor processes them sequentially.

| GPU | VRAM | Recommended `SEG_MAX_CONCURRENCY` |
|-----|------|----------------------------------|
| RTX 4080 Super | 16 GB | 8 |
| L40 | 48 GB | 24 |
| RTX 6000 Pro Blackwell | 96 GB | 48 |

## Docker

```bash
# Build (pass HF token for model pre-download)
docker compose build --build-arg SEG_HF_TOKEN=hf_...

# Run with GPU
docker compose up -d
```

See `docker-compose.yml` for GPU reservation and environment configuration.

## Testing

```bash
uv sync --group dev
uv run python -m pytest tests/ -v

# Integration tests against real audio (requires model)
uv run python -m pytest tests/test_integration_audio.py -v -s
```

## Integration with audio_asr_pipeline

Set `coarse_segmenter_backend="remote"` and `segmentation_service_url` in `PipelineConfig`:

```python
from audio_asr_pipeline import PipelineConfig

cfg = PipelineConfig(
    coarse_segmenter_backend="remote",
    segmentation_service_url="http://gpu-host:8001",
)
```

Or via eval CLI:
```bash
uv run python scripts/eval_test_audio.py \
  --coarse-backend remote \
  --segmentation-url http://gpu-host:8001 \
  --audio-dir test_audio \
  --base-url http://stt-host:8000
```
