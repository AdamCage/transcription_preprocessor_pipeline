# Segmentation Service

GPU-accelerated speech segmentation service using **pyannote.audio**. Classifies audio regions as speech or non-speech and returns timestamped segments.

## Architecture

- **Model:** `pyannote/voice-activity-detection` (configurable via `SEG_MODEL_NAME`)
- **Runtime:** FastAPI + uvicorn, GPU inference via PyTorch
- **Concurrency:** `asyncio.Semaphore` + `ThreadPoolExecutor` (pyannote is blocking)
- **Memory:** ~300MB VRAM per model instance; scale via uvicorn `--workers`

## Quick Start

```bash
cd services/segmentation_service

# Install dependencies
uv sync

# Set HuggingFace token (required for pyannote models)
export SEG_HF_TOKEN="hf_..."

# Start the service
uvicorn segmentation_service.app:app --host 0.0.0.0 --port 8001
```

## Configuration

All settings are via environment variables (prefix `SEG_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SEG_MODEL_NAME` | `pyannote/voice-activity-detection` | pyannote pipeline name |
| `SEG_HF_TOKEN` | `""` | HuggingFace auth token |
| `SEG_DEVICE` | `cuda:0` | PyTorch device |
| `SEG_MAX_CONCURRENCY` | `4` | Max concurrent inference requests |
| `SEG_EXECUTOR_WORKERS` | `4` | Thread pool size for blocking inference |
| `SEG_HOST` | `0.0.0.0` | Bind host |
| `SEG_PORT` | `8001` | Bind port |
| `SEG_LOG_LEVEL` | `info` | Log level |

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
- `422` — invalid/empty audio or unsupported content type
- `500` — inference failure

### `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model": "pyannote/voice-activity-detection",
  "device": "cuda:0",
  "gpu_memory_used_mb": 312.5,
  "gpu_memory_total_mb": 16384.0
}
```

## GPU Scaling

| GPU | VRAM | Recommended workers |
|-----|------|-------------------|
| RTX 4080 Super | 16 GB | 4–8 |
| L40 | 48 GB | 16–32 |
| RTX 6000 Pro Blackwell | 96 GB | 32–64 |

Scale with: `uvicorn segmentation_service.app:app --workers N`

## Testing

```bash
uv sync --group dev
uv run python -m pytest tests/ -v
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
