# VAD Service

GPU-accelerated Voice Activity Detection service using **Silero VAD**. Refines coarse speech segments into precise speech boundaries by running Silero VAD on GPU.

## Architecture

- **Model:** Silero VAD v5 (JIT, via `torch.hub`)
- **Runtime:** FastAPI + uvicorn, GPU inference via PyTorch CUDA
- **Concurrency:** `asyncio.Semaphore` + `ThreadPoolExecutor`
- **Memory:** ~2MB VRAM per model instance

## Quick Start

```bash
cd services/vad_service

# Install dependencies
uv sync

# Start the service
uvicorn vad_service.app:app --host 0.0.0.0 --port 8002
```

## Configuration

All settings are via environment variables (prefix `VAD_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `VAD_DEVICE` | `cuda:0` | PyTorch device |
| `VAD_MAX_CONCURRENCY` | `8` | Max concurrent inference requests |
| `VAD_EXECUTOR_WORKERS` | `4` | Thread pool size for blocking inference |
| `VAD_HOST` | `0.0.0.0` | Bind host |
| `VAD_PORT` | `8002` | Bind port |
| `VAD_LOG_LEVEL` | `info` | Log level |

## API Reference

### `POST /refine`

Refine coarse speech spans using Silero VAD.

**Request:** multipart form with:
- `audio` — WAV file (16kHz mono recommended)
- `request` — JSON string with parameters:

```json
{
  "spans": [
    {"start": 0.0, "end": 5.0},
    {"start": 8.0, "end": 15.0}
  ],
  "threshold": 0.5,
  "min_speech_duration_ms": 250,
  "min_silence_duration_ms": 200,
  "speech_pad_ms": 200,
  "merge_gap_seconds": 0.5
}
```

**Response:**
```json
{
  "spans": [
    {"start": 0.12, "end": 4.85},
    {"start": 8.03, "end": 12.1},
    {"start": 13.5, "end": 14.9}
  ]
}
```

**Parameters:**
| Field | Default | Description |
|-------|---------|-------------|
| `threshold` | `0.5` | VAD confidence threshold (0.0–1.0) |
| `min_speech_duration_ms` | `250` | Minimum speech segment length |
| `min_silence_duration_ms` | `200` | Minimum silence to split segments |
| `speech_pad_ms` | `200` | Padding around detected speech |
| `merge_gap_seconds` | `0.5` | Merge spans closer than this gap |

**Errors:**
- `422` — invalid audio, malformed JSON, or invalid parameters
- `500` — inference failure

### `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model": "silero_vad",
  "device": "cuda:0",
  "gpu_memory_used_mb": 2.1,
  "gpu_memory_total_mb": 16384.0
}
```

## Testing

```bash
uv sync --group dev
uv run python -m pytest tests/ -v
```

## Integration with audio_asr_pipeline

Set `vad_backend="remote"` and `vad_service_url` in `PipelineConfig`:

```python
from audio_asr_pipeline import PipelineConfig

cfg = PipelineConfig(
    vad_backend="remote",
    vad_service_url="http://gpu-host:8002",
)
```

Or via eval CLI:
```bash
uv run python scripts/eval_test_audio.py \
  --vad-backend remote \
  --vad-url http://gpu-host:8002 \
  --audio-dir test_audio \
  --base-url http://stt-host:8000
```
