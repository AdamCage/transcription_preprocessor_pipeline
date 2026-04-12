# VAD Service

GPU-accelerated Voice Activity Detection service using **Silero VAD**. Refines coarse speech segments into precise speech boundaries by running Silero VAD on GPU.

## Architecture

- **Model:** Silero VAD v5 (JIT, via `torch.hub` or local file)
- **Runtime:** FastAPI + uvicorn, GPU inference via PyTorch CUDA
- **Concurrency:** `asyncio.Semaphore` (request-level) + `ThreadPoolExecutor` with model replica pool (thread-level)
- **Parallel spans:** when `VAD_EXECUTOR_WORKERS > 1`, input spans within a single request are distributed across worker threads, each with its own model replica
- **Logging:** structlog (JSON) → stderr + `TimedRotatingFileHandler` (daily rotation, 7 backups)
- **Metrics:** Prometheus (`GET /metrics`) — request count, latency histogram, spans in/out, GPU memory
- **Memory:** ~10 MB VRAM per model replica

## Quick Start

```bash
cd services/vad_service

# Install dependencies (CUDA torch pulled from pytorch-cu124 index automatically)
uv sync

# Copy config template and adjust if needed
cp .env.example .env

# Start the service
uvicorn vad_service.app:app --host 0.0.0.0 --port 8002
```

## Configuration

All settings are via environment variables with prefix `VAD_` (or `.env` file, read by pydantic-settings):

| Variable | Default | Description |
|----------|---------|-------------|
| `VAD_DEVICE` | `cuda:0` | PyTorch device (`cuda:0`, `cpu`, etc.) |
| `VAD_EXECUTOR_WORKERS` | `1` | Number of model replicas / worker threads. >1 enables parallel span processing within a single request |
| `VAD_MAX_CONCURRENCY` | `8` | Max concurrent HTTP requests (asyncio Semaphore) |
| `VAD_HOST` | `0.0.0.0` | Bind host |
| `VAD_PORT` | `8002` | Bind port |
| `VAD_LOG_LEVEL` | `info` | Log level (`debug`, `info`, `warning`, `error`) |
| `VAD_LOG_DIR` | `logs` | Directory for JSON log files (daily rotation, 7 backups) |
| `VAD_MODEL_PATH` | _(empty)_ | Path to a local Silero JIT model file. If empty, downloads via `torch.hub` |
| `VAD_MAX_AUDIO_SIZE_MB` | `200.0` | Max audio payload size (MB). Exceeding → HTTP 413 |
| `VAD_INFERENCE_TIMEOUT_SEC` | `120.0` | Inference timeout (seconds). Exceeding → HTTP 504 |

## API Reference

### `POST /refine`

Refine coarse speech spans using Silero VAD.

**Request:** multipart form with:
- `audio` — WAV file (16 kHz mono recommended; 8 kHz also supported)
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
- `413` — audio payload too large (exceeds `VAD_MAX_AUDIO_SIZE_MB`)
- `422` — invalid audio, malformed JSON, or invalid parameters
- `500` — inference failure
- `504` — inference timeout (exceeds `VAD_INFERENCE_TIMEOUT_SEC`)

### `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model": "silero_vad",
  "device": "cuda:0",
  "gpu_memory_used_mb": 10.2,
  "gpu_memory_total_mb": 16384.0
}
```

### `GET /metrics`

Prometheus exposition format. Exposed counters/histograms:

| Metric | Type | Description |
|--------|------|-------------|
| `vad_requests_total` | Counter | Total `/refine` requests (label: `status`) |
| `vad_request_duration_seconds` | Histogram | End-to-end `/refine` latency |
| `vad_spans_input_total` | Counter | Cumulative input spans |
| `vad_spans_output_total` | Counter | Cumulative output spans |
| `vad_gpu_memory_bytes` | Gauge | GPU memory allocated |

## Logging

Structured JSON logs via **structlog**. Two sinks:

1. **stderr** — ConsoleRenderer (TTY) or JSON (non-TTY)
2. **File** — `<VAD_LOG_DIR>/vad_service.log`, daily rotation at midnight, 7 backups

Each request gets a unique `request_id` (from `X-Request-ID` header or auto-generated). Worker/span context (`worker_id`, `span_idx`, `inference_sec`) is included in debug-level log entries.

## Testing

### Unit tests (mocked model, no GPU)

```bash
uv sync --extra dev
uv run python -m pytest tests/ -v
```

### GPU integration benchmark

Runs all scenarios (sequential, multi-span, concurrent) against a live service with real audio from `test_audio_mad` and writes latency/throughput metrics to JSON:

```bash
# Start the service (in a separate terminal)
uvicorn vad_service.app:app --host 127.0.0.1 --port 8002

# Run benchmark (connects to running service)
uv run --extra dev python tests/test_gpu_integration.py \
  --url http://127.0.0.1:8002 --workers 1 --device cuda:0

# Compare multi-worker throughput
uv run --extra dev python tests/test_gpu_integration.py \
  --url http://127.0.0.1:8002 --workers 4 --device cuda:0
```

Results are written to `tests/results/<timestamp>_<config>_metrics.json`.

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

## Docker

```bash
docker build -t vad-service .
docker run --gpus all -p 8002:8002 vad-service
```

The Dockerfile pre-downloads the Silero model so the container works in air-gapped environments.
