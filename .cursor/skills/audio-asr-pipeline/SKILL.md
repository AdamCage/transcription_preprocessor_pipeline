---
name: audio-asr-pipeline
description: >-
  Develops and debugs the transcription_preprocessor_pipeline repo: audio_asr_pipeline
  (coarse seg local or remote GPU, Silero VAD local/GPU, chunking, OpenAI STT via
  openai/httpx, Gemma chat ASR via Ollama/vLLM, merge), GPU microservices
  (services/segmentation_service pyannote, services/vad_service Silero GPU),
  and scripts/eval_test_audio.py (XLSX, stereo-call, tqdm).
  Use when the user works on this workspace, ASR preprocessing, ina, pyannote, VAD,
  remote GPU services, Gemma-4, eval reports, or Windows TF/uv extras.
---

# Audio ASR preprocessor pipeline

## Scope

Python package **`audio_asr_pipeline`** + CLI **`scripts/eval_test_audio.py`** + GPU services under **`services/`**. Do not refactor unrelated files.

## Layout

- **`audio_asr_pipeline/config.py`** — `PipelineConfig`, `VLLMTranscribeConfig`, `CoarseBackendName` (`whole_file`/`ina`/`remote`), `VADBackendName` (`silero`/`none`/`remote`), `STTBackendName` (`httpx`/`openai`/`gemma`), service URLs.
- **`audio_asr_pipeline/pipeline.py`** — `AudioTranscriptionPipeline`: shared **`_file_sem`**; `_prepare_file_cpu` (local) or `_prepare_file_remote` (GPU services); then async chunk STT.
- **`audio_asr_pipeline/remote_clients.py`** — `RemoteSegmentationClient` (POST /segment), `RemoteVADClient` (POST /refine) — async httpx clients for GPU services.
- **`audio_asr_pipeline/transcribe.py`** — `OpenAITranscriptionClient` (openai `AsyncOpenAI`, recommended for vLLM Whisper), `VLLMTranscriptionClient` (httpx raw), `GemmaTranscriptionClient` (Gemma-4 chat ASR via Ollama native or OpenAI-compatible chat API), `create_stt_client` factory, `STTClient` Protocol.
- **`audio_asr_pipeline/preprocess.py`** — load → coarse → VAD (local backends) → speech `TimeSpan[]`.
- **`audio_asr_pipeline/segmenters.py`** — `ina` (map `female`/`male` → `speech`); `whole_file`.
- **`audio_asr_pipeline/vad.py`** — Silero via **ONNX** (local CPU); no unconditional `.to()` on wrapper.
- **`audio_asr_pipeline/io.py`** — **`split_stereo_channels`** for librosa `(n,2)` or `(2,n)`; `write_mono_wav`.
- **`services/segmentation_service/`** — FastAPI GPU service: pyannote.audio speech segmentation. Config via `SEG_*` env vars.
- **`services/vad_service/`** — FastAPI GPU service: Silero VAD JIT on CUDA. Config via `VAD_*` env vars.
- **`scripts/eval_test_audio.py`** — `--stereo-call`, `--coarse-backend` (`ina`/`whole_file`/`remote`), `--vad-backend` (`silero`/`remote`/`none`), `--segmentation-url`, `--vad-url`, `--stt-backend`, `--api-key`, `--concurrency`, `--no-vad`, tqdm progress.

## Conventions

- Match existing style; minimal diffs; no drive-by renames.
- **`uv sync`** installs ALL dependency-groups (dev, eval, ina) via `[tool.uv] default-groups`. Optional deps for pip: `pip install .[eval,ina]`. Windows: `[tool.uv] override-dependencies` pins plain tensorflow.
- STT default backend: **`openai`** (`AsyncOpenAI`); fallback: **`httpx`** (raw multipart); Gemma chat ASR: **`gemma`** (`GemmaTranscriptionClient`, Ollama native or OpenAI chat). `trust_env=False` by default (avoid proxy hijacking localhost).
- Gemma audio limit: **30 seconds** per clip (pipeline chunks at 28s). Audio sent as base64 WAV in chat message. Returns synthetic verbose_json (chunk-level segments, no word timestamps).

## Stereo eval

- Channel **0** → `call_from`, **1** → `call_to`; refs `{stem}_call_from.txt`, `{stem}_call_to.txt`.
- Always use **`split_stereo_channels`** after `librosa.load(..., mono=False)`; never assume `y[:,0]` is time without checking shape.
- WER/CER: when hypothesis is empty but reference exists, reports WER=1.0 / CER=1.0.

## Debugging checklist

1. **No speech spans after coarse** — check ina labels (gender → speech mapping), `drop_*` flags. For remote: check segmentation service health endpoint.
2. **ina ImportError** — `uv sync` should install it; if excluded: `uv sync --no-group ina` was used.
3. **Stereo near-zero duration / ina negative dimensions** — wrong channel axis; verify `split_stereo_channels`.
4. **Parallelism** — `max_concurrent_files` is instance-level semaphore; `max_in_flight_requests` is global STT cap. Stereo uses two `process_file` calls in parallel (two slots). Check DEBUG logs for `acquired_file_sem` / `stt_send` to verify concurrency.
5. **STT auth errors (401/403)** — set `api_key` in `VLLMTranscribeConfig` or `--api-key` in eval CLI.
6. **Gemma Ollama 500 errors** — intermittent GGML crash ([ollama#15333](https://github.com/ollama/ollama/issues/15333)); retry logic handles it. Ensure 16kHz mono input. Lower `num_ctx` if persistent.
7. **Skip VAD for fast debugging** — `--no-vad` or `--vad-backend none` sets `vad_backend="none"`; audio chunked by duration only.
8. **Remote service unreachable** — check `segmentation_service_url` / `vad_service_url` and that services are running. GET `/health` on each service to verify.
9. **Remote segmentation returns no speech** — pyannote may not detect speech in very short/noisy audio. Check service logs. Fallback: `--coarse-backend whole_file`.

## Reference

- Architecture and Mermaid C4/sequence: **`docs/ARCHITECTURE.md`**.
- Segmentation service: **`services/segmentation_service/README.md`** and **`SKILL.md`**.
- VAD service: **`services/vad_service/README.md`** and **`SKILL.md`**.
