---
name: audio-asr-pipeline
description: >-
  Develops and debugs the transcription_preprocessor_pipeline repo: audio_asr_pipeline
  (coarse seg, Silero VAD ONNX, chunking, OpenAI STT via openai/httpx, merge) and
  scripts/eval_test_audio.py (XLSX, stereo-call, tqdm). Use when the user works on this
  workspace, ASR preprocessing, ina, VAD, eval reports, or Windows TF/uv extras.
---

# Audio ASR preprocessor pipeline

## Scope

Python package **`audio_asr_pipeline`** + CLI **`scripts/eval_test_audio.py`**. Do not refactor unrelated files.

## Layout

- **`audio_asr_pipeline/config.py`** — `PipelineConfig`, `VLLMTranscribeConfig` (`stt_backend`, `api_key`, `base_url`, `model`), `STTBackendName`, `DEFAULT_COARSE_SEGMENTER_BACKEND`.
- **`audio_asr_pipeline/pipeline.py`** — `AudioTranscriptionPipeline`: shared **`_file_sem`** from `max_concurrent_files`; CPU prep in `run_in_executor`; then async chunk STT. Dual backend via `create_stt_client`.
- **`audio_asr_pipeline/transcribe.py`** — `OpenAITranscriptionClient` (openai `AsyncOpenAI`, recommended for vLLM), `VLLMTranscriptionClient` (httpx raw), `create_stt_client` factory, `STTClient` Protocol.
- **`audio_asr_pipeline/preprocess.py`** — load → coarse → Silero VAD → speech `TimeSpan[]`.
- **`audio_asr_pipeline/segmenters.py`** — `ina` (map `female`/`male` → `speech`; optional `segment` vs `Segmenter()` API); `whole_file`.
- **`audio_asr_pipeline/vad.py`** — Silero via **ONNX**; no unconditional `.to()` on wrapper.
- **`audio_asr_pipeline/io.py`** — **`split_stereo_channels`** for librosa `(n,2)` or `(2,n)`; `write_mono_wav`.
- **`scripts/eval_test_audio.py`** — `--stereo-call`, `--coarse-backend`, `--ina-allow-gpu`, `--stt-backend`, `--api-key`, `--concurrency`, tqdm progress.

## Conventions

- Match existing style; minimal diffs; no drive-by renames.
- **`uv sync`** installs ALL dependency-groups (dev, eval, ina) via `[tool.uv] default-groups`. Optional deps for pip: `pip install .[eval,ina]`. Windows: `[tool.uv] override-dependencies` pins plain tensorflow.
- STT default backend: **`openai`** (`AsyncOpenAI`); fallback: **`httpx`** (raw multipart). `trust_env=False` by default (avoid proxy hijacking localhost).

## Stereo eval

- Channel **0** → `call_from`, **1** → `call_to`; refs `{stem}_call_from.txt`, `{stem}_call_to.txt`.
- Always use **`split_stereo_channels`** after `librosa.load(..., mono=False)`; never assume `y[:,0]` is time without checking shape.
- WER/CER: when hypothesis is empty but reference exists, reports WER=1.0 / CER=1.0.

## Debugging checklist

1. **No speech spans after coarse** — check ina labels (gender → speech mapping), `drop_*` flags.
2. **ina ImportError** — `uv sync` should install it; if excluded: `uv sync --no-group ina` was used.
3. **Stereo near-zero duration / ina negative dimensions** — wrong channel axis; verify `split_stereo_channels`.
4. **Parallelism** — `max_concurrent_files` is instance-level semaphore; `max_in_flight_requests` is global STT cap. Stereo uses two `process_file` calls in parallel (two slots). Check DEBUG logs for `acquired_file_sem` / `stt_send` to verify concurrency.
5. **STT auth errors (401/403)** — set `api_key` in `VLLMTranscribeConfig` or `--api-key` in eval CLI.

## Reference

- Architecture and Mermaid C4/sequence: **`docs/ARCHITECTURE.md`**.
