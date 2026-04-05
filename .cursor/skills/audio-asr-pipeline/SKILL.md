---
name: audio-asr-pipeline
description: >-
  Develops and debugs the transcription_preprocessor_pipeline repo: audio_asr_pipeline
  (coarse seg, Silero VAD ONNX, chunking, OpenAI STT, merge) and scripts/eval_test_audio.py
  (XLSX, stereo-call). Use when the user works on this workspace, ASR preprocessing, ina,
  VAD, eval reports, or Windows TF/uv extras.
---

# Audio ASR preprocessor pipeline

## Scope

Python package **`audio_asr_pipeline`** + CLI **`scripts/eval_test_audio.py`**. Do not refactor unrelated files.

## Layout

- **`audio_asr_pipeline/config.py`** — `PipelineConfig`, `VLLMTranscribeConfig`, `DEFAULT_COARSE_SEGMENTER_BACKEND` (keep eval default aligned).
- **`audio_asr_pipeline/pipeline.py`** — `AudioTranscriptionPipeline`: shared **`_file_sem`** from `max_concurrent_files`; CPU prep in `run_in_executor`; then async chunk STT.
- **`audio_asr_pipeline/preprocess.py`** — load → coarse → Silero VAD → speech `TimeSpan[]`.
- **`audio_asr_pipeline/segmenters.py`** — `ina` (map `female`/`male` → `speech`; optional `segment` vs `Segmenter()` API); `whole_file`.
- **`audio_asr_pipeline/vad.py`** — Silero via **ONNX**; no unconditional `.to()` on wrapper.
- **`audio_asr_pipeline/io.py`** — **`split_stereo_channels`** for librosa `(n,2)` or `(2,n)`; `write_mono_wav`.
- **`scripts/eval_test_audio.py`** — `--stereo-call`, `--coarse-backend`, `--ina-allow-gpu`, concurrency = file parallelism.

## Conventions

- Match existing style; minimal diffs; no drive-by renames.
- Optional deps: **`uv sync --extra eval`** (jiwer, openpyxl); **`uv sync --extra ina`** (inaSpeechSegmenter; Windows uses `[tool.uv] override-dependencies` for plain tensorflow in `pyproject.toml`).
- STT: httpx **`trust_env=False`** by default (avoid proxy hijacking localhost).

## Stereo eval

- Channel **0** → `call_from`, **1** → `call_to`; refs `{stem}_call_from.txt`, `{stem}_call_to.txt`.
- Always use **`split_stereo_channels`** after `librosa.load(..., mono=False)`; never assume `y[:,0]` is time without checking shape.

## Debugging checklist

1. **No speech spans after coarse** — check ina labels (gender → speech mapping), `drop_*` flags.
2. **ina ImportError** — install extra `ina`; message should mention `uv sync --extra ina`.
3. **Stereo near-zero duration / ina negative dimensions** — wrong channel axis; verify `split_stereo_channels`.
4. **Parallelism** — `max_concurrent_files` is instance-level semaphore; stereo uses two `process_file` calls in parallel (two slots).

## Reference

- Architecture and Mermaid C4/sequence: **`docs/ARCHITECTURE.md`**.
