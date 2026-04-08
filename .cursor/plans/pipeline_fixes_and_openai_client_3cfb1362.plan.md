---
name: Pipeline fixes and OpenAI client
overview: "Fix four issues in the transcription pipeline: restructure dependencies for single-command install, harden stereo-call WER/CER calculation, add `openai` AsyncClient mode for vLLM, and fix the concurrency model so `--concurrency N` actually parallelizes work. Then update docs, skill, and airflow scaffold."
todos:
  - id: deps
    content: "Restructure pyproject.toml: add dependency-groups (eval, ina), default-groups in [tool.uv], add openai to core deps. Regenerate uv.lock."
    status: completed
  - id: stereo-fix
    content: "Fix stereo WER/CER edge case: guard against empty normalized hypothesis in overall metrics block; consider showing WER=1.0 when hyp empty + ref exists."
    status: completed
  - id: openai-client
    content: Add OpenAITranscriptionClient in transcribe.py; add stt_backend + api_key to VLLMTranscribeConfig; update pipeline.py for dual-client selection.
    status: completed
  - id: concurrency
    content: "Fix concurrency: explicit httpx pool limits, flatten nested semaphores in _transcribe_all, raise default max_concurrent_chunks, add diagnostic logging."
    status: completed
  - id: eval-cli
    content: Add --stt-backend and --api-key flags to eval_test_audio.py; wire through to VLLMTranscribeConfig.
    status: completed
  - id: docs
    content: Update ARCHITECTURE.md, AIRFLOW.md, README.md, SKILL.md, and airflow_scaffold config with new features.
    status: completed
  - id: regen
    content: Regenerate uv.lock and requirements-minimal.txt after all changes.
    status: completed
  - id: todo-1775688388094-2gf6xolkz
    content: Add tqdm for indicating progress for all audio files in `eval_test-audio.py`
    status: completed
isProject: false
---

# Pipeline fixes: deps, stereo, openai client, concurrency

## 1. Fix `uv sync` to install all packages

**Problem:** `uv sync` only installs core deps; `eval` (jiwer, openpyxl) and `ina` (inaSpeechSegmenter) require extra flags (`--extra eval`, `--extra ina`).

**Approach:** Use PEP 735 **dependency-groups** with `default-groups` in `[tool.uv]`. This keeps `[project.optional-dependencies]` for `pip install .[eval]` compatibility, while making `uv sync` pull everything by default.

Changes in [pyproject.toml](pyproject.toml):

```toml
[dependency-groups]
dev = ["pytest>=8.0.0"]
eval = ["jiwer>=3.0.0", "openpyxl>=3.1.0"]
ina = ["inaSpeechSegmenter>=0.7.0"]

[tool.uv]
default-groups = ["dev", "eval", "ina"]
```

Also add `openai>=1.0.0` to core `[project].dependencies` (needed for issue 3). Regenerate `uv.lock` and `requirements-minimal.txt`.

---

## 2. Stereo-call WER/CER: edge cases and clarity

**Status:** The stereo mode works correctly overall. Reference files `{stem}_call_from.txt` / `{stem}_call_to.txt` are loaded from the same directory as the stereo WAV. Per-channel and overall WER/CER are computed via `jiwer`.

**Bug found:** In the overall WER/CER block (`_run_stereo`, around line 688), if `ref_join` is non-empty but `hyp_join` normalizes to an empty string (both channels produced no text), `jiwer.wer(rj, "")` may raise `ValueError` in some jiwer versions. Needs a defensive check:

```python
if rj and hj:
    wer_o = float(jiwer.wer(rj, hj))
    cer_o = float(jiwer.cer(rj, hj))
```

Same pattern for per-channel blocks: already guarded (`if ref_f.strip() and hyp_nf`), but the overall calculation is not.

**Minor fix:** Per-channel WER/CER should show `1.0` when hypothesis is empty but reference exists (currently shows blank -- this is a design choice to reconsider).

---

## 3. Add `openai` AsyncClient mode for vLLM

**Problem:** Current STT goes through raw `httpx` multipart POST. vLLM works better with the `openai` library which handles auth headers, retries, connection pooling, and model specification natively.

### 3a. Config changes in [config.py](audio_asr_pipeline/config.py)

Add to `VLLMTranscribeConfig`:
- `stt_backend: Literal["httpx", "openai"] = "openai"` -- default to openai for better vLLM compat
- `api_key: str | None = None` -- `Authorization: Bearer <key>` for vLLM `--api-key`

### 3b. New client in [transcribe.py](audio_asr_pipeline/transcribe.py)

Add `OpenAITranscriptionClient` alongside existing `VLLMTranscriptionClient`:

```python
from openai import AsyncOpenAI

class OpenAITranscriptionClient:
    def __init__(self, config: VLLMTranscribeConfig) -> None:
        self._cfg = config
        self._client: AsyncOpenAI | None = None

    async def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._cfg.api_key or "EMPTY",
                base_url=self._cfg.base_url.rstrip("/") + "/v1",
                timeout=self._cfg.request_timeout_sec,
                max_retries=self._cfg.max_retries,
            )
        return self._client

    async def transcribe_chunk(self, chunk: AudioChunk, **kw) -> dict:
        client = await self._ensure_client()
        granularities = ["segment"]
        if self._cfg.include_word_timestamps:
            granularities.append("word")
        resp = await client.audio.transcriptions.create(
            model=self._cfg.model,
            file=(f"{chunk.chunk_id}.wav", chunk.audio_bytes, "audio/wav"),
            response_format="verbose_json",
            timestamp_granularities=granularities,
            language=self._cfg.language,
            temperature=self._cfg.temperature,
            prompt=self._cfg.prompt,
        )
        return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
```

Key advantage: `AsyncOpenAI` manages its own connection pool, retries (429/503), and sends `Authorization` header.

### 3c. Pipeline changes in [pipeline.py](audio_asr_pipeline/pipeline.py)

- In `__init__`, choose client based on `config.vllm.stt_backend`:
  - `"openai"` -> `OpenAITranscriptionClient`
  - `"httpx"` -> `VLLMTranscriptionClient` (existing)
- When using openai client, skip manual httpx client creation (`_ensure_http_client`)
- Adjust `_transcribe_all`: openai client does not need an external `httpx.AsyncClient` param
- `aclose()`: close the openai client if used

### 3d. Eval script CLI in [eval_test_audio.py](scripts/eval_test_audio.py)

Add flags:
- `--stt-backend {httpx,openai}` (default: `openai`)
- `--api-key` (string, forwarded to `VLLMTranscribeConfig.api_key`)

---

## 4. Fix concurrency model

**Diagnosis:** The `asyncio.gather` + semaphore architecture is correct. Multiple files DO start concurrently, CPU prep runs in `ThreadPoolExecutor`, and STT chunks use async I/O. The observed sequential behavior likely stems from:

- **CPU prep serialization under GIL** -- `_prepare_file_cpu` (librosa, numpy, ina/TF) runs in threads that compete for GIL
- **Single-file STT dominance** -- each file's `_transcribe_all` issues `gather(*chunks)`, but the next file's chunks don't start until ITS CPU prep finishes (in a thread)
- **httpx connection pool** -- default is fine (100 max), but explicit limits improve clarity
- **Semaphore nesting** -- `_global_stt_sem` then `chunk_sem` is correct but could serialize under high load if global sem fills

### Fixes in [pipeline.py](audio_asr_pipeline/pipeline.py)

- **Explicit httpx pool limits** when using httpx backend:
  

```python
  httpx.AsyncClient(
      limits=httpx.Limits(
          max_connections=max(100, cfg.max_in_flight_requests * 2),
          max_keepalive_connections=cfg.max_in_flight_requests,
      ),
      ...
  )
  

```
- **Flatten semaphore nesting** in `_transcribe_all`: remove the redundant `chunk_sem` and rely solely on `_global_stt_sem` for limiting in-flight requests. Currently two nested semaphores create unnecessary serialization.
- **Diagnostic logging** -- log when file-sem is acquired/released and how many STT requests are in-flight, so the user can see actual concurrency in action.

### Fixes in [config.py](audio_asr_pipeline/config.py)

- Default `max_concurrent_chunks` raised to match `max_in_flight_requests` (currently defaults to 3, which limits per-file chunk parallelism regardless of `--concurrency`).

### Eval script fix

- Pass `max_concurrent_chunks` proportional to `--concurrency` (currently hardcoded equal to `args.concurrency`, which is correct -- but the defaults in `PipelineConfig` itself are the issue).

---

## 5. Update docs, skills, and airflow scaffold

### [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Add openai client mode to component diagram and sequence diagrams
- Document dual-backend (httpx / openai) in section 9 (dependencies)
- Update config table in section 7 with new fields

### [docs/AIRFLOW.md](docs/AIRFLOW.md)
- Update config examples with `stt_backend` and `api_key`
- Note that `openai` mode is recommended for vLLM

### [airflow_scaffold/plugins/asr_helpers/config.py](airflow_scaffold/plugins/asr_helpers/config.py)
- Add `api_key` from Airflow Variable/Connection to `VLLMTranscribeConfig`
- Add `stt_backend` support

### [.cursor/skills/audio-asr-pipeline/SKILL.md](.cursor/skills/audio-asr-pipeline/SKILL.md)
- Update conventions: `uv sync` now installs everything; extras kept for pip
- Add `stt_backend` / `api_key` to config reference
- Update debugging checklist with concurrency diagnostics

### [README.md](README.md)
- Simplify install section: `uv sync` (installs all) -- no extra flags needed
- Add `--stt-backend openai --api-key ...` to eval examples
- Update requirements file references

### requirements-minimal.txt
- Regenerate with `uv export --frozen ...` after lock update

---

## Key files changed (summary)

- `pyproject.toml` -- dependency-groups, openai dep, default-groups
- `audio_asr_pipeline/config.py` -- `stt_backend`, `api_key`
- `audio_asr_pipeline/transcribe.py` -- `OpenAITranscriptionClient`
- `audio_asr_pipeline/pipeline.py` -- dual-client support, pool limits, flatten semaphores
- `audio_asr_pipeline/__init__.py` -- export new names if needed
- `scripts/eval_test_audio.py` -- `--stt-backend`, `--api-key`, stereo WER fix
- `docs/ARCHITECTURE.md`, `docs/AIRFLOW.md` -- updated diagrams/examples
- `airflow_scaffold/plugins/asr_helpers/config.py` -- api_key/backend support
- `.cursor/skills/audio-asr-pipeline/SKILL.md` -- updated reference
- `README.md` -- simplified install, new CLI flags
