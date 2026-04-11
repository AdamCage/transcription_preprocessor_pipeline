---
name: Gemma ASR backend
overview: Add a `gemma` STT backend to the audio_asr_pipeline that uses Gemma-4-E4B-it for ASR via Ollama (now) and vLLM (future), with configurable prompt templates for Russian language, synthetic verbose_json, and an option to skip VAD for debugging.
todos:
  - id: config-extend
    content: Extend STTBackendName, VADBackendName, and VLLMTranscribeConfig with Gemma fields
    status: completed
  - id: gemma-client
    content: Implement GemmaTranscriptionClient (ollama_native + openai_chat modes) in transcribe.py
    status: completed
  - id: vad-none
    content: Add vad_backend='none' support in preprocess.py to skip VAD
    status: completed
  - id: eval-cli
    content: Add --stt-backend gemma, --gemma-api-style, --no-vad to eval_test_audio.py
    status: completed
  - id: phase1-test
    content: Test on test_audio_mad without coarse/VAD
    status: completed
  - id: phase2-test
    content: Test on test_audio_mad with VAD
    status: completed
  - id: phase3-test
    content: Test on test_audio_cc2 with stereo + coarse/VAD
    status: completed
isProject: false
---

# Gemma-4-E4B-it ASR Backend

## Context

The pipeline currently supports two STT backends (`openai`, `httpx`) that target Whisper-style `/v1/audio/transcriptions`. Gemma-4-E4B-it is a multimodal LLM that does ASR via chat completions, not a dedicated transcription endpoint. Audio is sent as base64 in a chat message alongside a text prompt. Max audio length: 30 seconds (pipeline already chunks at 28s).

**Current deployment**: Ollama at `localhost:11434`, model `gemma4:e4b`.
**Future deployment**: vLLM serving `google/gemma-4-E4B-it` via OpenAI-compatible API.

## Key Design Decisions

### API layer: two styles, one protocol

Both Ollama and vLLM expose different APIs for multimodal chat:

- **Ollama native** (`/api/chat`): audio goes in the `images` field as base64. Requires raw httpx.
- **OpenAI-compatible** (`/v1/chat/completions`): audio goes as `{"type": "input_audio", "input_audio": {"data": "<b64>", "format": "wav"}}` content blocks. Uses `AsyncOpenAI`.

Both return plain text in the assistant message (no Whisper-style verbose_json).

**Decision**: Implement a `GemmaTranscriptionClient` that selects between `ollama_native` and `openai_chat` modes via config. Start with `ollama_native` for current debugging; switch to `openai_chat` when migrating to vLLM. Both reuse the same prompt template, response parsing, and retry logic.

### Verbose JSON

Gemma returns plain text with no timestamps. The client constructs a synthetic response dict per chunk:

```python
{
    "text": "<transcribed text>",
    "language": "ru",
    "segments": [{"start": 0.0, "end": <chunk_duration>, "text": "<transcribed text>"}],
}
```

The existing merge module handles combining these into a unified verbose_json. When coarse+VAD segmentation is used, each chunk maps to a real speech span with correct timeline offsets.

### Skip VAD option

The user wants to debug without coarse and VAD first. `coarse_backend=whole_file` already skips coarse filtering. We need a new `vad_backend="none"` option to skip VAD entirely -- the whole-file span is chunked purely by duration (max 28s).

---

## Changes by File

### 1. [audio_asr_pipeline/config.py](audio_asr_pipeline/config.py)

- Extend `STTBackendName`:
```python
STTBackendName = Literal["httpx", "openai", "gemma"]
```

- Extend `VADBackendName`:
```python
VADBackendName = Literal["silero", "none"]
```

- Add Gemma-specific fields to `VLLMTranscribeConfig` (reuse existing config class to minimize diff):
  - `gemma_api_style: Literal["ollama_native", "openai_chat"] = "ollama_native"` -- Ollama now, vLLM later
  - `gemma_asr_prompt: str | None = None` -- custom prompt; if None, auto-generated from `language`
  - `gemma_max_tokens: int = 512` -- max_new_tokens for generation
  - `gemma_thinking: bool = False` -- enable Gemma thinking mode

### 2. [audio_asr_pipeline/transcribe.py](audio_asr_pipeline/transcribe.py)

New class `GemmaTranscriptionClient` implementing `STTClient`:

```python
class GemmaTranscriptionClient:
    def __init__(self, config: VLLMTranscribeConfig) -> None: ...

    async def transcribe_chunk(self, chunk: AudioChunk, **kw) -> dict[str, Any]:
        # 1. base64-encode chunk.audio_bytes
        # 2. build prompt (from config or default Russian ASR template)
        # 3. dispatch to _ollama_native() or _openai_chat() based on gemma_api_style
        # 4. extract text from response
        # 5. return synthetic verbose_json-compatible dict

    async def _ollama_native(self, audio_b64, prompt) -> str:
        # POST /api/chat {"model": ..., "stream": false,
        #   "messages": [{"role":"user","content": prompt, "images": [audio_b64]}]}

    async def _openai_chat(self, audio_b64, prompt) -> str:
        # AsyncOpenAI.chat.completions.create(...) with audio content blocks

    async def aclose(self) -> None: ...
```

**Default ASR prompt** (based on Gemma docs, tailored for Russian):
```
Transcribe the following speech segment in Russian into Russian text.
Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits.
```

Update `create_stt_client()` to handle `"gemma"` backend.

### 3. [audio_asr_pipeline/preprocess.py](audio_asr_pipeline/preprocess.py)

Handle `vad_backend == "none"`: skip Silero VAD, pass coarse speech spans directly to the output (still filtered by `min_segment_duration_sec`).

```python
if config.vad_backend == "none":
    timings["vad_sec"] = 0.0
    refined = speech_spans
else:
    # existing Silero VAD logic
```

### 4. [scripts/eval_test_audio.py](scripts/eval_test_audio.py)

- Extend `--stt-backend` choices: `("httpx", "openai", "gemma")`
- Add new CLI flags:
  - `--gemma-api-style {ollama_native,openai_chat}` (default: `ollama_native`)
  - `--no-vad` flag -> sets `vad_backend="none"` in PipelineConfig
- Default `--model` to `gemma4:e4b` when `--stt-backend gemma` (unless explicitly set)
- Default `--base-url` to `http://localhost:11434` when `--stt-backend gemma`
- Default `--language` to `ru` when `--stt-backend gemma` and language not set
- Adjust STT log line to show chat endpoint instead of `/v1/audio/transcriptions`

### 5. [pyproject.toml](pyproject.toml)

No new dependencies needed: `httpx` (already present) handles Ollama native API; `openai` (already present) handles future vLLM chat completions.

---

## Testing / Debugging Phases

### Phase 1: Gemma without coarse/VAD on test_audio_mad

```bash
uv run python scripts/eval_test_audio.py \
  --stt-backend gemma \
  --audio-dir test_audio_mad \
  --base-url http://localhost:11434 \
  --model gemma4:e4b \
  --language ru \
  --coarse-backend whole_file \
  --no-vad \
  -v
```

Validates: Gemma client works, audio encoding correct, prompt produces good Russian ASR, merge produces valid verbose_json.

### Phase 2: Gemma with coarse + VAD on test_audio_mad

```bash
uv run python scripts/eval_test_audio.py \
  --stt-backend gemma \
  --audio-dir test_audio_mad \
  --base-url http://localhost:11434 \
  --coarse-backend whole_file \
  --language ru \
  -v
```

Validates: VAD chunking works correctly with Gemma, multi-chunk merge works.

### Phase 3: Gemma on test_audio_cc2 (stereo call-center)

```bash
uv run python scripts/eval_test_audio.py \
  --stt-backend gemma \
  --stereo-call \
  --audio-dir test_audio_cc2 \
  --base-url http://localhost:11434 \
  --coarse-backend ina \
  --language ru \
  -v
```

Validates: stereo splitting + coarse/VAD + Gemma ASR end-to-end.

---

## Known Risks

- **Ollama crash bug**: gemma4:e4b has intermittent GGML assertion crashes during audio inference ([ollama#15333](https://github.com/ollama/ollama/issues/15333)). Mitigation: robust retry with exponential backoff, ensure 16kHz mono input.
- **30-second audio limit**: Gemma supports max 30s per clip. Pipeline chunks at 28s -- safe margin.
- **No word-level timestamps**: Gemma returns only text. Verbose_json will have chunk-level segments only.
