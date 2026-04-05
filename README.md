# Audio ASR pipeline

Library: coarse speech/music segmentation, Silero VAD, chunking, OpenAI-compatible STT, `verbose_json` merge.

Architecture (C4 + sequence, Mermaid): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Apache Airflow (PythonOperator, TaskFlow, XCom): [docs/AIRFLOW.md](docs/AIRFLOW.md). Пример DAG’ов и `plugins/asr_helpers`: [airflow_scaffold/README.md](airflow_scaffold/README.md). Cursor skill for agents: [.cursor/skills/audio-asr-pipeline/SKILL.md](.cursor/skills/audio-asr-pipeline/SKILL.md).

```bash
uv sync
uv sync --extra eval   # jiwer + openpyxl for scripts/eval_test_audio.py
uv sync --extra ina    # inaSpeechSegmenter (default coarse segmenter); pyproject pins plain tensorflow on Windows (no TF CUDA extra)
```

Resolution is pinned in **`uv.lock`** (`uv lock` / `uv sync`). For **pip**, use exported files (hashed pins): **`requirements-minimal.txt`** (core runtime only) or **`requirements.txt`** (all extras `eval`+`ina` and dev tools). After `pip install -r …`, install the package: `pip install -e .`.

```bash
uv run python scripts/eval_test_audio.py --audio-dir test_audio --base-url http://127.0.0.1:8000
uv run python scripts/eval_test_audio.py -v ...   # log each POST to STT
# Stereo call-center eval: channel 0 = call_from, 1 = call_to; refs <stem>_call_from.txt + <stem>_call_to.txt
uv run python scripts/eval_test_audio.py --stereo-call --audio-dir ... --base-url http://127.0.0.1:8000
```

By default `trust_env=False` for httpx so `HTTP_PROXY` / `HTTPS_PROXY` do not redirect local STT traffic. Use `--trust-env` only if you rely on a proxy.

Default coarse segmenter is **`ina`** (speech / music / noise via `inaSpeechSegmenter`). Install with **`uv sync --extra ina`**. TensorFlow for ina is **CPU** by default (`ina_force_cpu`); eval **`--ina-allow-gpu`** allows GPU. Without ina, use **`--coarse-backend whole_file`**.
