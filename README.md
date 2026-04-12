# Audio ASR pipeline

Библиотека **`audio_asr_pipeline`**: грубая сегментация речи/музыки/шума (`ina`, `whole_file` или удалённый GPU-сервис), VAD (Silero локально или удалённый сервис), нарезка чанков, распознавание через STT (Whisper по OpenAI-совместимому `/v1/audio/transcriptions` **или** Gemma-4 через chat API), склейка текста и **`verbose_json`**.

Подробности по архитектуре и интеграциям: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/AIRFLOW.md](docs/AIRFLOW.md). GPU-сервисы: [services/segmentation_service/README.md](services/segmentation_service/README.md), [services/vad_service/README.md](services/vad_service/README.md).

## Установка

```bash
uv sync
```

## Скрипт `scripts/eval_test_audio.py`

Пакетная обработка WAV из каталога: препроцессинг пайплайна, транскрипция чанков через выбранный STT-бэкенд, отчёт **XLSX** (WER/CER, тайминги, RT) и каталог **`verbose_json`**. Эталонные `.txt` для метрик необязательны.

## Флаги

| Флаг | По умолчанию | Описание |
|------|----------------|----------|
| `--audio-dir` | `test_audio` | Каталог с `.wav` (и при стерео — эталонами рядом). |
| `--base-url` | `http://127.0.0.1:8000` | Базовый URL STT (Whisper и т.п.). Для `--stt-backend gemma` без явной подстановки скрипт может сменить URL на Ollama (см. ниже). |
| `--concurrency` | `3` | Сколько файлов обрабатывается параллельно (coarse + VAD + чанки, затем STT). В стерео один исходный WAV занимает **два** слота (по каналу). |
| `--model` | `large-v3-turbo` | Имя модели на STT-сервере. При `--stt-backend gemma` и дефолтном значении подставляется `gemma4:e4b`. |
| `--language` | авто (`None`) | Код языка ISO 639-1. При `--stt-backend gemma` и отсутствии значения подставляется `ru`. |
| `--output-dir` | нет | Каталог результатов; если не задан — `eval_runs/<UTC_timestamp>`. |
| `-v` / `--verbose` | выкл | DEBUG для скрипта и `audio_asr_pipeline` (в т.ч. запросы к STT). |
| `--log-level` | `INFO` | Уровень лога: `DEBUG`, `INFO`, `WARNING`, `ERROR` (`-v` принудительно DEBUG). |
| `--log-file` | `<output-dir>/eval.log` | Файл лога UTF-8 (если не отключён). |
| `--no-log-file` | выкл | Только stderr, без файла в `output-dir`. |
| `--trust-env` | выкл | Разрешить httpx использовать `HTTP_PROXY` / `HTTPS_PROXY` из окружения. |
| `--coarse-backend` | `ina` | Грубый сегментатор: `ina`, `whole_file` или `remote` (HTTP, см. `--segmentation-url`). |
| `--ina-allow-gpu` | выкл | Разрешить TensorFlow для ina использовать GPU (по умолчанию только CPU). |
| `--segmentation-url` | `http://127.0.0.1:8001` | URL **segmentation_service** для `--coarse-backend remote`. |
| `--vad-url` | `http://127.0.0.1:8002` | URL **vad_service** для `--vad-backend remote`. |
| `--vad-backend` | см. ниже | Явно: `silero` (локально), `remote` (GPU-сервис) или `none`. Если флаг **не** передан: при `--no-vad` — `none`, иначе `silero`. Явный `--vad-backend` перекрывает `--no-vad`. |
| `--stereo-call` | выкл | Стерео WAV: канал 0 = `call_from`, 1 = `call_to`; эталоны `{stem}_call_from.txt` и `{stem}_call_to.txt`. |
| `--stt-backend` | `openai` | Клиент STT: `openai`, `httpx` или `gemma` (Gemma-4 через Ollama/vLLM). |
| `--api-key` | нет | Ключ для STT (`Authorization: Bearer`), если сервер требует. |
| `--no-vad` | выкл | Пропустить VAD (границы по длительности чанков); учитывается только если `--vad-backend` не задан явно. |
| `--gemma-api-style` | `ollama_native` | Для `--stt-backend gemma`: `ollama_native` или `openai_chat` (vLLM). |

Поведение **`--stt-backend gemma`**: если `base-url` остался дефолтным `http://127.0.0.1:8000`, подставляется `http://localhost:11434`; при дефолтной модели — `gemma4:e4b`; при отсутствии `--language` — `ru` (всё можно переопределить явно).

## Запуск eval с `segmentation_service` и `vad_service`

Перед запуском поднимите STT (например vLLM Whisper на порту 8000), **segmentation_service** (обычно порт **8001**) и **vad_service** (обычно **8002**) — см. README в `services/segmentation_service` и `services/vad_service`.

**Моно** (один канал на файл):

```bash
uv run python scripts/eval_test_audio.py \
  --audio-dir test_audio \
  --base-url http://127.0.0.1:8000 \
  --coarse-backend remote \
  --segmentation-url http://127.0.0.1:8001 \
  --vad-backend remote \
  --vad-url http://127.0.0.1:8002
```

Замените хосты/порты на те, где реально слушают сервисы (например удалённый GPU: `http://gpu-host:8001`).

**Стерео** (call-center: два канала, эталоны по шаблону имён):

```bash
uv run python scripts/eval_test_audio.py \
  --stereo-call \
  --audio-dir stereo_wavs \
  --base-url http://127.0.0.1:8000 \
  --coarse-backend remote \
  --segmentation-url http://127.0.0.1:8001 \
  --vad-backend remote \
  --vad-url http://127.0.0.1:8002 \
  --concurrency 4
```

В стерео один WAV использует два слота параллелизма; `--concurrency` лучше брать с запасом.

## Прочее

Cursor skill для агентов: [.cursor/skills/audio-asr-pipeline/SKILL.md](.cursor/skills/audio-asr-pipeline/SKILL.md).
