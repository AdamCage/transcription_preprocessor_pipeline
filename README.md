# Audio ASR pipeline

Библиотека **`audio_asr_pipeline`**: грубая сегментация речи/музыки/шума, VAD (Silero через ONNX), нарезка чанков, вызов STT (Whisper через OpenAI-совместимый `/v1/audio/transcriptions` **или** Gemma-4 через chat API), склейка текста и **`verbose_json`**.

Подробная архитектура (C4, последовательности, ошибки, лимиты): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Конвейер данных (кратко)

1. Загрузка WAV → нормализация сэмплрейта / моно (по конфигу).
2. **Coarse** (`ina`, `whole_file` или `remote` — GPU-сервис pyannote) → интервалы с метками.
3. **VAD** уточняет границы речи (`silero`, `remote` — GPU-сервис, или `none`); фильтрация по конфигу.
4. **Chunking** — WAV-байты на чанк для HTTP.
5. **Transcribe** — вызов STT через один из трёх бэкендов (см. ниже); лимит `max_in_flight_requests`.
6. **Merge** — итоговый текст и `verbose_json`.

Упрощённая схема модулей (см. также Mermaid в [ARCHITECTURE.md](docs/ARCHITECTURE.md)):

```mermaid
flowchart LR
    subgraph External["Внешнее"]
        STT["STT API"]
        WAV["WAV файлы"]
    end
    APP["audio_asr_pipeline"]
    APP --> WAV
    APP --> STT
```

```mermaid
flowchart TB
    subgraph pkg["audio_asr_pipeline"]
        P[pipeline.py]
        PR[preprocess.py]
        SG[segmenters.py]
        V[vad.py]
        CH[chunking.py]
        T[transcribe.py]
        M[merge.py]
        IO[io.py]
    end
    STT["STT HTTP"]
    P --> PR
    PR --> SG
    PR --> V
    P --> CH
    P --> T
    P --> M
    PR --> IO
    CH --> IO
    T --> STT
```

| Модуль | Роль |
|--------|------|
| `pipeline.py` | `AudioTranscriptionPipeline`, `process_file_sync`, оркестрация (local + remote) |
| `preprocess.py` | coarse + VAD + спаны (local backends) |
| `segmenters.py` | `ina` / `whole_file` |
| `vad.py` | Silero ONNX (local) |
| `remote_clients.py` | `RemoteSegmentationClient`, `RemoteVADClient` — HTTP-клиенты GPU-сервисов |
| `chunking.py` | ограничения длины чанка |
| `transcribe.py` | STT: `OpenAITranscriptionClient` (openai lib) / `VLLMTranscriptionClient` (httpx) / `GemmaTranscriptionClient` (chat API); ретраи / 429 |
| `merge.py` | склейка ответов |
| `config.py` | `PipelineConfig`, `VLLMTranscribeConfig`, service URLs |

Результат **`PipelineResult`**: `text`, `verbose_json`, `stats`, опционально **`error`** — при `fail_fast=False` сбой одного файла в батче не валит остальные (см. [ARCHITECTURE.md](docs/ARCHITECTURE.md)).

## Установка

```bash
uv sync                # ставит ВСЕ зависимости (core + eval + ina + dev) через dependency-groups
uv sync --no-group ina # без inaSpeechSegmenter / TensorFlow (лёгкий режим, coarse = whole_file)
```

Пины в **`uv.lock`**. Для **pip**: `pip install -e .[eval,ina]` из корня репозитория. Минимальный pip-набор: `pip install -e .` (без jiwer/openpyxl/ina). Экспорт lock-файла: **`requirements-minimal.txt`** (ядро).

## Eval (батч WAV → XLSX + verbose_json)

```bash
# По умолчанию STT через openai клиент (рекомендуется для vLLM):
uv run python scripts/eval_test_audio.py --audio-dir test_audio --base-url http://127.0.0.1:8000

# С API-ключом (vLLM --api-key):
uv run python scripts/eval_test_audio.py --audio-dir test_audio --base-url http://127.0.0.1:8000 --api-key sk-xxx

# Через httpx (raw multipart POST):
uv run python scripts/eval_test_audio.py --stt-backend httpx --audio-dir test_audio --base-url http://127.0.0.1:8000

uv run python scripts/eval_test_audio.py -v ...   # лог каждого POST в STT
# Стерео call-center: канал 0 = call_from, 1 = call_to; эталоны <stem>_call_from.txt и <stem>_call_to.txt
uv run python scripts/eval_test_audio.py --stereo-call --audio-dir ... --base-url http://127.0.0.1:8000
```

По умолчанию **`trust_env=False`** (локальный STT не уезжает в `HTTP_PROXY`). Включайте **`--trust-env`** только если прокси нужен.

Дефолтный coarse — **`ina`** (`uv sync` ставит его автоматически). TF для ina по умолчанию на CPU (`ina_force_cpu`); флаг eval **`--ina-allow-gpu`** разрешает GPU. Без ina: **`--coarse-backend whole_file`**.

## Eval stereo call-center (два канала + .txt-эталоны)

Режим `--stereo-call` предназначен для стерео WAV-записей телефонных звонков, где каждый канал содержит речь отдельного участника. Скрипт разделяет стерео на два моно-файла, прогоняет каждый через пайплайн независимо и сравнивает результат с эталонными текстами.

### Структура директории

Все `.wav` и `.txt` лежат в одной папке. Для каждого стерео WAV создаётся пара эталонных текстов по шаблону имени:

```
stereo_wavs/
  call_001.wav                  # стерео WAV (2 канала)
  call_001_call_from.txt        # эталон для канала 0 (звонящий)
  call_001_call_to.txt          # эталон для канала 1 (принимающий)
  call_002.wav
  call_002_call_from.txt
  call_002_call_to.txt
  call_003.wav                  # .txt необязательны — без них WER/CER будут пустыми
```

- **Канал 0** (`call_from`) — звонящий (левый канал стерео).
- **Канал 1** (`call_to`) — принимающий (правый канал стерео).
- `.txt`-файлы: обычный текст в UTF-8, одна или несколько строк. Пустые строки игнорируются, непустые склеиваются через пробел.
- Если `.txt` отсутствует или пуст — строка в XLSX всё равно появится, но колонки WER/CER и эталон будут пустыми, а расхождения покажут «нет эталона».
- Если эталон есть, но гипотеза пуста (пайплайн не распознал речь) — WER и CER выставляются в `1.0`.

### Запуск для vLLM

```bash
uv run python scripts/eval_test_audio.py \
  --stereo-call \
  --audio-dir ./stereo_wavs \
  --base-url http://127.0.0.1:8000 \
  --api-key sk-your-vllm-key \
  --model large-v3-turbo \
  --concurrency 6 \
  --language ru \
  --coarse-backend ina \
  --output-dir ./eval_results \
  -v
```

### Описание флагов

| Флаг | Значение по умолчанию | Описание |
|------|----------------------|----------|
| `--stereo-call` | выкл | Включает стерео-режим: канал 0 = `call_from`, канал 1 = `call_to`. Каждый канал обрабатывается как отдельный моно-файл. Один стерео WAV занимает **два** слота параллелизма. |
| `--audio-dir` | `test_audio` | Путь к директории с `.wav`-файлами (и `.txt`-эталонами рядом с ними). |
| `--base-url` | `http://127.0.0.1:8000` | URL STT-сервера (OpenAI-совместимый endpoint). Для vLLM: адрес, на котором запущен `vllm serve`. |
| `--api-key` | нет | API-ключ для авторизации на STT-сервере (отправляется как `Authorization: Bearer <key>`). Обязателен, если vLLM запущен с `--api-key`. |
| `--stt-backend` | `openai` | Бэкенд HTTP-клиента: `openai` (библиотека `openai`, рекомендуется для Whisper на vLLM), `httpx` (raw multipart POST) или `gemma` (chat-based ASR через Gemma-4, см. раздел ниже). |
| `--model` | `large-v3-turbo` | Название модели Whisper на сервере. Передаётся в поле `model` запроса. |
| `--language` | нет (авто) | Код языка ISO 639-1 (`ru`, `en`, `de` и т.д.). Если не задан, модель определяет язык автоматически. |
| `--concurrency` | `3` | Максимум файлов, обрабатываемых параллельно (препроцессинг + STT). В стерео-режиме один WAV занимает два слота (по каналу). Рекомендуется ставить кратно числу GPU-воркеров на STT-сервере. |
| `--coarse-backend` | `ina` | Грубый сегментатор: `ina` (inaSpeechSegmenter), `whole_file` (всё = речь) или `remote` (GPU-сервис pyannote, см. `--segmentation-url`). |
| `--ina-allow-gpu` | выкл | Разрешить TensorFlow (для ina) использовать GPU. По умолчанию ina работает на CPU. |
| `--segmentation-url` | `http://127.0.0.1:8001` | URL remote-сервиса сегментации (для `--coarse-backend remote`). |
| `--vad-url` | `http://127.0.0.1:8002` | URL remote-сервиса VAD (для `--vad-backend remote`). |
| `--vad-backend` | `silero` | Бэкенд VAD: `silero` (локальный CPU), `remote` (GPU-сервис) или `none` (пропустить). Перекрывает `--no-vad`. |
| `--output-dir` | `eval_runs/<UTC_timestamp>` | Директория для результатов. Внутри создаются `report.xlsx`, `verbose_json/`, `stereo_mono_wav/`, `eval.log`. |
| `-v` / `--verbose` | выкл | Включает уровень DEBUG для eval-скрипта и пакета `audio_asr_pipeline`. Показывает каждый POST в STT, семафоры, тайминги. |
| `--log-level` | `INFO` | Уровень логирования (`DEBUG`, `INFO`, `WARNING`, `ERROR`). `-v` форсирует `DEBUG`. |
| `--log-file` | `<output-dir>/eval.log` | Путь к файлу лога. По умолчанию создаётся рядом с `report.xlsx`. |
| `--no-log-file` | выкл | Логировать только в stderr (не создавать файл лога). |
| `--trust-env` | выкл | Разрешить httpx использовать `HTTP_PROXY` / `HTTPS_PROXY` из окружения. По умолчанию выключено, чтобы локальный STT не уходил через прокси. |

### Что на выходе

```
eval_results/
  report.xlsx              # XLSX с листами per_file и summary
  verbose_json/
    call_001_call_from_verbose.json
    call_001_call_to_verbose.json
    call_002_call_from_verbose.json
    ...
  stereo_mono_wav/
    call_001__call_from.wav    # моно WAV канала 0 (промежуточный)
    call_001__call_to.wav      # моно WAV канала 1
    ...
  eval.log                 # полный лог прогона
```

Лист **per_file** в `report.xlsx` содержит для каждого стерео WAV: тайминги препроцессинга и транскрипции по каналам, эталонный и гипотезный текст, WER/CER по каналам и overall, описание расхождений.

Лист **summary** — агрегаты (mean, median, p25, p75) по WER/CER, длительностям, RT-метрикам.

## Apache Airflow

Паттерны интеграции (async `await pipeline.process_file` vs **`process_file_sync`** в синхронных тасках, вложенный event loop, XCom, **`expand` по списку путей**): [docs/AIRFLOW.md](docs/AIRFLOW.md).

Рекомендация: **`audio_asr_pipeline` ставить в образ/venv воркера** (`pip install .` / wheel), а не копировать исходники в DAG. Пример двух DAG (стерео call-center + моно) и тонкий слой **`plugins/asr_helpers`**: [airflow_scaffold/README.md](airflow_scaffold/README.md).

## Remote GPU Services (Segmentation + VAD)

Для production-масштаба (тысячи файлов через Airflow) сегментация и VAD выносятся в отдельные **FastAPI GPU-сервисы**. Библиотека вызывает их по HTTP через `coarse_segmenter_backend="remote"` и `vad_backend="remote"`.

### Архитектура

```mermaid
flowchart TB
    subgraph airflow_vm["Airflow VM (CPU)"]
        LIB["audio_asr_pipeline\n(remote backends)"]
    end
    subgraph gpu_vm["GPU VM"]
        SEG["Segmentation Service\npyannote :8001"]
        VAD_SVC["VAD Service\nSilero GPU :8002"]
        STT["STT Service\nvLLM Whisper :8000"]
    end
    LIB -->|"POST /segment"| SEG
    LIB -->|"POST /refine"| VAD_SVC
    LIB -->|"POST /v1/audio/transcriptions"| STT
```

### Запуск сервисов

```bash
# Segmentation Service (pyannote, GPU)
cd services/segmentation_service
export SEG_HF_TOKEN="hf_..."
uvicorn segmentation_service.app:app --host 0.0.0.0 --port 8001

# VAD Service (Silero, GPU)
cd services/vad_service
uvicorn vad_service.app:app --host 0.0.0.0 --port 8002
```

### Eval с remote backends

```bash
uv run python scripts/eval_test_audio.py \
  --coarse-backend remote \
  --segmentation-url http://gpu-host:8001 \
  --vad-backend remote \
  --vad-url http://gpu-host:8002 \
  --audio-dir test_audio \
  --base-url http://gpu-host:8000 \
  --concurrency 10
```

### Программный API

```python
from audio_asr_pipeline import PipelineConfig, AudioTranscriptionPipeline

cfg = PipelineConfig(
    coarse_segmenter_backend="remote",
    vad_backend="remote",
    segmentation_service_url="http://gpu-host:8001",
    vad_service_url="http://gpu-host:8002",
)

async with AudioTranscriptionPipeline(cfg) as pipeline:
    result = await pipeline.process_file(Path("audio.wav"))
```

Подробная документация сервисов: [services/segmentation_service/README.md](services/segmentation_service/README.md) и [services/vad_service/README.md](services/vad_service/README.md).

## Gemma-4 ASR (chat-based STT)

Бэкенд `gemma` позволяет использовать мультимодальную модель **Gemma-4** (`gemma-4-E4B-it` и др.) для распознавания речи через chat completions API. Gemma принимает аудио в base64, возвращает plain-text транскрипцию, а пайплайн автоматически формирует синтетический `verbose_json` с таймингами на уровне чанков.

### Поддерживаемые серверы

| Сервер | API style | Как передаётся аудио | Endpoint |
|--------|-----------|----------------------|----------|
| **Ollama** | `ollama_native` (по умолчанию) | base64 WAV в поле `images` | `POST /api/chat` |
| **vLLM** / OpenAI-совместимый | `openai_chat` | `input_audio` content block | `POST /v1/chat/completions` |

### Подготовка (Ollama)

```bash
# Установить Ollama: https://ollama.com/download
# Скачать модель (≈3 GB для E4B):
ollama pull gemma4:e4b

# Проверить, что модель доступна:
ollama list | findstr gemma
```

Ollama по умолчанию слушает на `http://localhost:11434`.

### Запуск: моно-файлы (простейший случай)

Минимальная команда — транскрипция всех WAV из директории без coarse-сегментации и без VAD:

```bash
uv run python scripts/eval_test_audio.py \
  --stt-backend gemma \
  --audio-dir ./test_audio_mad \
  --coarse-backend whole_file \
  --no-vad \
  -v
```

При `--stt-backend gemma` скрипт автоматически выставляет значения по умолчанию:
- `--base-url` → `http://localhost:11434`
- `--model` → `gemma4:e4b`
- `--language` → `ru`

Любое из этих значений можно переопределить явно.

### Запуск: моно-файлы с coarse + VAD

```bash
uv run python scripts/eval_test_audio.py \
  --stt-backend gemma \
  --audio-dir ./test_audio_mad \
  --coarse-backend ina \
  -v
```

Здесь `ina` выполняет грубую сегментацию (речь / музыка / шум / тишина), затем Silero VAD уточняет границы речевых участков. Транскрибируются только речевые чанки.

### Запуск: стерео call-center

```bash
uv run python scripts/eval_test_audio.py \
  --stt-backend gemma \
  --stereo-call \
  --audio-dir ./test_audio_cc2 \
  --coarse-backend ina \
  --concurrency 2 \
  -v
```

Стерео WAV разделяется на два моно-канала (`call_from` / `call_to`), каждый проходит полный пайплайн независимо.

### Запуск: OpenAI-совместимый сервер (vLLM)

Если Gemma развёрнута на vLLM или другом сервере с OpenAI-совместимым chat API:

```bash
uv run python scripts/eval_test_audio.py \
  --stt-backend gemma \
  --gemma-api-style openai_chat \
  --base-url http://127.0.0.1:8000 \
  --model gemma-4-E4B-it \
  --audio-dir ./test_audio_mad \
  --coarse-backend whole_file \
  --no-vad \
  -v
```

### Полная справка по флагам Gemma

| Флаг | Значение по умолчанию | Описание |
|------|----------------------|----------|
| `--stt-backend gemma` | — | Включает Gemma chat-based ASR вместо Whisper. |
| `--gemma-api-style` | `ollama_native` | Стиль API: `ollama_native` (Ollama `/api/chat`) или `openai_chat` (vLLM `/v1/chat/completions`). |
| `--base-url` | `http://localhost:11434` (для gemma) | URL сервера. Для Ollama — `http://localhost:11434`, для vLLM — адрес сервера. |
| `--model` | `gemma4:e4b` (для gemma) | Название модели на сервере. Для Ollama: `gemma4:e4b`; для vLLM: как указано при `vllm serve`. |
| `--language` | `ru` (для gemma) | Код языка ISO 639-1. Встраивается в ASR-промпт для Gemma. |
| `--no-vad` | выкл | Пропустить Silero VAD — coarse-спаны напрямую идут в chunking. Полезно для отладки. |
| `--coarse-backend` | `ina` | `whole_file` — всё аудио считается речью; `ina` — отделяет речь от музыки/шума. |
| `--concurrency` | `3` | Параллельность обработки файлов. Для Ollama на одной GPU рекомендуется `1`–`2`. |
| `--api-key` | нет | API-ключ (если сервер требует авторизацию). |
| `-v` | выкл | DEBUG-логирование: видно каждый запрос к Gemma, промпт, ретраи. |

### Ограничения Gemma ASR

- **Лимит длины аудио**: Gemma поддерживает до ~30 секунд аудио на вход. Пайплайн нарезает чанки по 28 с (`max_chunk_duration_sec`), что укладывается в лимит.
- **Нет word-level timestamps**: Gemma возвращает plain text. `verbose_json` формируется синтетически — один сегмент на чанк с chunk-level таймингами (start/end).
- **Ollama 500**: Ollama иногда возвращает `500 Internal Server Error` на тяжёлых запросах ([ollama#15333](https://github.com/ollama/ollama/issues/15333)). Клиент автоматически ретраит с экспоненциальным бэкоффом.
- **Скорость**: На CPU Ollama обработка значительно медленнее, чем Whisper. Рекомендуется GPU.

### Пример полного конфига (программный вызов)

```python
from audio_asr_pipeline import PipelineConfig, AudioTranscriptionPipeline

config = PipelineConfig(
    coarse_segmenter_backend="ina",
    vad_backend="silero",          # или "none" для отключения VAD
    vllm={
        "stt_backend": "gemma",
        "base_url": "http://localhost:11434",
        "model": "gemma4:e4b",
        "language": "ru",
        "gemma_api_style": "ollama_native",  # или "openai_chat" для vLLM
        "gemma_max_tokens": 512,
        "request_timeout_sec": 120.0,
        "max_retries": 3,
    },
)

pipeline = AudioTranscriptionPipeline(config)
result = await pipeline.process_file("recording.wav")
print(result.text)
print(result.verbose_json)
```

## Прочее

Cursor skill для агентов: [.cursor/skills/audio-asr-pipeline/SKILL.md](.cursor/skills/audio-asr-pipeline/SKILL.md).
