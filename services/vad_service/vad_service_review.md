# VAD Service — Deep Review

Ревью наброска `services/vad_service/` на соответствие контракту библиотеки `audio_asr_pipeline` (README.md, docs/ARCHITECTURE.md) и готовность к production-развёртыванию на GPU (L40, RTX 6000).

---

## 1. Структура и соответствие паттерну segmentation_service

| Аспект | segmentation_service | vad_service | Статус |
|--------|---------------------|-------------|--------|
| `app.py` — lifespan, endpoints | да | да | OK |
| `inference.py` — GPU-обёртка | `PyannoteSegmenter` | `SileroVADGPU` | OK |
| `models.py` — Pydantic-схемы | да | да | OK |
| `config.py` — `BaseSettings` с env prefix | `SEG_`, `env_file=".env"` | `VAD_`, **нет `env_file`** | Расхождение |
| `.env` / `.env.example` | есть | **отсутствуют** | Расхождение |
| `tests/` — conftest, models, inference, endpoints | да | да | OK |
| Dockerfile | нет | нет | Оба отсутствуют |

**Вывод:** общая архитектура корректно повторяет паттерн segmentation_service. Расхождения по конфигу — мелкие, но должны быть устранены для единообразия.

---

## 2. Совместимость API с RemoteVADClient

Проверка контракта `audio_asr_pipeline/remote_clients.py::RemoteVADClient` против `vad_service/app.py`.

### POST /refine

| Что отправляет клиент | Что ожидает сервер | Совпадение |
|-----------------------|-------------------|------------|
| multipart `audio` (WAV bytes) | `UploadFile` param `audio` | OK |
| form field `request` (JSON string) | `str = Form(...)` param `request` | OK |
| JSON: `spans`, `threshold`, `min_speech_duration_ms`, `min_silence_duration_ms`, `speech_pad_ms`, `merge_gap_seconds` | `RefineRequest` model с теми же полями | OK |
| Ожидает `{"spans": [...]}` в ответе | `RefineResponse(spans=[...])` | OK |
| Fallback: если `spans` пуст — возвращает оригинальные | Сервер: пустые `spans` → `[]`; клиент сам делает fallback | OK |

### GET /health

| Клиент | Сервер | Совпадение |
|--------|--------|------------|
| Не вызывается из `RemoteVADClient` (опционально через eval/мониторинг) | Возвращает `HealthResponse` | OK |

### Совместимость с PipelineConfig

```
vad_backend="remote"
vad_service_url="http://gpu-host:8002"
remote_request_timeout_sec=120.0
remote_connect_timeout_sec=10.0
```

Клиент корректно подключается, таймауты пробрасываются. Полная совместимость.

---

## 3. Положительные стороны

1. **Чистая модульность** — app / inference / models / config / tests разделены корректно.
2. **Pydantic-валидация** — `threshold` ограничен `[0, 1]`, `TimeSpanIn.start/end >= 0`, `min_*_ms >= 0`.
3. **Concurrency-паттерн** — `asyncio.Semaphore` + `ThreadPoolExecutor` — стандартный подход для блокирующего inference в async-фреймворке.
4. **CPU fallback** — `_resolve_device()` переключается на CPU, если CUDA недоступна.
5. **Fallback при отсутствии речи** — `refine()` возвращает исходные spans, если VAD ничего не нашёл.
6. **Тест-покрытие** — 3 модуля тестов (models, inference, endpoints), fake Silero через monkeypatch.
7. **Content-type валидация** — отклоняет заведомо не-аудио типы.
8. **Health endpoint** — отдаёт VRAM usage через `torch.cuda.memory_allocated`.

---

## 4. Проблемы и замечания

### 4.1 CRITICAL: Thread safety при GPU-инференсе

**Файл:** `inference.py`, `config.py`

`ThreadPoolExecutor(max_workers=4)` запускает до 4 потоков, одновременно вызывающих `self._model(...)` на одном и том же JIT-модуле на GPU.

Проблемы:
- PyTorch GPU inference **не thread-safe** по умолчанию — конкурентные `forward()` на одной модели приводят к data races на CUDA.
- Silero `get_speech_timestamps` внутренне вызывает `model.reset_states()` — параллельные вызовы корраптят разделяемое состояние модели.
- `asyncio.Semaphore(8)` ограничивает async-вход, но `executor_workers=4` создаёт реальный параллелизм на уровне потоков.

**Рекомендация:** `executor_workers` по умолчанию должен быть `1`, либо нужен пул из N реплик модели (по одной на воркер).

### 4.2 HIGH: torch.hub.load при старте — невоспроизводимость

**Файл:** `inference.py:60-67`

```python
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
    trust_repo=True,
)
```

- Скачивает модель с GitHub при каждом холодном старте.
- В air-gapped / restricted-network среде — сервис не поднимется.
- Версия модели не зафиксирована — `silero-vad` `master` может сломать совместимость.

**Рекомендация:** вендорить модель (скачать JIT файл на этапе сборки Docker-образа), загружать через `torch.jit.load(local_path)`.

### 4.3 HIGH: Нет лимита размера тела запроса

**Файл:** `app.py`

Нет middleware для ограничения размера загружаемого аудио. Файл на 2 ГБ приведёт к OOM.

**Рекомендация:** добавить `max_audio_size_mb` в конфиг + проверку `len(raw)` после `audio.read()`.

### 4.4 MEDIUM: Нет таймаута на inference

**Файл:** `app.py:95-105`

```python
async with sem:
    refined = await vad.refine_async(...)
```

Если модель зависнет (CUDA hang), поток заблокирован навсегда, семафор не освобождается.

**Рекомендация:** обернуть в `asyncio.wait_for(..., timeout=config.inference_timeout_sec)`.

### 4.5 MEDIUM: Нет warm-up после загрузки модели

**Файл:** `app.py` lifespan

Первый реальный запрос после старта будет медленным (JIT compilation, CUDA context init).

**Рекомендация:** после `vad.load()` выполнить dummy inference на коротком тензоре.

### 4.6 MEDIUM: Лишние зависимости в pyproject.toml

**Файл:** `pyproject.toml`

| Пакет | Используется | Замечание |
|-------|-------------|-----------|
| `torchaudio>=2.1.0` | **Нет** — аудио читается через `soundfile` | Лишняя зависимость (~600 МБ), увеличивает образ |
| `packaging>=23.0` | **Нет** — нигде не импортируется | Мёртвая зависимость |

### 4.7 MEDIUM: Отсутствие валидации sample rate

**Файл:** `inference.py:88`

Silero VAD ожидает 16 kHz. Если придёт 44.1 kHz аудио, результат будет некорректным без предупреждения.

**Рекомендация:** проверять `sr` после `sf.read()`, логировать warning или ресэмплировать.

### 4.8 LOW: Типизация модели

**Файл:** `inference.py:41-42`

```python
self._model: object | None = None
self._get_speech_timestamps: object | None = None
```

Теряется вся типовая информация. В segmentation_service та же проблема (`_pipeline: object | None`), но здесь можно уточнить до `torch.jit.ScriptModule | None` и `Callable`.

### 4.9 LOW: Последовательная обработка spans

**Файл:** `inference.py:95-113`

Цикл `for sp in spans:` обрабатывает каждый span последовательно. Для длинных файлов с десятками spans это замедляет ответ.

**Рекомендация:** при необходимости — батчевая обработка (конкатенация spans с маркерами).

### 4.10 LOW: shutdown без ожидания in-flight запросов

**Файл:** `inference.py:155`

```python
self._executor.shutdown(wait=False)
```

При остановке сервиса текущие inference-вызовы обрываются.

**Рекомендация:** `shutdown(wait=True)` с таймаутом, либо дождаться освобождения семафора в lifespan.

### 4.11 LOW: config.py — нет env_file

**Файл:** `config.py:9`

```python
model_config = {"env_prefix": "VAD_"}
```

В segmentation_service:
```python
model_config = {"env_prefix": "SEG_", "env_file": ".env", "env_file_encoding": "utf-8"}
```

Нет поддержки `.env`-файла, что неудобно для локальной разработки.

---

## 5. Соответствие ARCHITECTURE.md

| Требование из ARCHITECTURE.md | Реализация | Статус |
|-------------------------------|------------|--------|
| FastAPI + Silero VAD JIT on GPU (секция 8) | `SileroVADGPU`, JIT load, GPU device | OK |
| POST /refine (audio + spans) | Endpoint `/refine` с multipart | OK |
| GET /health | Endpoint `/health` | OK |
| Порт 8002 | Дефолт `VAD_PORT=8002` | OK |
| Config через `VAD_*` env vars | `ServiceConfig(env_prefix="VAD_")` | OK |
| Async HTTP-клиент `RemoteVADClient` | Полная совместимость контракта | OK |
| Диаграмма seq: POST /refine → refined TimeSpan[] | Поток соответствует | OK |

---

## 6. Соответствие README.md

| Требование из README.md | Реализация | Статус |
|-------------------------|------------|--------|
| `vad_backend="remote"`, `vad_service_url` | `RemoteVADClient` корректно вызывает | OK |
| CLI `--vad-backend remote --vad-url http://...:8002` | Пробрасывается в `PipelineConfig` | OK |
| `uvicorn vad_service.app:app --host 0.0.0.0 --port 8002` | app = `_build_app()` на уровне модуля | OK |

---

## 7. План рефакторинга под production GPU (L40 / RTX 6000)

### 7.1 Thread safety (приоритет 1)

- `executor_workers` по умолчанию → `1`
- Опционально: pool реплик модели (по одной на воркер). Silero ~2 МБ VRAM, на L40 (48 ГБ) можно держать 16+ реплик.
- Явный `model.reset_states()` перед каждым `refine()`.

### 7.2 Вендорирование модели (приоритет 1)

- Скачивать JIT-файл Silero при сборке Docker-образа (`RUN python -c "torch.hub.load(...)"` в кэш).
- Добавить `VAD_MODEL_PATH` в конфиг для загрузки из локального файла.
- Fallback на `torch.hub.load` если путь не указан.

### 7.3 Dockerfile (приоритет 1)

```dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY vad_service/ vad_service/
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=False, trust_repo=True)"
EXPOSE 8002
CMD ["uvicorn", "vad_service.app:app", "--host", "0.0.0.0", "--port", "8002"]
```

### 7.4 Защита и таймауты (приоритет 2)

- `max_audio_size_mb` в конфиге + проверка после `audio.read()`.
- `asyncio.wait_for(vad.refine_async(...), timeout=...)` в endpoint.
- Валидация sample rate (warn + resample если не 16 kHz).

### 7.5 Warm-up (приоритет 2)

- После `vad.load()` в lifespan — dummy inference на 1с тишины.
- Логировать время warm-up.

### 7.6 Observability (приоритет 2)

- `/metrics` endpoint (prometheus_client).
- Метрики: `vad_requests_total`, `vad_request_duration_seconds` (histogram), `vad_spans_input_total`, `vad_spans_output_total`, `vad_gpu_memory_bytes`.
- Structured JSON logging для production (uvicorn `--log-config`).

### 7.7 Graceful shutdown (приоритет 3)

- `executor.shutdown(wait=True)` с таймаутом.
- В lifespan yield: дождаться освобождения семафора перед shutdown.

### 7.8 Чистка зависимостей (приоритет 3)

- Убрать `torchaudio` (экономия ~600 МБ в образе).
- Убрать `packaging`.
- Добавить `prometheus-client` для метрик.

### 7.9 Config parity (приоритет 3)

- Добавить `env_file=".env"` в `model_config`.
- Создать `.env.example`.

---

## 8. Итог

Набросок VAD-сервиса качественный: API-контракт полностью совместим с библиотекой, структура единообразна с segmentation_service, тесты покрывают основные сценарии. Главный блокер для production — **thread safety при параллельном GPU-инференсе** (4.1). Остальные пункты — улучшения для надёжности и эксплуатации на L40/RTX 6000.
