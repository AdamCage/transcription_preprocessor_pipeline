# Airflow scaffold: ASR stereo + mono DAGs

Пример развёртывания оркестрации поверх пакета **`audio_asr_pipeline`** (тот же поток, что и [`scripts/eval_test_audio.py`](../scripts/eval_test_audio.py)): стерео — разделение каналов `call_from` / `call_to`, два прогона пайплайна, два `verbose_json`; моно — один файл, один прогон.

## Принцип: библиотека в образе, не в `plugins`

Соберите образ (или venv воркера) так, чтобы **`audio_asr_pipeline` был установлен как пакет** из этого репозитория:

```dockerfile
COPY . /opt/audio-asr-pipeline
RUN pip install /opt/audio-asr-pipeline
# или: pip install "audio-asr-pipeline==..." с вашего индекса
# optional coarse ina:
# RUN pip install "/opt/audio-asr-pipeline[ina]"
```

Каталог **`plugins/asr_helpers/`** — только тонкий слой: конфиг Airflow, разбивка стерео, запись в БД. Ядро ASR не копируется сюда.

## Подключение к Airflow

1. Смонтируйте **`airflow_scaffold/dags`** в `AIRFLOW__CORE__DAGS_FOLDER` (или в подкаталог, который сканируется).
2. Смонтируйте **`airflow_scaffold/plugins`** в `AIRFLOW__CORE__PLUGINS_FOLDER`.

У воркера и scheduler должен быть **один и тот же** Python-окружение с `audio_asr_pipeline` и зависимостями из [pyproject.toml](../pyproject.toml) (torch, librosa, httpx, …). Учитывайте **RAM** (модели VAD / при `ina` — TensorFlow).

## Переменные и соединения (пример)

| Имя | Назначение |
|-----|------------|
| Variable `asr_stt_base_url` | URL OpenAI-совместимого STT (например `http://vllm-whisper:8000`). |
| Connection `asr_stt` (опционально, тип HTTP) | Если задан, базовый URL собирается из соединения (`host`, `port`, схема); иначе используется переменная выше. |
| Variable `asr_stt_api_key` | API-ключ для STT (Authorization: Bearer). Пусто / не задан = без авторизации. |
| Variable `asr_stt_backend` | `openai` (рекомендуется для vLLM) или `httpx` (raw multipart POST). По умолчанию `openai`. |
| Variable `asr_wav_paths_json` | JSON-массив путей к **стерео** WAV (DAG `asr_stereo_callcenter`). |
| Variable `asr_mono_wav_paths_json` | JSON-массив путей к **моно** WAV (DAG `asr_mono`). В проде список можно отдавать из XCom/S3 вместо Variable. |
| Variable `asr_stereo_work_dir` | Каталог для временных mono WAV при стерео (должен быть доступен воркеру). По умолчанию `/tmp/asr_stereo_work`. |
| Connection `asr_results_db` (Postgres) | Целевая БД для `save_asr_result`. Нужен пакет `apache-airflow-providers-postgres`. |

## Таблица для примера `save_asr_result`

Создайте таблицу (имя может отличаться — поправьте SQL в [`plugins/asr_helpers/persistence.py`](plugins/asr_helpers/persistence.py)):

```sql
CREATE TABLE IF NOT EXISTS asr_transcription_results (
    id BIGSERIAL PRIMARY KEY,
    dag_run_id TEXT NOT NULL,
    source_file TEXT NOT NULL,
    channel TEXT,
    text TEXT,
    verbose_json JSONB,
    stats JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_asr_results_run ON asr_transcription_results (dag_run_id);
```

## DAGs

- `dag_asr_stereo_callcenter.py` — dynamic task mapping: один стерео WAV → split → два канала → STT → две строки в БД.
- `dag_asr_mono.py` — один mono WAV → STT → одна строка в БД.

Ограничьте параллелизм пула Airflow и/или `max_concurrent_files` / `max_in_flight_requests` в конфиге, чтобы не перегрузить STT.

## Документация

Общие паттерны Airflow + XCom: [docs/AIRFLOW.md](../docs/AIRFLOW.md).
