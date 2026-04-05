# Интеграция с Apache Airflow

Этот документ описывает типичные паттерны вызова **audio_asr_pipeline** из DAG Airflow 2.x: конфигурация из Variables/Connections, батч файлов, TaskFlow (`@task`), динамическое маппирование задач и сериализация в XCom.

## Предпочтительный API

- В **async**-контексте (например, отдельный worker-процесс без уже запущенного event loop в том же потоке): используйте `await pipeline.process_file(...)` или `await pipeline.process_files(...)` и по завершении `await pipeline.aclose()` (или `async with AudioTranscriptionPipeline(cfg) as pipeline:`).
- В **синхронных** операторах Airflow (`PythonOperator`, синхронный `@task`): удобны функции **`process_file_sync`** / **`process_files_sync`**. Они безопасны, когда в текущем потоке уже есть running loop (например Jupyter или часть окружений): реализация переносит `asyncio.run` в **отдельный поток**, избегая `RuntimeError: asyncio.run() cannot be called from a running event loop`.

После **`aclose()`** или выхода из **`async with`** на экземпляре, который сам создал внутренний `ThreadPoolExecutor`, повторно использовать тот же объект пайплайна не следует — создайте новый `AudioTranscriptionPipeline`.

## Результат и ошибки per-file

`PipelineResult` (`audio_asr_pipeline.models`) содержит поля транскрипции и опционально **`error: str | None`**. При **`fail_fast=False`** (по умолчанию в типичной production-конфигурации) сбой одного файла в батче **не** роняет остальные: для проблемного пути возвращается `PipelineResult` с заполненным `error`. Для XCom удобно отдавать **словарь**, а не сырой pydantic-объект:

```python
def pipeline_result_to_xcom(r) -> dict:
    return {
        "file_id": r.file_id,
        "text": r.text,
        "stats": r.stats,
        "error": r.error,
    }
```

Пути в XCom храните как **`str`**, не как `pathlib.Path`.

## Пример: PythonOperator, батч путей

```python
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from audio_asr_pipeline.config import PipelineConfig, VLLMTranscribeConfig
from audio_asr_pipeline.pipeline import process_files_sync


def transcribe_batch(**context):
    # Из Connection / Variables — здесь упрощённо
    base_url = "http://stt-service:8000"
    wav_dir = Path("/data/wavs")
    paths = sorted(wav_dir.glob("*.wav"))

    cfg = PipelineConfig(
        coarse_segmenter_backend="whole_file",
        vllm=VLLMTranscribeConfig(base_url=base_url, model="whisper-1"),
        max_concurrent_files=2,
        fail_fast=False,
    )
    results = process_files_sync(paths, cfg)
    return [pipeline_result_to_xcom(r) for r in results]


with DAG(
    dag_id="asr_batch_example",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    PythonOperator(
        task_id="transcribe_wavs",
        python_callable=transcribe_batch,
    )
```

## Пример: TaskFlow (`@task`) и список из XCom

```python
from datetime import datetime

from airflow.decorators import dag, task
from airflow.models import Variable

from audio_asr_pipeline.config import PipelineConfig, VLLMTranscribeConfig
from audio_asr_pipeline.pipeline import process_file_sync


@dag(start_date=datetime(2025, 1, 1), schedule=None, catchup=False)
def asr_taskflow_example():
    @task
    def list_wav_paths() -> list[str]:
        return ["/data/a.wav", "/data/b.wav"]

    @task
    def transcribe_one(path_str: str) -> dict:
        base_url = Variable.get("stt_base_url", default_var="http://127.0.0.1:8000")
        cfg = PipelineConfig(
            coarse_segmenter_backend="whole_file",
            vllm=VLLMTranscribeConfig(base_url=base_url),
            fail_fast=False,
        )
        r = process_file_sync(path_str, cfg)
        return pipeline_result_to_xcom(r)

    paths = list_wav_paths()
    transcribe_one.expand(path_str=paths)


dag = asr_taskflow_example()
```

Шаблон Jinja в `base_url` выше иллюстративен; в коде оператора обычно читают `Variable.get` / `Connection.get_connection_from_secrets` в рантайме без Jinja внутри литерала Python.

## Dynamic task mapping

Список путей из S3 или предыдущей задачи передаётся в **`.expand(...)`** (как в примере выше). Каждый элемент должен быть сериализуемым (строка пути, не `Path`).

## `asyncio.run` в worker Airflow

В отдельном процессе воркера, где **нет** running loop в потоке оператора, допустимо:

```python
import asyncio
from audio_asr_pipeline.pipeline import AudioTranscriptionPipeline

def run_async():
    async def main():
        cfg = ...  # PipelineConfig
        async with AudioTranscriptionPipeline(cfg) as pipe:
            return await pipe.process_files(paths)

    return asyncio.run(main())
```

Если среда уже создаёт event loop в том же потоке, **не** вызывайте `asyncio.run` из него — используйте **`await`** в async-операторе или **`process_file_sync`** / поток с новым loop (как в библиотеке).

## Ограничения по ресурсам (см. production review)

- **`max_concurrent_files`** ограничивает параллелизм по файлам и размер пула CPU-prepare.
- Длинные файлы: пайплайн освобождает крупные буферы после нарезки чанков, но при очень длинном аудио и высоком параллелизме следует держать лимиты и мониторить RAM.
- Глобальный семафор **`max_in_flight_requests`** ограничивает одновременные POST к STT по **всем** файлам.

## Ссылки

- Готовый скaffold с двумя DAG (стерео call-center + моно), `plugins/asr_helpers` и примером записи в Postgres: [airflow_scaffold/README.md](../airflow_scaffold/README.md)
- Архитектура пакета: [ARCHITECTURE.md](ARCHITECTURE.md)
- Настройки `trust_env` и прокси: [README.md](../README.md)
