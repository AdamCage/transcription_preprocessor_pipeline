````markdown
Нужно реализовать универсальный Python-модуль для preprocessing и transcription аудио через vLLM Whisper/OpenAI-compatible STT API.

Цель
Создать переиспользуемый библиотечный модуль, который:
1. принимает один или несколько аудиофайлов;
2. удаляет из них неречевые участки:
   - тишину,
   - шум,
   - музыку без речи;
3. оставляет только speech-сегменты;
4. режет speech на удобные чанки;
5. параллельно отправляет чанки в vLLM Whisper endpoint;
6. собирает единый результат в формате, совместимом с `verbose_json`;
7. может импортироваться и использоваться в разных проектах без привязки к конкретному приложению.

Важно:
- это должен быть именно library-style модуль, не одноразовый скрипт;
- нужен чистый и расширяемый API;
- должна быть поддержка параллельной обработки нескольких файлов одновременно;
- должна быть поддержка параллельной отправки чанков одного файла в whisper/vLLM;
- модуль должен быть пригоден для production-like использования.

Требования к архитектуре

Нужно сделать отдельный пакет, например:
`audio_asr_pipeline/`

Предлагаемая структура:
- `audio_asr_pipeline/__init__.py`
- `audio_asr_pipeline/config.py`
- `audio_asr_pipeline/models.py`
- `audio_asr_pipeline/io.py`
- `audio_asr_pipeline/preprocess.py`
- `audio_asr_pipeline/segmenters.py`
- `audio_asr_pipeline/chunking.py`
- `audio_asr_pipeline/transcribe.py`
- `audio_asr_pipeline/merge.py`
- `audio_asr_pipeline/pipeline.py`
- `audio_asr_pipeline/utils.py`

Нужны четкие dataclass / pydantic-модели для внутренних сущностей.

Основные сущности
Нужно ввести как минимум такие модели:

1. `AudioFileTask`
- `source_path: Path`
- `file_id: str`
- `language: str | None`
- `metadata: dict[str, Any] | None`

2. `TimeSpan`
- `start: float`
- `end: float`

3. `LabeledSegment`
- `start: float`
- `end: float`
- `label: Literal["speech", "music", "noise", "silence", "unknown"]`
- `score: float | None = None`

4. `AudioChunk`
- `chunk_id: str`
- `file_id: str`
- `start: float`
- `end: float`
- `audio_bytes: bytes | None`
- `sample_rate: int`
- `num_samples: int`
- `source_path: Path | None`

5. `TranscribedChunk`
- `chunk_id: str`
- `file_id: str`
- `start_offset: float`
- `end_offset: float`
- `response: dict[str, Any]`

6. `PipelineResult`
- `file_id: str`
- `source_path: Path`
- `text: str`
- `verbose_json: dict[str, Any]`
- `stats: dict[str, Any]`

Функциональные требования

1. Загрузка и нормализация аудио
Нужно:
- читать wav/mp3/flac/m4a/ogg/webm через ffmpeg/librosa/soundfile/pydub;
- приводить аудио к mono;
- приводить sample rate к целевому значению, по умолчанию `16000`;
- хранить соответствие между processed audio timeline и original timeline;
- корректно работать с длинными файлами.

2. Speech vs music/noise предварительная сегментация
Нужно реализовать coarse segmentation:
- определить участки `speech`, `music`, `noise`, `silence/other`;
- по умолчанию использовать `inaSpeechSegmenter` как coarse speech/music/noise classifier;
- предусмотреть интерфейс-абстракцию, чтобы потом можно было подменить backend;
- все сегментаторы должны реализовывать единый protocol/interface, например:
  `BaseSegmenter.segment(path_or_audio) -> list[LabeledSegment]`.

Требование:
- в downstream-пайплайн передавать только участки, классифицированные как `speech`;
- `music`, `noise`, `silence` вырезать;
- участки с речью поверх музыки не отбрасывать, если backend классифицирует их как speech.

3. VAD для уточнения speech-границ
После coarse segmentation нужно прогонять speech-зоны через VAD:
- по умолчанию использовать `Silero VAD`;
- VAD работает только внутри coarse speech segments;
- нужно подрезать лишние хвосты тишины по краям;
- нужно уметь объединять соседние speech spans, если gap между ними меньше заданного порога.

Нужны настраиваемые параметры:
- `vad_threshold`
- `min_speech_duration_ms`
- `min_silence_duration_ms`
- `speech_pad_ms`
- `merge_gap_seconds`

4. Правила фильтрации
Нужно реализовать конфигурируемые правила:
- удалять сегменты короче минимальной длительности;
- удалять полностью music/noise-only сегменты;
- удалять сегменты с очень низкой энергией;
- опционально удалять слишком длинные неречевые промежутки;
- не отправлять пустые или почти пустые чанки в whisper.

5. Чанкинг
Нужно реализовать chunk builder:
- строить чанки только из очищенных speech segments;
- не склеивать далеко расположенные куски аудио в один физический chunk;
- каждый chunk должен соответствовать непрерывному участку исходного аудио;
- max duration chunk по умолчанию 20–30 секунд;
- если speech segment длиннее max duration — аккуратно резать на несколько subchunks;
- chunking должен сохранять возможность простого восстановления глобальных timestamps через offset.

6. Интеграция с vLLM Whisper
Нужно реализовать клиент OpenAI-compatible STT endpoint:
- POST `/v1/audio/transcriptions`
- multipart/form-data
- обязательные поля:
  - `model`
  - `file`
  - `response_format=verbose_json`
- опциональные поля:
  - `language`
  - `temperature`
  - `prompt`
  - `timestamp_granularities[]=segment`
  - опционально `timestamp_granularities[]=word`

Важно:
- по умолчанию использовать только `segment`, чтобы не плодить latency;
- `word` timestamps включать отдельным флагом;
- сделать отдельный клиентский класс, например `VLLMTranscriptionClient`;
- использовать `httpx.AsyncClient`;
- должна быть retry-логика;
- должны быть configurable timeout / concurrency limit;
- должен быть корректный handling 4xx/5xx/network errors.

7. Параллельность
Это ключевое требование.

Нужно поддержать два уровня параллельности:

A. file-level parallelism
- одновременно обрабатывать несколько входных файлов;

B. chunk-level parallelism
- внутри одного файла параллельно отправлять несколько чанков в whisper.

Нужен единый async orchestration слой:
- `asyncio`
- `Semaphore`/лимиты на:
  - количество одновременно обрабатываемых файлов,
  - количество одновременно отправляемых transcription requests.

Нужны параметры:
- `max_concurrent_files`
- `max_concurrent_chunks`
- `max_in_flight_requests`

Важно:
- не допускать бесконтрольного распараллеливания;
- не загружать память всеми файлами сразу;
- обрабатывать backpressure;
- предусмотреть bounded queue.

8. Сборка итогового verbose_json
После получения ответов по chunk’ам нужно:
- отсортировать chunks по original timeline;
- сместить все `segment.start`, `segment.end`, а также `word.start`, `word.end` на глобальный offset;
- склеить `text` по chunk order;
- собрать единый JSON-результат, совместимый по структуре с `verbose_json`.

Итоговый объект должен содержать:
- `text`
- `language`
- `duration`
- `segments`
- `words` (если были включены)
- служебную секцию `pipeline_meta`, например:
  - количество вырезанных music segments,
  - количество вырезанных noise/silence spans,
  - число speech chunks,
  - total original duration,
  - total kept speech duration,
  - percent dropped,
  - backend names and versions if доступны.

Важно:
- если какие-то поля отсутствуют в ответах vLLM, сборка не должна падать;
- агрегатор должен быть tolerant to missing optional fields.

9. Универсальный публичный API
Нужен удобный внешний API.

Сделать:
1. Высокоуровневый orchestrator:
```python
pipeline = AudioTranscriptionPipeline(config)
result = await pipeline.process_file(path)
results = await pipeline.process_files(paths)
````

2. Отдельную sync-обертку для простых проектов:

```python
result = process_file_sync(path, config)
results = process_files_sync(paths, config)
```

3. Возможность отдельно использовать стадии:

* `segment_audio(...)`
* `refine_speech_with_vad(...)`
* `build_chunks(...)`
* `transcribe_chunks(...)`
* `merge_transcriptions(...)`

Требование:

* любой этап должен быть переиспользуем отдельно;
* модуль нельзя проектировать как монолит только под один use case.

10. Конфигурация
    Нужна единая конфигурационная модель, например `PipelineConfig`, включающая:

General:

* `target_sample_rate`
* `mono`
* `temp_dir`
* `keep_intermediate_files`
* `save_cleaned_chunks`

Segmentation:

* `coarse_segmenter_backend`
* `drop_music`
* `drop_noise`
* `drop_silence`
* `min_segment_duration_sec`

VAD:

* `vad_backend`
* `vad_threshold`
* `min_speech_duration_ms`
* `min_silence_duration_ms`
* `speech_pad_ms`
* `merge_gap_seconds`

Chunking:

* `max_chunk_duration_sec`
* `min_chunk_duration_sec`

vLLM / Whisper:

* `base_url`
* `model`
* `language`
* `temperature`
* `prompt`
* `include_word_timestamps`
* `request_timeout_sec`
* `max_retries`
* `retry_backoff_sec`

Concurrency:

* `max_concurrent_files`
* `max_concurrent_chunks`
* `max_in_flight_requests`

11. Логирование и наблюдаемость
    Нужно:

* structured logging;
* логировать этапы:

  * loading,
  * coarse segmentation,
  * VAD refinement,
  * chunking,
  * transcription,
  * merge;
* логировать длительность каждого этапа;
* логировать причины drop сегментов;
* логировать ошибки по chunk/file без потери контекста.

Желательно:

* вернуть stats в финальном результате;
* предусмотреть hooks/callbacks для интеграции с внешними метриками.

12. Ошибки и устойчивость
    Нужно:

* если один chunk не транскрибировался, это не должно валить весь batch файлов;
* должна быть configurable политика:

  * `fail_fast=False/True`
  * `skip_failed_chunks=True/False`
* ошибки должны быть типизированы:

  * `AudioLoadError`
  * `SegmentationError`
  * `VADProcessingError`
  * `TranscriptionRequestError`
  * `MergeError`

13. Сохранение результатов
    Нужно поддержать:

* возврат результата в памяти;
* опциональное сохранение:

  * итогового `verbose_json`
  * plain text
  * промежуточных cleaned chunks
  * sidecar metadata json

Нужен helper:

```python
save_result(result, output_dir)
```

14. Тестируемость
    Нужно покрыть:

* unit tests для merge offsets;
* unit tests для chunk ordering;
* unit tests для drop/filter rules;
* unit tests для retries/transcription client;
* integration tests на mocked vLLM endpoint;
* smoke test на 2–3 реальных коротких аудиофайлах.

15. Ограничения и требования к реализации

* Python 3.11+
* typing обязателен
* dataclasses или pydantic models
* async-first design
* без жесткой привязки к FastAPI, CLI или конкретному приложению
* CLI можно добавить как бонус, но библиотечный API важнее
* код должен быть чистым, modular, production-friendly
* без лишней магии и глобального состояния
* модели/клиенты должны инициализироваться явно
* heavy backends не должны загружаться повторно на каждый chunk

16. Предпочтительные библиотеки
    Можно использовать:

* `httpx`
* `asyncio`
* `numpy`
* `soundfile`
* `librosa`
* `ffmpeg` / `ffmpeg-python` / `pydub`
* `silero-vad`
* `inaSpeechSegmenter`

Но нужно:

* инкапсулировать конкретные backends за интерфейсами;
* сделать так, чтобы в будущем можно было заменить `inaSpeechSegmenter` на другой music/speech classifier.

17. Что важно не делать

* не делать subprocess-вызов whisper на каждый chunk;
* не делать код, жестко завязанный на один проект;
* не делать chunk timestamps в локальной шкале без восстановления global offsets;
* не склеивать удаленные участки аудио в один fake continuous audio file;
* не захламлять GPU preprocessing-задачами по умолчанию;
* не писать giant god-class на весь pipeline.

18. Что нужно выдать в результате
    Нужно сгенерировать:
19. полный код пакета;
20. пример использования как библиотеки;
21. минимальный integration example;
22. README с описанием API;
23. requirements/pyproject;
24. набор тестов;
25. пример конфигурации.

Дополнительное архитектурное решение
Сделать backends так:

* `BaseSpeechMusicSegmenter`
* `InaSpeechMusicSegmenter`
* `BaseVADBackend`
* `SileroVADBackend`
* `BaseTranscriptionClient`
* `VLLMTranscriptionClient`

Ожидаемое поведение по умолчанию

* coarse speech/music/noise segmentation: CPU
* VAD: CPU
* transcription via vLLM Whisper: GPU server-side
* parallelism:

  * несколько файлов одновременно,
  * несколько chunks одновременно,
  * лимиты через semaphores.

Критерии приемки

1. Можно передать список из нескольких аудиофайлов и получить список итоговых результатов.
2. Из аудио удаляются music/noise/silence участки, а в whisper уходят только очищенные speech chunks.
3. Итог содержит единый корректный `verbose_json`.
4. Все timestamps восстановлены в глобальной шкале исходного файла.
5. Есть управление параллельностью на уровне файлов и чанков.
6. Код можно импортировать из другого проекта без доработки.
7. Нет жесткой привязки к CLI/FastAPI.
8. Есть тесты на offset merge и concurrency behavior.

Если где-то нужно выбрать между простотой и абстрактностью:

* сначала сделать надежный универсальный library API;
* потом уже optional CLI/adapters.

```

Если резать по делу, я бы еще ужесточил ТЗ одной фразой: **по умолчанию preprocessing не должен занимать GPU вообще**. Silero это спокойно переживает на CPU, а `inaSpeechSegmenter` тоже логичнее держать отдельно от GPU Whisper-контура. :contentReference[oaicite:3]{index=3}

Могу сразу следующим сообщением дать сокращенную версию этого ТЗ, уже в виде компактного prompt-а для Cursor без лишней воды.
::contentReference[oaicite:4]{index=4}
```

[1]: https://github.com/snakers4/silero-vad "GitHub - snakers4/silero-vad: Silero VAD: pre-trained enterprise-grade Voice Activity Detector · GitHub"
[2]: https://github.com/ina-foss/inaSpeechSegmenter?utm_source=chatgpt.com "ina-foss/inaSpeechSegmenter"
[3]: https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/speech_to_text/protocol/ "protocol - vLLM"
