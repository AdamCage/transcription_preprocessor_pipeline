# Архитектура audio-asr-pipeline

Документ описывает назначение системы, основные потоки данных и границы компонентов. Диаграммы в формате [Mermaid](https://mermaid.js.org/) (C4-стиль и последовательности).

## 1. Назначение

Библиотека **`audio_asr_pipeline`** загружает аудио, выполняет грубую сегментацию (речь / музыка / шум), уточняет границы речи через VAD (Silero, ONNX), режет сигнал на чанки и отправляет их в **OpenAI-совместимый** endpoint `/v1/audio/transcriptions` (например, vLLM + Whisper). Результаты объединяются в единый текст и расширенный **`verbose_json`**.

Скрипт **`scripts/eval_test_audio.py`** прогоняет каталог WAV, строит отчёт XLSX (WER/CER, тайминги, RT-метрики) и сохраняет `verbose_json` по файлам. Режим **`--stereo-call`** разделяет стерео на два моно-канала (call_from / call_to), поддерживая обе раскладки librosa `(n_samples, 2)` и `(2, n_samples)`.

## 2. C4: контекст системы

Внешние акторы и системы: оператор/интегратор, файловое хранилище WAV, STT-сервис, опционально ffmpeg (для ina), GPU/CPU для TF и PyTorch.

```mermaid
C4Context
title Уровень 1 — контекст (Audio ASR pipeline)

Person(operator, "Оператор / интегратор", "Запускает eval или встраивает библиотеку в свой код.")
System(pipeline, "Audio ASR Pipeline", "Препроцессинг аудио, чанкование, вызов STT, склейка результата.")
System_Ext(stt, "STT API", "OpenAI-compatible POST /v1/audio/transcriptions (vLLM, и т.д.).")
System_Ext(fs, "Файловая система", "Исходные WAV, опционально эталоны .txt.")

Rel(operator, pipeline, "Конфигурирует и запускает")
Rel(pipeline, fs, "Читает WAV, пишет отчёты / json")
Rel(pipeline, stt, "HTTPS: чанки как WAV multipart")
```

Если рендерер не поддерживает `C4Context`, используйте эквивалент ниже:

```mermaid
flowchart LR
    subgraph External["Внешнее"]
        STT["STT API\n/v1/audio/transcriptions"]
        WAV["WAV / эталоны .txt"]
    end
    OP["Оператор"] --> LIB["audio_asr_pipeline\n+ eval_test_audio"]
    LIB --> WAV
    LIB --> STT
```

## 3. C4: контейнеры

«Контейнеры» здесь — разворачиваемые/логические единицы: Python-пакет, CLI eval, процесс STT.

```mermaid
C4Container
title Уровень 2 — контейнеры

Person(user, "Пользователь", "")

Container_Boundary(app, "Репозиторий transcription_preprocessor_pipeline") {
    Container(lib, "audio_asr_pipeline", "Python package", "preprocess → chunk → transcribe → merge")
    Container(eval, "eval_test_audio.py", "CLI", "батч WAV, XLSX, stereo-call")
}

Container_Ext(stt, "STT сервис", "HTTP", "Whisper / совместимый API")

Rel(user, eval, "uv run python …")
Rel(user, lib, "import / process_file")
Rel(eval, lib, "AudioTranscriptionPipeline")
Rel(lib, stt, "httpx AsyncClient, WAV bytes")
```

Упрощённая блок-схема зависимостей:

```mermaid
flowchart TB
    subgraph cli["CLI"]
        E[eval_test_audio.py]
    end
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
    STT["STT HTTP API"]
    E --> P
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

## 4. C4: компоненты (внутри пакета)

```mermaid
C4Component
title Уровень 3 — компоненты библиотеки

Container_Boundary(api, "audio_asr_pipeline") {
    Component(pipeline, "AudioTranscriptionPipeline", "Python", "Семафор файлов, executor для CPU, оркестрация STT")
    Component(pre, "preprocess_audio", "Python", "load → coarse → VAD → TimeSpan[]")
    Component(seg, "segmenters", "Python", "inaSpeechSegmenter | whole_file")
    Component(vad, "vad", "Python", "Silero ONNX refine spans")
    Component(ch, "chunking", "Python", "spans → AudioChunk + WAV bytes")
    Component(tr, "VLLMTranscriptionClient", "Python", "multipart POST")
    Component(mg, "merge", "Python", "verbose_json, текст")

    Rel(pipeline, pre, "вызывает")
    Rel(pre, seg, "coarse labels")
    Rel(pre, vad, "refine")
    Rel(pipeline, ch, "после preprocess")
    Rel(pipeline, tr, "чанки")
    Rel(pipeline, mg, "ответы STT")
}
```

## 5. Диаграмма последовательности: один файл (mono)

Отражает `AudioTranscriptionPipeline.process_file`: общий семафор `_file_sem` ограничивает число файлов в полном цикле одновременно.

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Вызывающий код / eval
    participant Pipe as AudioTranscriptionPipeline
    participant Exec as Thread pool executor
    participant Pre as preprocess_audio
    participant Seg as segmenter (ina | whole_file)
    participant VAD as Silero VAD
    participant Chunk as chunking + io
    participant HTTP as httpx → STT
    participant Merge as merge_transcriptions

    Caller->>Pipe: process_file(path)
    Pipe->>Pipe: acquire _file_sem
    Pipe->>Exec: run_in_executor(_prepare_file_cpu)
    Exec->>Pre: preprocess_audio(path)
    Pre->>Pre: load_normalized_audio (librosa)
    Pre->>Seg: segment(path) coarse labels
    Seg-->>Pre: LabeledSegment[]
    Pre->>VAD: refine_speech_spans_with_silero
    VAD-->>Pre: refined TimeSpan[]
    Pre-->>Exec: loaded + spans + labeled
    Exec->>Chunk: build_chunks, extract_span_to_wav_bytes
    Chunk-->>Pipe: chunks[].audio_bytes
    loop По чанкам (семафор чанков)
        Pipe->>HTTP: transcribe_chunk WAV
        HTTP-->>Pipe: JSON сегментов / text
    end
    Pipe->>Merge: merge_transcriptions + skeleton
    Merge-->>Pipe: verbose_json
    Pipe-->>Caller: PipelineResult
    Pipe->>Pipe: release _file_sem
```

## 6. Диаграмма последовательности: eval stereo-call

Для каждого стерео WAV: загрузка с `mono=False`, `split_stereo_channels`, запись двух моно WAV, затем два параллельных `process_file` (два слота семафоров pipeline).

```mermaid
sequenceDiagram
    autonumber
    participant Eval as eval_test_audio _run_stereo
    participant Lib as librosa + split_stereo_channels
    participant Disk as stereo_mono_wav/*.wav
    participant P as pipeline × 2

    Eval->>Lib: load wav mono=False
    Lib-->>Eval: y (T,2) или (2,T)
    Eval->>Lib: split_stereo_channels
    Lib-->>Eval: ch0, ch1
    Eval->>Disk: write_mono_wav __call_from / __call_to
    par Канал call_from
        Eval->>P: process_file(p_from)
    and Канал call_to
        Eval->>P: process_file(p_to)
    end
    Eval->>Eval: строка XLSX + WER по каналам
```

## 7. Ключевые конфигурационные объекты

| Объект | Файл | Роль |
|--------|------|------|
| `PipelineConfig` | `config.py` | sample_rate, coarse backend, VAD, chunk limits, drop_music/noise/silence, concurrency caps |
| `VLLMTranscribeConfig` | `config.py` | base_url, model, timeouts, trust_env |
| `DEFAULT_COARSE_SEGMENTER_BACKEND` | `config.py` | согласован с дефолтом eval (`ina`) |

Важно: **`max_concurrent_files`** задаёт размер **общего** `asyncio.Semaphore` на экземпляр `AudioTranscriptionPipeline`, поэтому параллельно полностью обрабатывается не больше этого числа файлов (стерео занимает два слота, если каналы гоняются параллельно).

## 8. Ошибки и краевые случаи

| Исключение | Когда |
|------------|--------|
| `AudioLoadError` | Пустое/битое аудио, неверная стерео-форма для split |
| `SegmentationError` | Нет ina, сбой ina, нет speech spans после coarse |
| `TranscriptionRequestError` | Сеть/HTTP/4xx/5xx STT |
| `MergeError` | Несовместимость ответов при склейке (частично mitigated в pipeline) |

При **`fail_fast=False`** типичные сбои загрузки/сегментации/STT для одного пути не пробрасываются наружу из батча: вместо этого для этого файла возвращается **`PipelineResult`** с заполненным полем **`error`** (успех: `error is None`). Подробнее про оркестрацию из Airflow, XCom и синхронные обёртки: **[AIRFLOW.md](AIRFLOW.md)**.

Экземпляр **`AudioTranscriptionPipeline`** рекомендуется закрывать через **`async with`** или **`await aclose()`**: общий **`httpx.AsyncClient`** и (если не передан внешний) внутренний **`ThreadPoolExecutor`** освобождаются при выходе.

## 9. Зависимости (логически)

- **Звук:** librosa, soundfile, numpy  
- **VAD:** torch + Silero через **ONNX** (`onnxruntime`), см. `vad.py`  
- **Coarse ina:** optional extra `ina`, TensorFlow; на Windows в `pyproject` override для plain `tensorflow`  
- **STT:** httpx async, multipart WAV  

## 10. Где читать код

| Модуль | Ответственность |
|--------|-----------------|
| `pipeline.py` | Точка входа async, семафор файлов, `_prepare_file_cpu`, `_transcribe_all` |
| `preprocess.py` | Склейка coarse + VAD + фильтры span |
| `segmenters.py` | ina (в т.ч. female/male → speech), whole_file |
| `vad.py` | Silero ONNX, паддинг/мердж промежутков |
| `chunking.py` | Ограничение длины чанка, `AudioChunk` |
| `transcribe.py` | OpenAI-form POST |
| `merge.py` | Таймкоды, итоговый текст |
| `io.py` | load, WAV bytes, `split_stereo_channels`, `write_mono_wav` |

---

*Для навыков агента Cursor см. `.cursor/skills/audio-asr-pipeline/SKILL.md`.*
