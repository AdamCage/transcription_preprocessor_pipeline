---
name: ТЗ упрощённого ASR
overview: "Подготовить markdown-техническое задание на новую упрощённую библиотеку пайплайна распознавания (ввод файлов/папок/архива, segmentation_service, vad_service, vLLM STT только через SDK `openai` с `response_format: text`, сборка и склейка `verbose_json`, многопоточность до 128 потоков) и на переписывание `scripts/eval_test_audio.py` под отладку и метрики без лишней сложности."
todos:
  - id: draft-md-path
    content: Зафиксировать путь и имя файла ТЗ (docs/TZ_simplified_asr_pipeline.md) и оглавление
    status: completed
  - id: embed-service-contracts
    content: Включить в ТЗ точные контракты segmentation_service и vad_service (эндпоинты, поля)
    status: completed
  - id: stt-text-only
    content: "Описать контракт vLLM STT: SDK openai (синхронный), только response_format=text, сборка verbose_json вручную"
    status: completed
  - id: threading-model
    content: Описать модель ThreadPoolExecutor (1–128), уровень параллелизма файлы vs сегменты
    status: completed
  - id: eval-script-spec
    content: "Раздел ТЗ по упрощённому eval: CLI, метрики, выход только report.xlsx (openpyxl), WER/CER через библиотеку"
    status: completed
  - id: mermaid-flow
    content: "В ТЗ: Mermaid sequenceDiagram (пайплайн + сервисы + openai) и C4Component (субграфы/внешние системы); при необходимости дублирующий flowchart"
    status: completed
isProject: false
---

# ТЗ: упрощённая библиотека ASR и скрипт eval

## Цель документа

Создать один файл спецификации (например [`docs/TZ_simplified_asr_pipeline.md`](docs/TZ_simplified_asr_pipeline.md)), на основе которого можно реализовать новую библиотеку и упростить eval, **без** немедленной правки кода. Документ на русском, со ссылками на существующие контракты сервисов и на текущие проблемы кодовой базы. В состав ТЗ **обязательно** входят диаграммы **Mermaid**: **sequence** и **C4 (уровень компонентов)** — см. п. 7 структуры.

## Контекст из репозитория (зафиксировать в ТЗ как «наследие»)

- Текущий пакет [`audio_asr_pipeline/`](audio_asr_pipeline/): много модулей (`pipeline.py`, `preprocess.py`, `transcribe.py` с несколькими бэкендами, async-семафоры), STT для vLLM в [`audio_asr_pipeline/transcribe.py`](audio_asr_pipeline/transcribe.py) по умолчанию запрашивает **`verbose_json`** и гранулярности — это **не** соответствует новому требованию.
- Формат результата: [`audio_asr_pipeline/models.py`](audio_asr_pipeline/models.py) (`PipelineResult.verbose_json`), скелет/слияние — [`audio_asr_pipeline/merge.py`](audio_asr_pipeline/merge.py) (OpenAI-подобные поля `text`, `duration`, `segments`, `words`, `pipeline_meta`).
- Удалённые сервисы (контракты для ТЗ):
  - Segmentation: [`services/segmentation_service/README.md`](services/segmentation_service/README.md) — `POST /segment`, multipart `audio`, ответ `segments[]` + `duration_sec`.
  - VAD: [`services/vad_service/README.md`](services/vad_service/README.md) — `POST /refine`, `audio` + JSON `request` со `spans`, ответ `spans`.
- Клиенты-референс для HTTP-формата: [`audio_asr_pipeline/remote_clients.py`](audio_asr_pipeline/remote_clients.py) (`RemoteSegmentationClient`, `RemoteVADClient`).
- Стерео: [`audio_asr_pipeline/io.py`](audio_asr_pipeline/io.py) — `split_stereo_channels` (важно описать ось времени в ТЗ).
- Текущий eval: [`scripts/eval_test_audio.py`](scripts/eval_test_audio.py) — asyncio, тяжёлый XLSX, много режимов; в ТЗ задать **целевое** упрощённое поведение.

## Структура markdown-ТЗ (содержание файла)

1. **Введение и цели** — заменить перегруженную библиотеку; обеспечить работу с **реальными** vLLM STT в проде; явный отказ от лишних бэкендов и глубокого наследования в пользу плоских функций/небольших классов «композиция».

2. **Границы системы и термины** — «сегмент» (интервал времени + метка/роль), «speech-сегмент» для STT, `verbose_json` (совместимость с OpenAI-подобной схемой из `merge.py`), каналы стерео (`call_from` / `call_to` или L/R).

3. **Функциональные требования** (нумерация как у заказчика):
   - **Ввод**: один путь к файлу; каталог с рекурсивным или нерекурсивным обходом (уточнить в ТЗ флаг); **архив** — минимум ZIP с аудио внутри (форматы: wav/flac/mp3 и т.д. как у segmentation service); опционально в разделе «расширения» — tar.gz.
   - **Стерео**: опция разделения на два моно-потока; нормализация оси каналов (как в существующем `split_stereo_channels`).
   - **Сегментация**: режим `segmentation_service` — вызов `POST /segment`, маппинг лейблов (`speech` / `non_speech` → внутренние `speech`/`silence`/`noise` и т.д., по аналогии с `_map_remote_label`).
   - **VAD**: режим `vad_service` — подача **грубых** интервалов (например только `speech` из segmentation или whole-file) в `POST /refine`; параметры VAD из конфига; комбинация режимов в ТЗ описать явно: «только segmentation», «только VAD на whole file», «segmentation → VAD».
   - **Подготовка `verbose_json` до STT**: из финального списка интервалов речи построить структуру с таймкодами (сегменты разговора); сегменты **вне** речи — в `pipeline_meta` (удалённая тишина, нерелевантные по лейблу регионы) с `start`/`end`/`label`/`reason`.
   - **STT (строго vLLM, строго библиотека `openai`)**: только синхронный клиент **`OpenAI`** из пакета **`openai`** (не `AsyncOpenAI`, не сырой `httpx`/`requests` к `/v1/audio/transcriptions`). Вызов **`client.audio.transcriptions.create`** (или эквивалент в актуальной версии SDK) с **`response_format="text"`**, **`file`** — байты/файловый объект WAV сегмента; **`model`**, опционально **`language`**, **`temperature`**, **`prompt`**. Ответ SDK при `text` — **строка** (не JSON); запрет `verbose_json` / `json` в прод-режиме. В ТЗ описать использование в **`ThreadPoolExecutor`**: либо один общий `OpenAI` на процесс (если документация SDK допускает потокобезопасное переиспользование), либо **клиент на поток** / явная политика из документации `openai` — зафиксировать выбранный вариант при реализации. Ядро библиотеки без `asyncio` для STT.
   - **Сборка `verbose_json` после STT**: для каждого speech-интервала взять текст ответа и **вписать** в соответствующий элемент `segments[]` с полями `start`, `end`, `text`; общее поле `text` — конкатенация; `words` — пустой массив или опционально отсутствует (явно описать).
   - **Склейка стерео**: два канала → два `verbose_json` → один итоговый: упорядоченное объединение сегментов по времени с полем `channel` / `speaker`; в `pipeline_meta` — сводка по отброшенным регионам по каналам; единая длительность = длительность исходного файла.

4. **Нефункциональные требования**
   - Параллелизм: **`concurrent.futures.ThreadPoolExecutor`** + `as_completed`, параметр `max_workers` 1…128; распараллеливание на уровне **файлов** и/или **сегментов STT** (в ТЗ выбрать стратегию: например файлы в пуле, внутри файла сегменты последовательно **или** ограниченный под-пул для сегментов с отдельным семафором — зафиксировать одну простую схему, чтобы не плодить вложенность).
   - Таймауты, ретраи (429/503), лимит соединений — отдельный подраздел.
   - Логирование: структурированные события по этапам (load → segment → vad → stt → merge).
   - Зависимости: для STT — обязательно **`openai`** (синхронный API); для вызовов segmentation/VAD — `httpx` или `requests` (без требования совпадать с STT-транспортом); при необходимости `numpy`/`soundfile`; для **`scripts/eval_test_audio.py`** — обязательно **`openpyxl`** (выход только `.xlsx`); без async-обязаловки в ядре библиотеки; **не** тянуть `ina`/локальный pyannote, если выбран только remote segmentation.

5. **Публичный API библиотеки (черновик для реализации)**
   - Датакласс конфигурации (URL сервисов, URL vLLM, ключ, `max_workers`, флаги стерео, режим сегментации).
   - Функции верхнего уровня: `transcribe_path(...)`, `transcribe_directory(...)`, `transcribe_archive(...)` → список/итератор результатов с `verbose_json` и таймингами по этапам.
   - Отдельный модуль: `metrics.normalize_text_for_wer_cer` (lower + удаление пунктуации) и обёртки над `jiwer` — чтобы eval не дублировал логику.

6. **Спецификация переписанного [`scripts/eval_test_audio.py`](scripts/eval_test_audio.py)**
   - Короткий CLI: директория (и опционально один файл), `--workers`, флаги стерео, URL сервисов и vLLM.
   - Инициализация только через новую библиотеку.
   - Запуск обработки каталога; сбор расшифровок.
   - Если рядом эталонные `.txt` — WER/CER через функции библиотеки после нормализации (как сейчас `_normalize_for_metrics` + jiwer).
   - **Выход — строго `.xlsx`**: один файл отчёта (например `report.xlsx` в каталоге прогона или путь из `--output`), формирование через **`openpyxl`** (обязательная зависимость скрипта / группы `eval`): минимум **два листа** — детальный по каждому аудиофайлу (пути, гипотеза, эталон при наличии, WER/CER, тайминги по этапам, вспомогательные поля) и **сводный** (агрегаты: общее wall time, `max_workers`, распределение WER/CER — mean/median/p25/p75, среднее время на файл, статистика времени обработки vs длительности аудио, **скорость этапов**: секунды обработанного аудио на 1 с wall time по этапам segmentation, vad, stt, полный пайплайн — формулы как в текущем summary: `audio_duration / stage_wall_time`). JSON/CSV как основной вывод **не** предусматривать.

7. **Диаграммы Mermaid (обязательные)** — отдельный раздел или подразделы в `.md` ТЗ:
   - **`sequenceDiagram`**: минимум одна последовательность «один аудиофайл» — взаимодействие **библиотеки** с `segmentation_service` (`POST /segment`), при включённом режиме — с `vad_service` (`POST /refine`), затем с **vLLM** через SDK **`openai`** (транскрипция сегментов, `response_format=text`), затем **сборка/склейка** `verbose_json` (и при стерео — отдельная нота или вторая диаграмма: два канала → merge). Участники: вызывающий код / `ThreadPoolExecutor` при необходимости, HTTP-клиенты сервисов, `OpenAI`-клиент.
   - **C4, уровень компонентов**: диаграмма **`C4Component`** (или эквивалентная **`C4Container`**, если рендерер не поддерживает Component) в синтаксисе Mermaid — границы контейнера «приложение/процесс», внутри компоненты: **ядро библиотеки** (оркестрация, нарезка сегментов, сборка JSON), **клиент segmentation**, **клиент VAD**, **клиент STT (`openai`)**, при необходимости **утилиты ввода** (файл/каталог/ZIP); **внешние системы**: `segmentation_service`, `vad_service`, **vLLM STT**; опционально **`eval_test_audio`** как отдельный компонент потребитель. Связи подписать протоколом (HTTPS + multipart/JSON). Если целевой просмотрщик Markdown не рендерит C4 — в ТЗ продублировать **упрощённую `flowchart`/`graph`** с теми же узлами и границами (`subgraph`) как запасной вариант для читабельности, но **основной** артефакт — C4.

8. **Открытые вопросы / решения при реализации**
   - Имя пакета: замена `audio_asr_pipeline` vs новый namespace (v2).
   - Поддерживаемые форматы архивов кроме ZIP.
   - Нужны ли word-level таймкоды в будущем (сейчас — нет, т.к. `response_format=text`).

## Что не входит в этот шаг

- Реализация кода и правки существующих модулей — только подготовка `.md` ТЗ после утверждения плана.

## Ключевая ссылка для согласованности STT

В ТЗ явно указать: доступ к vLLM OpenAI-compatible эндпоинту транскрипций **только через SDK `openai`** (`OpenAI`, `base_url` на vLLM, при необходимости `api_key`), параметр **`response_format="text"`**, итог сегмента — **строка**; вся временная разметка — **только** из пайплайна сегментации/VAD, не из модели.
