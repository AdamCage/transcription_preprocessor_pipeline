"""
Mono ASR DAG: one mapped task per WAV (no stereo split).

Requires ``audio_asr_pipeline`` on workers; helpers under ``plugins/asr_helpers``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.python import get_current_context

from asr_helpers.config import load_pipeline_config
from asr_helpers.persistence import pipeline_result_to_dict, save_asr_result
from audio_asr_pipeline.pipeline import process_file_sync

log = logging.getLogger(__name__)


@dag(
    dag_id="asr_mono",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["asr", "mono", "audio_asr_pipeline"],
    doc_md=__doc__,
)
def dag_asr_mono():
    @task(task_id="list_mono_wav_paths")
    def list_mono_wav_paths() -> list[str]:
        raw = Variable.get("asr_mono_wav_paths_json", default_var="[]")
        paths = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(paths, list):
            log.warning("asr_mono_wav_paths_json must be a JSON list, got %s", type(paths))
            return []
        return [str(p) for p in paths]

    @task(task_id="transcribe_mono_and_persist")
    def transcribe_mono_and_persist(wav_path: str) -> dict:
        ctx = get_current_context()
        run_id = str(ctx["dag_run"].run_id)
        cfg = load_pipeline_config()
        result = process_file_sync(wav_path, cfg)
        payload = pipeline_result_to_dict(result)
        save_asr_result(
            dag_run_id=run_id,
            source_file=wav_path,
            channel="mono",
            result_dict=payload,
        )
        return {"source": wav_path, "error": result.error}

    paths = list_mono_wav_paths()
    transcribe_mono_and_persist.expand(wav_path=paths)


dag_asr_mono()
