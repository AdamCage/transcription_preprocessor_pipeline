"""
Stereo call-center ASR DAG: one mapped task per stereo WAV.

Channel 0 = call_from, channel 1 = call_to (same convention as eval_test_audio --stereo-call).
Requires ``audio_asr_pipeline`` installed on workers; helpers under ``plugins/asr_helpers``.
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
from asr_helpers.stereo import run_stereo_file_sync

log = logging.getLogger(__name__)


@dag(
    dag_id="asr_stereo_callcenter",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["asr", "stereo", "audio_asr_pipeline"],
    doc_md=__doc__,
)
def dag_asr_stereo_callcenter():
    @task(task_id="list_stereo_wav_paths")
    def list_stereo_wav_paths() -> list[str]:
        raw = Variable.get("asr_wav_paths_json", default_var="[]")
        paths = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(paths, list):
            log.warning("asr_wav_paths_json must be a JSON list, got %s", type(paths))
            return []
        return [str(p) for p in paths]

    @task(task_id="transcribe_stereo_and_persist")
    def transcribe_stereo_and_persist(stereo_path: str) -> dict:
        ctx = get_current_context()
        run_id = str(ctx["dag_run"].run_id)
        cfg = load_pipeline_config()
        work_dir = Variable.get("asr_stereo_work_dir", default_var="/tmp/asr_stereo_work")

        res_from, res_to = run_stereo_file_sync(stereo_path, work_dir, cfg)

        d_from = pipeline_result_to_dict(res_from)
        d_to = pipeline_result_to_dict(res_to)

        save_asr_result(
            dag_run_id=run_id,
            source_file=stereo_path,
            channel="call_from",
            result_dict=d_from,
        )
        save_asr_result(
            dag_run_id=run_id,
            source_file=stereo_path,
            channel="call_to",
            result_dict=d_to,
        )

        return {
            "source": stereo_path,
            "call_from_error": res_from.error,
            "call_to_error": res_to.error,
        }

    paths = list_stereo_wav_paths()
    transcribe_stereo_and_persist.expand(stereo_path=paths)


dag_asr_stereo_callcenter()
