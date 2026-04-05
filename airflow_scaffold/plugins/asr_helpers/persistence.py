"""Serialize PipelineResult and optional Postgres persistence via PostgresHook."""

from __future__ import annotations

import json
import logging
from typing import Any

from audio_asr_pipeline.models import PipelineResult

log = logging.getLogger(__name__)


def pipeline_result_to_dict(r: PipelineResult) -> dict[str, Any]:
    """JSON-/XCom-friendly dict (paths as strings)."""
    return {
        "file_id": r.file_id,
        "source_path": str(r.source_path),
        "text": r.text,
        "verbose_json": r.verbose_json,
        "stats": r.stats,
        "error": r.error,
    }


def save_asr_result(
    *,
    dag_run_id: str,
    source_file: str,
    channel: str | None,
    result_dict: dict[str, Any],
    postgres_conn_id: str = "asr_results_db",
) -> None:
    """
    Insert one row into ``asr_transcription_results``.
    If ``apache-airflow-providers-postgres`` is missing, logs the payload and returns.

    ``channel``: ``call_from``, ``call_to``, or ``mono``.
    """
    text = result_dict.get("text") or ""
    verbose = result_dict.get("verbose_json") or {}
    stats = result_dict.get("stats") or {}
    err = result_dict.get("error")

    try:
        from airflow.providers.postgres.hooks.postgres import PostgresHook
    except ImportError:
        log.warning(
            "PostgresHook unavailable; skipping DB insert | run_id=%s file=%s channel=%s err=%s",
            dag_run_id,
            source_file,
            channel,
            err,
        )
        log.debug("asr payload verbose_json keys: %s", list(verbose.keys()) if verbose else [])
        return

    hook = PostgresHook(postgres_conn_id=postgres_conn_id)
    sql = """
        INSERT INTO asr_transcription_results
            (dag_run_id, source_file, channel, text, verbose_json, stats, error)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
    """
    hook.run(
        sql,
        parameters=(
            dag_run_id,
            source_file,
            channel,
            text,
            json.dumps(verbose),
            json.dumps(stats),
            err,
        ),
    )
