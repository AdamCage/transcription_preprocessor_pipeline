"""Build PipelineConfig from Airflow Variable / HTTP Connection."""

from __future__ import annotations

from urllib.parse import urlunparse

from airflow.hooks.base import BaseHook
from airflow.models import Variable

from audio_asr_pipeline.config import CoarseBackendName, PipelineConfig, VLLMTranscribeConfig


def _base_url_from_connection(conn_id: str) -> str | None:
    try:
        c = BaseHook.get_connection(conn_id)
    except Exception:
        return None
    scheme = (c.schema or "http").split("://")[0] if c.schema else "http"
    host = c.host or ""
    if not host:
        return None
    if c.port:
        netloc = f"{host}:{c.port}"
    else:
        netloc = host
    path = (c.extra_dejson.get("path") if c.extra_dejson else None) or ""
    if path and not path.startswith("/"):
        path = "/" + path
    return urlunparse((scheme, netloc, path.rstrip("/") or "", "", "", ""))


def load_pipeline_config(
    *,
    stt_conn_id: str = "asr_stt",
    stt_base_url_var: str = "asr_stt_base_url",
    stt_base_url_default: str = "http://127.0.0.1:8000",
    coarse_backend_var: str = "asr_coarse_backend",
    coarse_backend_default: CoarseBackendName = "whole_file",
    max_concurrent_files_var: str = "asr_max_concurrent_files",
) -> PipelineConfig:
    """
    Prefer HTTP Connection ``stt_conn_id`` when resolvable; else Variable ``stt_base_url_var``.
    Optional Variable ``coarse_backend_var``: ``whole_file`` or ``ina``.
    """
    base = _base_url_from_connection(stt_conn_id)
    if not base:
        base = Variable.get(stt_base_url_var, default_var=stt_base_url_default)

    coarse_raw = Variable.get(coarse_backend_var, default_var=coarse_backend_default)
    coarse: CoarseBackendName = coarse_raw if coarse_raw in ("whole_file", "ina") else "whole_file"

    max_cf_raw = Variable.get(max_concurrent_files_var, default_var="2")
    try:
        max_cf = max(1, int(max_cf_raw))
    except (TypeError, ValueError):
        max_cf = 2

    return PipelineConfig(
        coarse_segmenter_backend=coarse,
        fail_fast=False,
        max_concurrent_files=max_cf,
        max_in_flight_requests=max(4, max_cf * 2),
        vllm=VLLMTranscribeConfig(base_url=base),
    )
