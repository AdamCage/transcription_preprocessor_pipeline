"""Service configuration via environment variables."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings


class ServiceConfig(BaseSettings):
    model_config = {"env_prefix": "SEG_", "env_file": ".env", "env_file_encoding": "utf-8"}

    model_name: str = "pyannote/segmentation-3.0"
    model_path: str = ""
    hf_token: str = ""
    device: str = "cuda:0"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    max_concurrency: int = 4
    inference_timeout_sec: float = 300.0
    max_audio_bytes: int = 200 * 1024 * 1024  # 200 MB
    max_audio_duration_sec: float = 3600.0
    min_duration_on: float = 0.0
    min_duration_off: float = 0.0
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "info"
    log_dir: str = "logs"
    log_retention_days: int = 30


def get_config() -> ServiceConfig:
    return ServiceConfig()
