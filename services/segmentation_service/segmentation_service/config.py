"""Service configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class ServiceConfig(BaseSettings):
    model_config = {"env_prefix": "SEG_", "env_file": ".env", "env_file_encoding": "utf-8"}

    model_name: str = "pyannote/voice-activity-detection"
    hf_token: str = ""
    device: str = "cuda:0"
    max_concurrency: int = 4
    executor_workers: int = 4
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "info"


def get_config() -> ServiceConfig:
    return ServiceConfig()
