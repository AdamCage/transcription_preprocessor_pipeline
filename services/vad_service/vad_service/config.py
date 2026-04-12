"""Service configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class ServiceConfig(BaseSettings):
    model_config = {"env_prefix": "VAD_", "env_file": ".env", "env_file_encoding": "utf-8"}

    device: str = "cuda:0"
    max_concurrency: int = 8
    executor_workers: int = 1
    host: str = "0.0.0.0"
    port: int = 8002
    log_level: str = "info"
    log_dir: str = "logs"
    model_path: str = ""
    max_audio_size_mb: float = 200.0
    inference_timeout_sec: float = 120.0


def get_config() -> ServiceConfig:
    return ServiceConfig()
