"""Service configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class ServiceConfig(BaseSettings):
    model_config = {"env_prefix": "VAD_"}

    device: str = "cuda:0"
    max_concurrency: int = 8
    executor_workers: int = 4
    host: str = "0.0.0.0"
    port: int = 8002
    log_level: str = "info"


def get_config() -> ServiceConfig:
    return ServiceConfig()
