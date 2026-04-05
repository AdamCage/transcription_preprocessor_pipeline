"""Pipeline configuration (Pydantic)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

CoarseBackendName = Literal["whole_file", "ina"]
VADBackendName = Literal["silero"]

# Shared default so eval and PipelineConfig stay aligned
DEFAULT_COARSE_SEGMENTER_BACKEND: CoarseBackendName = "ina"


class VLLMTranscribeConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000"
    model: str = "large-v3-turbo"
    language: str | None = None
    temperature: float | None = None
    prompt: str | None = None
    include_word_timestamps: bool = False
    request_timeout_sec: float = 120.0
    connect_timeout_sec: float = 10.0
    max_retries: int = 3
    retry_backoff_sec: float = 1.0
    retry_after_cap_sec: float = 60.0  # max sleep for Retry-After on 429/503
    # httpx defaults trust_env=True: HTTP(S)_PROXY can steal localhost traffic
    trust_env: bool = False

    @model_validator(mode="after")
    def _validate_vllm(self) -> Self:
        if not str(self.base_url).strip():
            raise ValueError("base_url must be non-empty")
        if self.request_timeout_sec <= 0:
            raise ValueError("request_timeout_sec must be positive")
        if self.connect_timeout_sec <= 0:
            raise ValueError("connect_timeout_sec must be positive")
        if self.retry_after_cap_sec <= 0:
            raise ValueError("retry_after_cap_sec must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        return self


class PipelineConfig(BaseModel):
    target_sample_rate: int = 16000
    mono: bool = True
    temp_dir: Path | None = None
    keep_intermediate_files: bool = False
    save_cleaned_chunks: bool = False

    coarse_segmenter_backend: CoarseBackendName = DEFAULT_COARSE_SEGMENTER_BACKEND
    # inaSpeechSegmenter uses TensorFlow; hide GPUs before TF import when True
    ina_force_cpu: bool = True
    drop_music: bool = True
    drop_noise: bool = True
    drop_silence: bool = True
    min_segment_duration_sec: float = 0.25

    vad_backend: VADBackendName = "silero"
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 200
    speech_pad_ms: int = 200
    merge_gap_seconds: float = 0.5

    max_chunk_duration_sec: float = 28.0
    min_chunk_duration_sec: float = 0.5

    vllm: VLLMTranscribeConfig = Field(default_factory=VLLMTranscribeConfig)

    max_concurrent_files: int = 3
    max_concurrent_chunks: int = 3
    max_in_flight_requests: int = 8

    fail_fast: bool = False
    skip_failed_chunks: bool = True

    @model_validator(mode="after")
    def _validate_pipeline(self) -> Self:
        if self.max_chunk_duration_sec <= self.min_chunk_duration_sec:
            raise ValueError(
                "max_chunk_duration_sec must be greater than min_chunk_duration_sec"
            )
        if self.max_concurrent_files <= 0:
            raise ValueError("max_concurrent_files must be positive")
        if self.max_concurrent_chunks <= 0:
            raise ValueError("max_concurrent_chunks must be positive")
        if self.max_in_flight_requests <= 0:
            raise ValueError("max_in_flight_requests must be positive")
        return self