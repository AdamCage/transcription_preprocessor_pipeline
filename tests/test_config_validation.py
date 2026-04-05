import pytest

from audio_asr_pipeline.config import PipelineConfig, VLLMTranscribeConfig


def test_vllm_empty_base_url_rejected() -> None:
    with pytest.raises(ValueError, match="base_url"):
        VLLMTranscribeConfig(base_url="  ")


def test_vllm_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="request_timeout_sec"):
        VLLMTranscribeConfig(request_timeout_sec=0)


def test_pipeline_chunk_max_le_min() -> None:
    with pytest.raises(ValueError, match="max_chunk_duration_sec"):
        PipelineConfig(max_chunk_duration_sec=0.5, min_chunk_duration_sec=1.0)


def test_pipeline_zero_concurrency_rejected() -> None:
    with pytest.raises(ValueError, match="max_concurrent_files"):
        PipelineConfig(max_concurrent_files=0)
