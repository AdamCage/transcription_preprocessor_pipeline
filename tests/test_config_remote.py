"""Unit tests for PipelineConfig with remote backends."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from audio_asr_pipeline.config import PipelineConfig


class TestRemoteBackends:
    def test_remote_coarse_accepted(self):
        cfg = PipelineConfig(coarse_segmenter_backend="remote")
        assert cfg.coarse_segmenter_backend == "remote"

    def test_remote_vad_accepted(self):
        cfg = PipelineConfig(vad_backend="remote")
        assert cfg.vad_backend == "remote"

    def test_both_remote(self):
        cfg = PipelineConfig(
            coarse_segmenter_backend="remote",
            vad_backend="remote",
        )
        assert cfg.coarse_segmenter_backend == "remote"
        assert cfg.vad_backend == "remote"

    def test_default_service_urls(self):
        cfg = PipelineConfig()
        assert cfg.segmentation_service_url == "http://127.0.0.1:8001"
        assert cfg.vad_service_url == "http://127.0.0.1:8002"

    def test_custom_urls(self):
        cfg = PipelineConfig(
            coarse_segmenter_backend="remote",
            vad_backend="remote",
            segmentation_service_url="http://gpu-host:9001",
            vad_service_url="http://gpu-host:9002",
        )
        assert cfg.segmentation_service_url == "http://gpu-host:9001"
        assert cfg.vad_service_url == "http://gpu-host:9002"

    def test_remote_timeout_defaults(self):
        cfg = PipelineConfig()
        assert cfg.remote_request_timeout_sec == 120.0
        assert cfg.remote_connect_timeout_sec == 10.0

    def test_custom_timeouts(self):
        cfg = PipelineConfig(
            remote_request_timeout_sec=300.0,
            remote_connect_timeout_sec=30.0,
        )
        assert cfg.remote_request_timeout_sec == 300.0
        assert cfg.remote_connect_timeout_sec == 30.0

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValidationError):
            PipelineConfig(coarse_segmenter_backend="invalid")

    def test_invalid_vad_backend_rejected(self):
        with pytest.raises(ValidationError):
            PipelineConfig(vad_backend="invalid")

    def test_backward_compatible_ina(self):
        cfg = PipelineConfig(coarse_segmenter_backend="ina", vad_backend="silero")
        assert cfg.coarse_segmenter_backend == "ina"
        assert cfg.vad_backend == "silero"
