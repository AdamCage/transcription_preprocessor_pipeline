import asyncio
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_asr_pipeline.config import PipelineConfig, VLLMTranscribeConfig
from audio_asr_pipeline.pipeline import process_file_sync


def test_process_file_sync_under_running_loop(tmp_path: Path) -> None:
    """Offloads asyncio.run to a worker thread when a loop is already running (no nested-loop crash)."""
    wav = tmp_path / "x.wav"
    sf.write(wav, np.zeros(3200, dtype=np.float32), 16000)
    cfg = PipelineConfig(
        coarse_segmenter_backend="whole_file",
        vllm=VLLMTranscribeConfig(
            base_url="http://127.0.0.1:9",
            connect_timeout_sec=0.3,
            request_timeout_sec=0.5,
            max_retries=0,
        ),
    )

    async def caller() -> None:
        r = process_file_sync(wav, cfg)
        assert r.error is not None

    asyncio.run(caller())
