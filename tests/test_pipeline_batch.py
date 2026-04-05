import asyncio
from pathlib import Path

import pytest

from audio_asr_pipeline.config import PipelineConfig
from audio_asr_pipeline.pipeline import AudioTranscriptionPipeline


def test_process_files_isolates_prepare_failure(tmp_path: Path) -> None:
    async def main() -> None:
        cfg = PipelineConfig(
            fail_fast=False,
            coarse_segmenter_backend="whole_file",
        )
        pipe = AudioTranscriptionPipeline(cfg)
        bad = tmp_path / "missing.wav"
        try:
            res = await pipe.process_file(bad)
            assert res.error is not None
            assert "AudioLoadError" in res.error or "Failed" in res.error
        finally:
            await pipe.aclose()

    asyncio.run(main())


def test_fail_fast_prepare_raises(tmp_path: Path) -> None:
    async def main() -> None:
        cfg = PipelineConfig(
            fail_fast=True,
            coarse_segmenter_backend="whole_file",
        )
        pipe = AudioTranscriptionPipeline(cfg)
        bad = tmp_path / "missing2.wav"
        try:
            with pytest.raises(Exception):  # noqa: PT011
                await pipe.process_file(bad)
        finally:
            await pipe.aclose()

    asyncio.run(main())


def test_gather_returns_all_results_when_not_fail_fast(tmp_path: Path) -> None:
    async def main() -> None:
        cfg = PipelineConfig(fail_fast=False, coarse_segmenter_backend="whole_file")
        pipe = AudioTranscriptionPipeline(cfg)
        p1 = tmp_path / "a.wav"
        p2 = tmp_path / "b.wav"
        try:
            out = await pipe.process_files([p1, p2])
            assert len(out) == 2
            assert all(r.error for r in out)
        finally:
            await pipe.aclose()

    asyncio.run(main())
