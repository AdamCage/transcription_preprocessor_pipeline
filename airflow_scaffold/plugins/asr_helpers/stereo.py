"""Stereo call-center path: load 2ch WAV, split, run one pipeline with two process_file calls."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np

from audio_asr_pipeline.config import PipelineConfig
from audio_asr_pipeline.io import split_stereo_channels, write_mono_wav
from audio_asr_pipeline.models import PipelineResult
from audio_asr_pipeline.pipeline import AudioTranscriptionPipeline

log = logging.getLogger(__name__)


def _run_async_in_fresh_thread(factory):
    """Mirror audio_asr_pipeline.pipeline._run_async_in_fresh_thread (avoid importing private)."""

    def runner():
        return asyncio.run(factory())

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return runner()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(runner).result()


async def _run_stereo_async(
    stereo_path: Path,
    work_dir: Path,
    cfg: PipelineConfig,
) -> tuple[PipelineResult, PipelineResult]:
    work_dir.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(str(stereo_path), sr=cfg.target_sample_rate, mono=False)
    if y.ndim == 1:
        raise ValueError(f"expected stereo (2 channels), got mono: {stereo_path}")
    ch0, ch1, _n = split_stereo_channels(np.asarray(y))
    p_from = work_dir / f"{stereo_path.stem}__call_from.wav"
    p_to = work_dir / f"{stereo_path.stem}__call_to.wav"
    write_mono_wav(p_from, ch0, sr)
    write_mono_wav(p_to, ch1, sr)
    log.info("stereo split | %s -> %s, %s", stereo_path.name, p_from.name, p_to.name)

    async with AudioTranscriptionPipeline(cfg) as pipe:
        res_from, res_to = await asyncio.gather(
            pipe.process_file(p_from),
            pipe.process_file(p_to),
        )
    return res_from, res_to


def run_stereo_file_sync(
    stereo_path: Path | str,
    work_dir: Path | str,
    cfg: PipelineConfig,
) -> tuple[PipelineResult, PipelineResult]:
    """Load stereo WAV, write two mono files, transcribe both with a single pipeline instance."""

    sp = Path(stereo_path)
    wd = Path(work_dir)

    async def main():
        return await _run_stereo_async(sp, wd, cfg)

    return _run_async_in_fresh_thread(lambda: main())
