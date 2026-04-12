"""pyannote.audio inference wrapper with GPU lifecycle management."""

from __future__ import annotations

import asyncio
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch

from segmentation_service.models import SegmentItem

log = logging.getLogger(__name__)


class PyannoteSegmenter:
    """Wraps a pyannote VAD/segmentation pipeline for GPU inference."""

    def __init__(
        self,
        model_name: str,
        device: str,
        hf_token: str,
        executor_workers: int = 4,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._hf_token = hf_token or None
        self._pipeline: object | None = None
        self._executor = ThreadPoolExecutor(max_workers=executor_workers)

    def _resolve_device(self) -> torch.device:
        requested = self._device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            log.warning(
                "CUDA requested (%s) but not available — falling back to CPU", requested
            )
            self._device = "cpu"
            return torch.device("cpu")
        return torch.device(requested)

    def load(self) -> None:
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]

        device = self._resolve_device()
        log.info(
            "Loading pyannote pipeline model=%s device=%s",
            self._model_name,
            device,
        )
        t0 = time.perf_counter()
        pipeline = Pipeline.from_pretrained(
            self._model_name,
            use_auth_token=self._hf_token,
        )
        pipeline.to(device)
        self._pipeline = pipeline
        log.info("Model loaded in %.2fs on %s", time.perf_counter() - t0, device)

    def segment(self, wav_bytes: bytes) -> tuple[list[SegmentItem], float]:
        """Run segmentation (blocking). Returns (segments, duration_sec)."""
        if self._pipeline is None:
            raise RuntimeError("Model not loaded — call load() first")

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        duration_sec = float(len(data)) / sr

        waveform = torch.from_numpy(data).unsqueeze(0).to(self._device)
        audio_input = {"waveform": waveform, "sample_rate": sr}

        t0 = time.perf_counter()
        output = self._pipeline(audio_input)
        inference_sec = time.perf_counter() - t0
        log.debug("Inference took %.3fs for %.1fs audio", inference_sec, duration_sec)

        segments: list[SegmentItem] = []
        prev_end = 0.0
        for speech_turn in output.itertracks():
            seg_start = round(speech_turn[0].start, 4)
            seg_end = round(speech_turn[0].end, 4)
            if seg_start > prev_end + 0.01:
                segments.append(
                    SegmentItem(start=prev_end, end=seg_start, label="non_speech")
                )
            segments.append(SegmentItem(start=seg_start, end=seg_end, label="speech"))
            prev_end = seg_end

        if prev_end < duration_sec - 0.01:
            segments.append(
                SegmentItem(start=prev_end, end=duration_sec, label="non_speech")
            )

        if not segments:
            segments.append(
                SegmentItem(start=0.0, end=duration_sec, label="speech")
            )

        return segments, duration_sec

    async def segment_async(
        self, wav_bytes: bytes
    ) -> tuple[list[SegmentItem], float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.segment, wav_bytes)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_name(self) -> str:
        return self._model_name
