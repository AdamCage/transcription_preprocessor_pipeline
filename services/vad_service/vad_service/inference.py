"""Silero VAD inference wrapper with GPU lifecycle management."""

from __future__ import annotations

import asyncio
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch

from vad_service.models import TimeSpanIn, TimeSpanOut

log = logging.getLogger(__name__)


def _merge_nearby_spans(
    spans: list[TimeSpanOut], gap: float
) -> list[TimeSpanOut]:
    if not spans:
        return []
    merged: list[TimeSpanOut] = [spans[0]]
    for sp in spans[1:]:
        prev = merged[-1]
        if sp.start - prev.end <= gap:
            merged[-1] = TimeSpanOut(start=prev.start, end=max(prev.end, sp.end))
        else:
            merged.append(sp)
    return merged


class SileroVADGPU:
    """Wraps Silero VAD JIT model for GPU inference."""

    def __init__(self, device: str = "cuda:0", executor_workers: int = 4) -> None:
        self._device = device
        self._model: object | None = None
        self._get_speech_timestamps: object | None = None
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
        device = self._resolve_device()
        log.info("Loading Silero VAD JIT model on device=%s", device)
        t0 = time.perf_counter()
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        model = model.to(device)
        model.eval()
        self._model = model
        self._get_speech_timestamps = utils[0]
        log.info("Silero VAD loaded in %.2fs on %s", time.perf_counter() - t0, device)

    def refine(
        self,
        wav_bytes: bytes,
        spans: list[TimeSpanIn],
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 200,
        speech_pad_ms: int = 200,
        merge_gap_seconds: float = 0.5,
    ) -> list[TimeSpanOut]:
        """Run VAD refinement on coarse speech spans (blocking)."""
        if self._model is None or self._get_speech_timestamps is None:
            raise RuntimeError("Model not loaded — call load() first")

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        samples = data.astype(np.float32)

        refined: list[TimeSpanOut] = []
        t0 = time.perf_counter()
        for sp in spans:
            i0 = max(0, int(sp.start * sr))
            i1 = min(len(samples), int(sp.end * sr))
            if i1 <= i0:
                continue
            chunk = samples[i0:i1]
            wav_tensor = torch.from_numpy(chunk).float().to(self._device)

            stamps = self._get_speech_timestamps(
                wav_tensor,
                self._model,
                threshold=threshold,
                sampling_rate=sr,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
            )
            for st in stamps:
                s_idx = int(st.get("start", 0))
                e_idx = int(st.get("end", 0))
                t_start = sp.start + s_idx / float(sr)
                t_end = sp.start + e_idx / float(sr)
                if t_end > t_start:
                    refined.append(TimeSpanOut(start=round(t_start, 4), end=round(t_end, 4)))

        inference_sec = time.perf_counter() - t0
        log.debug(
            "VAD refinement: %d input spans -> %d raw refined in %.3fs",
            len(spans),
            len(refined),
            inference_sec,
        )

        if not refined:
            return [TimeSpanOut(start=sp.start, end=sp.end) for sp in spans]

        return _merge_nearby_spans(
            sorted(refined, key=lambda s: s.start), merge_gap_seconds
        )

    async def refine_async(
        self,
        wav_bytes: bytes,
        spans: list[TimeSpanIn],
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 200,
        speech_pad_ms: int = 200,
        merge_gap_seconds: float = 0.5,
    ) -> list[TimeSpanOut]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.refine(
                wav_bytes,
                spans,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
                merge_gap_seconds=merge_gap_seconds,
            ),
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
        self._model = None
        self._get_speech_timestamps = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def device(self) -> str:
        return self._device
