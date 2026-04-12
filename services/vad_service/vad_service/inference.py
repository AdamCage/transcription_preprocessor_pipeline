"""Silero VAD inference wrapper with GPU lifecycle management."""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vad_service.models import TimeSpanIn, TimeSpanOut

log = logging.getLogger(__name__)

_ModelPair = tuple[torch.jit.ScriptModule, Callable[..., list[dict[str, int]]]]

_SILERO_SUPPORTED_RATES = frozenset({8000, 16000})


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
    """Wraps Silero VAD JIT model for GPU inference.

    Uses a pool of model replicas (one per executor worker) so that
    concurrent threads never share mutable model state.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        executor_workers: int = 1,
        model_path: str = "",
    ) -> None:
        self._device = device
        self._executor_workers = executor_workers
        self._model_path = model_path
        self._model_pool: queue.Queue[_ModelPair] = queue.Queue()
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

    def _load_single_model(self, device: torch.device) -> _ModelPair:
        """Load one Silero VAD model replica onto *device*."""
        if self._model_path and Path(self._model_path).is_file():
            log.info("Loading Silero VAD from local path: %s", self._model_path)
            model = torch.jit.load(self._model_path, map_location=device)
            model.eval()
            try:
                from silero_vad.utils_vad import get_speech_timestamps  # type: ignore[import-untyped]
            except ImportError:
                from torch.hub import load as _hub_load
                _, utils = _hub_load(
                    "snakers4/silero-vad",
                    "silero_vad",
                    force_reload=False,
                    onnx=False,
                    trust_repo=True,
                )
                get_speech_timestamps = utils[0]
            return model, get_speech_timestamps

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        model = model.to(device)
        model.eval()
        return model, utils[0]

    def load(self) -> None:
        device = self._resolve_device()
        n = self._executor_workers
        log.info(
            "Loading %d Silero VAD replica(s) on device=%s", n, device,
        )
        t0 = time.perf_counter()
        for i in range(n):
            pair = self._load_single_model(device)
            self._model_pool.put(pair)
            log.debug("Replica %d/%d loaded", i + 1, n)
        log.info(
            "Silero VAD: %d replica(s) loaded in %.2fs on %s",
            n, time.perf_counter() - t0, device,
        )

    @property
    def loaded(self) -> bool:
        return not self._model_pool.empty()

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
        """Run VAD refinement on coarse speech spans (blocking, thread-safe)."""
        if self._model_pool.empty():
            raise RuntimeError("Model not loaded — call load() first")

        model, get_speech_timestamps = self._model_pool.get()
        try:
            return self._refine_with_model(
                model,
                get_speech_timestamps,
                wav_bytes,
                spans,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
                merge_gap_seconds=merge_gap_seconds,
            )
        finally:
            self._model_pool.put((model, get_speech_timestamps))

    def _refine_with_model(
        self,
        model: torch.jit.ScriptModule,
        get_speech_timestamps: Callable[..., list[dict[str, int]]],
        wav_bytes: bytes,
        spans: list[TimeSpanIn],
        *,
        threshold: float,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
        merge_gap_seconds: float,
    ) -> list[TimeSpanOut]:
        model.reset_states()

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        samples = data.astype(np.float32)

        if sr not in _SILERO_SUPPORTED_RATES:
            log.warning(
                "Unexpected sample rate %d Hz; Silero expects 8/16 kHz. "
                "Results may degrade.",
                sr,
            )

        refined: list[TimeSpanOut] = []
        t0 = time.perf_counter()
        for sp in spans:
            i0 = max(0, int(sp.start * sr))
            i1 = min(len(samples), int(sp.end * sr))
            if i1 <= i0:
                continue
            chunk = samples[i0:i1]
            wav_tensor = torch.from_numpy(chunk).float().to(self._device)

            stamps = get_speech_timestamps(
                wav_tensor,
                model,
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
        self._executor.shutdown(wait=True, cancel_futures=True)
        while not self._model_pool.empty():
            try:
                self._model_pool.get_nowait()
            except queue.Empty:
                break
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def device(self) -> str:
        return self._device
