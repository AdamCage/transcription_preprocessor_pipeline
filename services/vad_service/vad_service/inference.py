"""Silero VAD inference wrapper with GPU lifecycle management."""

from __future__ import annotations

import asyncio
import io
import math
import queue
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog
import torch

from vad_service.models import TimeSpanIn, TimeSpanOut

log = structlog.stdlib.get_logger(__name__)

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
    concurrent threads never share mutable model state.  When
    ``executor_workers > 1`` and a request contains multiple spans,
    the spans are processed in parallel across worker threads.
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
                "CUDA requested but not available, falling back to CPU",
                requested_device=requested,
            )
            self._device = "cpu"
            return torch.device("cpu")
        return torch.device(requested)

    def _load_single_model(self, device: torch.device) -> _ModelPair:
        """Load one Silero VAD model replica onto *device*."""
        if self._model_path and Path(self._model_path).is_file():
            log.info("Loading Silero VAD from local path", path=self._model_path)
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
        log.info("Loading Silero VAD replicas", count=n, device=str(device))
        t0 = time.perf_counter()
        for i in range(n):
            pair = self._load_single_model(device)
            self._model_pool.put(pair)
            log.debug("Replica loaded", replica=i + 1, total=n)
        log.info(
            "Silero VAD replicas ready",
            count=n,
            load_sec=round(time.perf_counter() - t0, 2),
            device=str(device),
        )

    @property
    def loaded(self) -> bool:
        return not self._model_pool.empty()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        if not spans:
            return []

        if self._model_pool.empty():
            raise RuntimeError("Model not loaded — call load() first")

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        samples = data.astype(np.float32)

        if sr not in _SILERO_SUPPORTED_RATES:
            log.warning(
                "Unexpected sample rate; Silero expects 8/16 kHz",
                sample_rate=sr,
            )

        vad_kwargs = dict(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        use_parallel = self._executor_workers > 1 and len(spans) > 1
        t0 = time.perf_counter()

        if use_parallel:
            refined = self._refine_spans_parallel(samples, sr, spans, vad_kwargs)
        else:
            refined = self._refine_spans_sequential(samples, sr, spans, vad_kwargs)

        wall_sec = time.perf_counter() - t0
        log.debug(
            "VAD refinement complete",
            spans_in=len(spans),
            spans_raw=len(refined),
            wall_sec=round(wall_sec, 3),
            parallel=use_parallel,
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

    # ------------------------------------------------------------------
    # Internal: sequential and parallel span processing
    # ------------------------------------------------------------------

    def _process_span(
        self,
        model: torch.jit.ScriptModule,
        get_speech_timestamps: Callable[..., list[dict[str, int]]],
        samples: np.ndarray,
        sr: int,
        sp: TimeSpanIn,
        vad_kwargs: dict,
        *,
        worker_id: int = 0,
        span_idx: int = 0,
    ) -> list[TimeSpanOut]:
        """Process a single span with the given model replica."""
        i0 = max(0, int(sp.start * sr))
        i1 = min(len(samples), int(sp.end * sr))
        if i1 <= i0:
            return []
        chunk = samples[i0:i1]
        wav_tensor = torch.from_numpy(chunk).float().to(self._device)

        t0 = time.perf_counter()
        stamps = get_speech_timestamps(
            wav_tensor,
            model,
            sampling_rate=sr,
            **vad_kwargs,
        )
        span_sec = time.perf_counter() - t0

        result: list[TimeSpanOut] = []
        for st in stamps:
            s_idx = int(st.get("start", 0))
            e_idx = int(st.get("end", 0))
            t_start = sp.start + s_idx / float(sr)
            t_end = sp.start + e_idx / float(sr)
            if t_end > t_start:
                result.append(TimeSpanOut(start=round(t_start, 4), end=round(t_end, 4)))

        log.debug(
            "Span processed",
            worker_id=worker_id,
            span_idx=span_idx,
            span_start=sp.start,
            span_end=sp.end,
            vad_segments=len(result),
            inference_sec=round(span_sec, 4),
        )
        return result

    def _refine_spans_sequential(
        self,
        samples: np.ndarray,
        sr: int,
        spans: list[TimeSpanIn],
        vad_kwargs: dict,
    ) -> list[TimeSpanOut]:
        """Process all spans sequentially with a single model replica."""
        model, get_ts = self._model_pool.get()
        try:
            model.reset_states()
            refined: list[TimeSpanOut] = []
            for idx, sp in enumerate(spans):
                refined.extend(
                    self._process_span(
                        model, get_ts, samples, sr, sp, vad_kwargs,
                        worker_id=0, span_idx=idx,
                    )
                )
            return refined
        finally:
            self._model_pool.put((model, get_ts))

    def _refine_spans_parallel(
        self,
        samples: np.ndarray,
        sr: int,
        spans: list[TimeSpanIn],
        vad_kwargs: dict,
    ) -> list[TimeSpanOut]:
        """Split spans across worker threads, each with its own model replica."""
        n_workers = min(self._executor_workers, len(spans))
        chunk_size = math.ceil(len(spans) / n_workers)
        span_chunks = [
            spans[i : i + chunk_size] for i in range(0, len(spans), chunk_size)
        ]

        log.debug(
            "Parallel span dispatch",
            n_workers=n_workers,
            total_spans=len(spans),
            chunk_sizes=[len(c) for c in span_chunks],
        )

        def _worker_fn(worker_id: int, batch: list[TimeSpanIn]) -> list[TimeSpanOut]:
            model, get_ts = self._model_pool.get()
            try:
                model.reset_states()
                results: list[TimeSpanOut] = []
                for local_idx, sp in enumerate(batch):
                    results.extend(
                        self._process_span(
                            model, get_ts, samples, sr, sp, vad_kwargs,
                            worker_id=worker_id, span_idx=local_idx,
                        )
                    )
                return results
            finally:
                self._model_pool.put((model, get_ts))

        futures: list[Future[list[TimeSpanOut]]] = []
        for wid, batch in enumerate(span_chunks):
            futures.append(self._executor.submit(_worker_fn, wid, batch))

        refined: list[TimeSpanOut] = []
        for fut in futures:
            refined.extend(fut.result())
        return refined
