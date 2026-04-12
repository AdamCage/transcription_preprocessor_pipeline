"""pyannote.audio inference wrapper with GPU lifecycle management."""

from __future__ import annotations

import asyncio
import functools
import io
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog
import torch

from segmentation_service.models import SegmentItem

log = structlog.stdlib.get_logger(__name__)


def _span(operation: str) -> str:
    """Generate a span ID and log entry for the operation."""
    span_id = uuid.uuid4().hex[:12]
    log.debug("span.start", span_id=span_id, operation=operation)
    return span_id


def _span_end(span_id: str, operation: str, t0: float) -> None:
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.debug("span.end", span_id=span_id, operation=operation, duration_ms=duration_ms)


class PyannoteSegmenter:
    """Wraps a pyannote segmentation model + VoiceActivityDetection pipeline."""

    _DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def __init__(
        self,
        model_name: str,
        device: str,
        hf_token: str,
        dtype: str = "float32",
        *,
        model_path: str = "",
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._model_path = model_path or ""
        self._device = device
        self._hf_token = hf_token or None
        self._dtype = self._DTYPE_MAP.get(dtype, torch.float32)
        self._min_duration_on = min_duration_on
        self._min_duration_off = min_duration_off
        self._pipeline: object | None = None
        # Single-threaded: pyannote Pipeline + PyTorch GPU ops are not
        # thread-safe on a shared model.  The asyncio.Semaphore in app.py
        # gates I/O concurrency; this executor serialises GPU work.
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _resolve_device(self) -> torch.device:
        requested = self._device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            log.warning(
                "CUDA requested (%s) but not available — falling back to CPU", requested
            )
            self._device = "cpu"
            return torch.device("cpu")
        return torch.device(requested)

    @staticmethod
    def _patch_hf_auth() -> None:
        """Translate ``use_auth_token`` -> ``token`` for huggingface_hub >= 1.0.

        pyannote.audio <= 3.4 passes the deprecated kwarg to
        ``hf_hub_download`` which newer huggingface_hub rejects.
        """
        import inspect

        try:
            import huggingface_hub as _hf
        except ImportError:
            return

        _real_dl = _hf.hf_hub_download
        if "use_auth_token" in inspect.signature(_real_dl).parameters:
            return  # old huggingface_hub that still supports it

        @functools.wraps(_real_dl)
        def _patched_dl(*a: object, **kw: object) -> object:
            if "use_auth_token" in kw:
                kw.setdefault("token", kw.pop("use_auth_token"))
            return _real_dl(*a, **kw)

        _hf.hf_hub_download = _patched_dl
        if hasattr(_hf, "file_download"):
            _hf.file_download.hf_hub_download = _patched_dl

        try:
            import pyannote.audio.core.pipeline as _pp  # type: ignore[import-untyped]

            if hasattr(_pp, "hf_hub_download"):
                _pp.hf_hub_download = _patched_dl  # type: ignore[attr-defined]
        except (ImportError, ModuleNotFoundError):
            pass

    @staticmethod
    def _stub_missing_speechbrain_modules() -> None:
        """Replace speechbrain ``LazyModule`` sentinels with real empty modules.

        pytorch_lightning's ``load_from_checkpoint`` calls ``inspect.stack()``,
        and CPython's ``inspect.getmodule`` probes ``__file__`` on every entry
        in ``sys.modules``.  speechbrain registers ``LazyModule`` sentinels for
        optional integrations (k2_fsa, huggingface, etc.); accessing
        ``__file__`` triggers the lazy import, which raises ``ImportError``
        when the optional dependency is absent.  Replacing all ``LazyModule``
        entries with real (empty) module objects prevents the lazy loader from
        ever firing during frame introspection.
        """
        import sys
        import types

        lazy_type = None
        try:
            from speechbrain.utils.importutils import LazyModule as _LM
            lazy_type = _LM
        except ImportError:
            return

        for mod_name, mod_obj in list(sys.modules.items()):
            if isinstance(mod_obj, lazy_type):
                sys.modules[mod_name] = types.ModuleType(mod_name)

    def load(self) -> None:
        import os

        sid = _span("model_load")
        t0 = time.perf_counter()

        if self._hf_token and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = self._hf_token

        self._patch_hf_auth()
        self._stub_missing_speechbrain_modules()

        from pyannote.audio import Model  # type: ignore[import-untyped]
        from pyannote.audio.pipelines import VoiceActivityDetection  # type: ignore[import-untyped]

        self._patch_hf_auth()

        device = self._resolve_device()

        if self._model_path:
            p = Path(self._model_path)
            if p.is_dir():
                ckpt = p / "pytorch_model.bin"
                checkpoint: str | Path = str(ckpt) if ckpt.is_file() else str(p)
            else:
                checkpoint = str(p)
        else:
            checkpoint = self._model_name
        log.info(
            "Loading pyannote segmentation model",
            checkpoint=str(checkpoint),
            device=str(device),
        )

        _orig_torch_load = torch.load

        @functools.wraps(_orig_torch_load)
        def _safe_load(*a: object, **kw: object) -> object:
            kw["weights_only"] = False
            return _orig_torch_load(*a, **kw)

        torch.load = _safe_load
        try:
            model = Model.from_pretrained(
                checkpoint,
                use_auth_token=self._hf_token,
            )
        finally:
            torch.load = _orig_torch_load
        model = model.to(device)

        pipeline = VoiceActivityDetection(segmentation=model)
        pipeline.instantiate({
            "min_duration_on": self._min_duration_on,
            "min_duration_off": self._min_duration_off,
        })

        if self._dtype != torch.float32:
            try:
                model.to(self._dtype)
                log.info("Model cast to %s", self._dtype)
            except Exception:  # noqa: BLE001
                log.warning("Failed to cast model to %s, keeping float32", self._dtype)
                self._dtype = torch.float32

        self._pipeline = pipeline
        _span_end(sid, "model_load", t0)
        log.info("Model loaded in %.2fs on %s", time.perf_counter() - t0, device)

    def warmup(self) -> None:
        """Run a throwaway inference to JIT-compile CUDA kernels."""
        sid = _span("warmup")
        t0 = time.perf_counter()
        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, silence, sr, format="WAV", subtype="PCM_16")
        self.segment(buf.getvalue())
        _span_end(sid, "warmup", t0)
        log.info("Warmup inference completed in %.2fs", time.perf_counter() - t0)

    def segment(
        self,
        wav_bytes: bytes,
        *,
        max_duration_sec: float = 0.0,
    ) -> tuple[list[SegmentItem], float]:
        """Run segmentation (blocking). Returns (segments, duration_sec).

        Args:
            wav_bytes: Raw audio file bytes (WAV, FLAC, etc.).
            max_duration_sec: Reject audio longer than this (0 = no limit).
        """
        if self._pipeline is None:
            raise RuntimeError("Model not loaded — call load() first")

        # --- decode ---
        sid_dec = _span("decode_audio")
        t_dec = time.perf_counter()
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        duration_sec = float(len(data)) / sr
        _span_end(sid_dec, "decode_audio", t_dec)

        if max_duration_sec > 0 and duration_sec > max_duration_sec:
            raise ValueError(
                f"Audio duration {duration_sec:.1f}s exceeds limit "
                f"{max_duration_sec:.0f}s"
            )

        # --- GPU inference ---
        sid_inf = _span("gpu_inference")
        t_inf = time.perf_counter()

        waveform = torch.from_numpy(data).unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": sr}

        output = self._pipeline(audio_input)
        inference_sec = time.perf_counter() - t_inf
        _span_end(sid_inf, "gpu_inference", t_inf)
        log.debug("Inference took %.3fs for %.1fs audio", inference_sec, duration_sec)

        # --- postprocess ---
        sid_pp = _span("postprocess")
        t_pp = time.perf_counter()

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

        _span_end(sid_pp, "postprocess", t_pp)
        return segments, duration_sec

    async def segment_async(
        self,
        wav_bytes: bytes,
        *,
        max_duration_sec: float = 0.0,
    ) -> tuple[list[SegmentItem], float]:
        ctx = structlog.contextvars.get_contextvars()

        def _run() -> tuple[list[SegmentItem], float]:
            structlog.contextvars.bind_contextvars(**ctx)
            return self.segment(wav_bytes, max_duration_sec=max_duration_sec)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _run)

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
