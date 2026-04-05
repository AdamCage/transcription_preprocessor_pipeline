"""Silero VAD refinement inside coarse speech spans."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np
import torch

from audio_asr_pipeline.errors import VADProcessingError
from audio_asr_pipeline.models import TimeSpan
from audio_asr_pipeline.utils import merge_nearby_spans

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)
_model_cache: tuple[object, tuple | list] | None = None
_model_lock = threading.Lock()


def _load_silero() -> tuple[object, tuple | list]:
    global _model_cache  # noqa: PLW0603
    if _model_cache is not None:
        return _model_cache
    with _model_lock:
        if _model_cache is not None:
            return _model_cache
        try:
            # ONNX avoids TorchScript "NYI" failures inside silero_vad.jit on newer PyTorch (often on Windows).
            model, utils = torch.hub.load(  # type: ignore[union-attr]
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
                force_onnx_cpu=True,
                trust_repo=True,
            )
        except Exception as e:  # noqa: BLE001
            raise VADProcessingError(
                f"Failed to load Silero VAD (ONNX): {e}. "
                "Ensure onnxruntime is installed (uv sync)."
            ) from e
        _model_cache = (model, utils)
        log.info("Silero VAD loaded (ONNX via torch.hub)")
    return _model_cache


def refine_speech_spans_with_silero(
    samples: np.ndarray,
    sample_rate: int,
    coarse_speech_spans: list[TimeSpan],
    *,
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    merge_gap_seconds: float,
) -> list[TimeSpan]:
    """
    Run Silero VAD on each coarse speech span and map timestamps back to the global axis.
    """
    model, utils = _load_silero()
    get_speech_timestamps = utils[0]
    device = torch.device("cpu")
    # Silero ONNX backend uses OnnxWrapper (no .to / .eval); JIT nn.Module has them.
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    refined: list[TimeSpan] = []
    for sp in coarse_speech_spans:
        i0 = max(0, int(sp.start * sample_rate))
        i1 = min(len(samples), int(sp.end * sample_rate))
        if i1 <= i0:
            continue
        chunk = samples[i0:i1]
        wav = torch.from_numpy(chunk).float().to(device)
        try:
            stamps = get_speech_timestamps(
                wav,
                model,
                threshold=threshold,
                sampling_rate=sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
            )
        except Exception as e:  # noqa: BLE001
            raise VADProcessingError("Silero get_speech_timestamps failed") from e

        for st in stamps:
            s_idx = int(st.get("start", 0))
            e_idx = int(st.get("end", 0))
            t0 = sp.start + s_idx / float(sample_rate)
            t1 = sp.start + e_idx / float(sample_rate)
            if t1 > t0:
                refined.append(TimeSpan(t0, t1))

    if not refined:
        return list(coarse_speech_spans)

    return merge_nearby_spans(refined, merge_gap_seconds)
