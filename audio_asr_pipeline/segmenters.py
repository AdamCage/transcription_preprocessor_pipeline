"""Coarse speech / music / noise segmenters."""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from audio_asr_pipeline.errors import SegmentationError
from audio_asr_pipeline.models import LabeledSegment, SegmentLabel

log = logging.getLogger(__name__)

_ina_segmenter_singleton: object | None = None
_ina_lock = threading.Lock()


def _apply_ina_tensorflow_cpu_only() -> None:
    """
    Prefer CPU for inaSpeechSegmenter (TensorFlow).

    If TensorFlow is not loaded yet, CUDA_VISIBLE_DEVICES=-1 prevents GPU init.
    If it is already loaded, try hiding GPU devices (best effort).
    """
    if "tensorflow" not in sys.modules:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        log.debug("ina: set CUDA_VISIBLE_DEVICES=-1 before TensorFlow import")
        return
    try:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        log.debug("ina: tf.config.set_visible_devices([], 'GPU')")
    except Exception as e:  # noqa: BLE001
        log.warning("ina: could not force CPU-only TensorFlow: %s", e)


@runtime_checkable
class BaseSpeechMusicSegmenter(Protocol):
    def segment(self, path: Path, *, duration_sec: float) -> list[LabeledSegment]:
        ...


def _map_ina_label(raw: str) -> SegmentLabel:
    low = raw.lower()
    # ina Segmenter(detect_gender=True) tags speech as female/male, not "speech"
    if low in ("female", "male"):
        return "speech"
    if "speech" in low and "no" not in low[:2]:  # naive
        return "speech"
    if "music" in low:
        return "music"
    if "noise" in low or "noenergy" in low or low == "energy":
        return "noise"
    return "silence"


class WholeFileCoarseSegmenter:
    """Single speech segment covering the whole file (speech-only inputs)."""

    def segment(self, path: Path, *, duration_sec: float) -> list[LabeledSegment]:  # noqa: ARG002
        if duration_sec <= 0:
            return []
        return [LabeledSegment(start=0.0, end=duration_sec, label="speech")]


class InaSpeechMusicSegmenter:
    """
    inaSpeechSegmenter backend. Install inaSpeechSegmenter in the venv (see README).

    By default TensorFlow is pinned to CPU via CUDA_VISIBLE_DEVICES / tf.config.
    """

    def __init__(self, *, force_cpu: bool = True) -> None:
        self._force_cpu = force_cpu

    def segment(self, path: Path, *, duration_sec: float) -> list[LabeledSegment]:  # noqa: ARG002
        if self._force_cpu:
            _apply_ina_tensorflow_cpu_only()
        try:
            try:
                from inaSpeechSegmenter import segment as ina_callable  # type: ignore[attr-defined]
            except ImportError:
                from inaSpeechSegmenter import Segmenter

                global _ina_segmenter_singleton
                if _ina_segmenter_singleton is None:
                    with _ina_lock:
                        if _ina_segmenter_singleton is None:
                            _ina_segmenter_singleton = Segmenter()
                ina_callable = _ina_segmenter_singleton
        except ImportError as e:
            raise SegmentationError(
                "inaSpeechSegmenter is not installed. "
                "Run: uv sync --extra ina (or pip install inaSpeechSegmenter in this venv). "
                "If coarse_segmenter_backend is 'ina', this package is required; "
                "or use coarse_segmenter_backend='whole_file' / eval --coarse-backend whole_file."
            ) from e

        try:
            raw = ina_callable(str(path.resolve()))
        except Exception as e:  # noqa: BLE001
            raise SegmentationError(f"inaSpeechSegmenter failed for {path}") from e

        return _parse_ina_result(raw, duration_sec)


def _parse_ina_result(raw: object, duration_sec: float) -> list[LabeledSegment]:
    """Normalize different return shapes from inaSpeechSegmenter versions."""
    out: list[LabeledSegment] = []
    if raw is None:
        return [LabeledSegment(0.0, duration_sec, "speech")]

    rows = raw if isinstance(raw, list) else getattr(raw, "tolist", lambda: [])()
    if not rows:
        return [LabeledSegment(0.0, duration_sec, "speech")]

    for row in rows:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            label_s, t0, t1 = row[0], float(row[1]), float(row[2])
            out.append(
                LabeledSegment(
                    start=max(0.0, t0),
                    end=min(duration_sec, t1) if duration_sec > 0 else t1,
                    label=_map_ina_label(str(label_s)),
                )
            )
        elif isinstance(row, dict):
            lab = str(row.get("label") or row.get("type") or "speech")
            t0 = float(row.get("start", row.get("from", 0.0)))
            t1 = float(row.get("end", row.get("to", duration_sec)))
            out.append(
                LabeledSegment(
                    start=max(0.0, t0),
                    end=min(duration_sec, t1) if duration_sec > 0 else t1,
                    label=_map_ina_label(lab),
                )
            )

    if not out:
        return [LabeledSegment(0.0, duration_sec, "speech")]
    out.sort(key=lambda s: s.start)
    return out


def get_coarse_segmenter(
    name: str,
    *,
    ina_force_cpu: bool = True,
) -> BaseSpeechMusicSegmenter:
    if name == "whole_file":
        return WholeFileCoarseSegmenter()
    if name == "ina":
        return InaSpeechMusicSegmenter(force_cpu=ina_force_cpu)
    raise SegmentationError(f"Unknown coarse_segmenter_backend: {name}")