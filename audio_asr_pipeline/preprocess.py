"""Coarse segmentation, filtering, and VAD orchestration."""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np

from audio_asr_pipeline.config import PipelineConfig
from audio_asr_pipeline.errors import SegmentationError
from audio_asr_pipeline.io import load_normalized_audio
from audio_asr_pipeline.models import LabeledSegment, LoadedAudio, TimeSpan
from audio_asr_pipeline.segmenters import get_coarse_segmenter
from audio_asr_pipeline.vad import refine_speech_spans_with_silero

log = logging.getLogger(__name__)


def _labeled_to_speech_spans(
    labeled: list[LabeledSegment],
    config: PipelineConfig,
) -> list[TimeSpan]:
    spans: list[TimeSpan] = []
    for seg in labeled:
        if seg.label == "speech":
            spans.append(TimeSpan(seg.start, seg.end))
            continue
        if seg.label == "music" and not config.drop_music:
            spans.append(TimeSpan(seg.start, seg.end))
        elif seg.label == "noise" and not config.drop_noise:
            spans.append(TimeSpan(seg.start, seg.end))
        elif seg.label in ("silence", "unknown") and not config.drop_silence:
            spans.append(TimeSpan(seg.start, seg.end))
    return spans


def _filter_short_spans(spans: list[TimeSpan], min_sec: float) -> list[TimeSpan]:
    return [s for s in spans if (s.end - s.start) >= min_sec]


def preprocess_audio(
    path: Path,
    config: PipelineConfig,
    *,
    loaded: LoadedAudio | None = None,
) -> tuple[LoadedAudio, list[TimeSpan], dict[str, float], list[LabeledSegment]]:
    """
    Load audio, run coarse segmentation + VAD, return kept speech time spans and timings.

    Returns
    -------
    loaded
        Normalized waveform.
    spans
        Speech spans on the original timeline (seconds).
    timings_sec
        coarse_segmentation_sec, vad_sec
    labeled_coarse
        Raw coarse labels (for pipeline_meta drops / debugging).
    """
    timings: dict[str, float] = {}
    fid = path.stem
    if loaded is None:
        log.info("preprocess | file_id=%s | stage=load_audio_begin | path=%s", fid, path.resolve())
        loaded = load_normalized_audio(
            path,
            target_sr=config.target_sample_rate,
            mono=config.mono,
        )
    else:
        log.info("preprocess | file_id=%s | stage=reuse_loaded | path=%s", fid, path.resolve())

    n_samples = int(loaded.samples.shape[0]) if getattr(loaded.samples, "shape", None) else 0
    log.info(
        "preprocess | file_id=%s | stage=loaded | duration_sec=%.3f | sr=%d | samples=%d",
        fid,
        loaded.duration_sec,
        loaded.sample_rate,
        n_samples,
    )

    segmenter = get_coarse_segmenter(
        config.coarse_segmenter_backend,
        ina_force_cpu=config.ina_force_cpu,
    )
    t0 = time.perf_counter()
    labeled = segmenter.segment(path.resolve(), duration_sec=loaded.duration_sec)
    timings["coarse_segmentation_sec"] = time.perf_counter() - t0
    label_counts = Counter(s.label for s in labeled)
    log.info(
        "preprocess | file_id=%s | stage=coarse_done | backend=%s | "
        "segments=%d | label_counts=%s | wall_sec=%.3f",
        fid,
        config.coarse_segmenter_backend,
        len(labeled),
        dict(label_counts),
        timings["coarse_segmentation_sec"],
    )

    speech_spans = _labeled_to_speech_spans(labeled, config)
    if not speech_spans:
        raise SegmentationError(f"No speech spans after coarse segmentation: {path}")

    if config.vad_backend == "none":
        timings["vad_sec"] = 0.0
        refined = speech_spans
        log.info(
            "preprocess | file_id=%s | stage=vad_skipped | spans=%d",
            fid,
            len(refined),
        )
    elif config.vad_backend == "silero":
        t1 = time.perf_counter()
        refined = refine_speech_spans_with_silero(
            np.asarray(loaded.samples, dtype=np.float32),
            loaded.sample_rate,
            speech_spans,
            threshold=config.vad_threshold,
            min_speech_duration_ms=config.min_speech_duration_ms,
            min_silence_duration_ms=config.min_silence_duration_ms,
            speech_pad_ms=config.speech_pad_ms,
            merge_gap_seconds=config.merge_gap_seconds,
        )
        timings["vad_sec"] = time.perf_counter() - t1
        pre_kept = sum(s.end - s.start for s in refined)
        log.info(
            "preprocess | file_id=%s | stage=vad_done | refined_spans=%d | "
            "kept_speech_sec_pre_filter=%.3f | wall_sec=%.3f",
            fid,
            len(refined),
            pre_kept,
            timings["vad_sec"],
        )
    else:
        raise SegmentationError(f"Unknown vad_backend: {config.vad_backend}")

    refined = _filter_short_spans(refined, config.min_segment_duration_sec)
    post_kept = sum(s.end - s.start for s in refined)
    log.info(
        "preprocess | file_id=%s | stage=filter_short_done | "
        "spans_after_min_dur=%d | kept_speech_sec=%.3f | min_segment_sec=%s",
        fid,
        len(refined),
        post_kept,
        config.min_segment_duration_sec,
    )
    if not refined:
        raise SegmentationError(f"No speech spans left after VAD/filter: {path}")

    return loaded, refined, timings, labeled
