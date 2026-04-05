"""verbose_json skeleton and merging chunk-level STT responses."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from audio_asr_pipeline.errors import MergeError
from audio_asr_pipeline.models import TranscribedChunk

log = logging.getLogger(__name__)


def build_verbose_json_skeleton(
    *,
    duration: float,
    pipeline_meta: dict[str, Any],
) -> dict[str, Any]:
    """Template compatible with OpenAI-style verbose_json plus pipeline_meta."""
    return {
        "text": "",
        "task": "transcribe",
        "language": None,
        "duration": duration,
        "segments": [],
        "words": [],
        "pipeline_meta": deepcopy(pipeline_meta),
    }


def _normalize_time_fields_ms_heuristic(
    d: dict[str, Any],
    chunk_duration_sec: float,
) -> dict[str, Any]:
    """
    If start/end look like milliseconds (much larger than chunk length in seconds),
    convert to seconds. Threshold: max(|start|, |end|) > chunk_duration_sec * 10.
    """
    out = dict(d)
    s = out.get("start")
    e = out.get("end")
    if not isinstance(s, (int, float)) or not isinstance(e, (int, float)):
        return out
    fs, fe = float(s), float(e)
    peak = max(abs(fs), abs(fe))
    if chunk_duration_sec > 0 and peak > chunk_duration_sec * 10:
        out["start"] = fs / 1000.0
        out["end"] = fe / 1000.0
    return out


def _shift_segment(
    seg: dict[str, Any],
    offset: float,
    *,
    chunk_duration_sec: float = 0.0,
) -> dict[str, Any]:
    seg = _normalize_time_fields_ms_heuristic(seg, chunk_duration_sec)
    out = dict(seg)
    for key in ("start", "end"):
        v = out.get(key)
        if isinstance(v, (int, float)):
            out[key] = float(v) + offset
    if "words" in out and isinstance(out["words"], list):
        out["words"] = [
            _shift_word(
                _normalize_time_fields_ms_heuristic(w, chunk_duration_sec),
                offset,
            )
            for w in out["words"]
            if isinstance(w, dict)
        ]
    return out


def _shift_word(w: dict[str, Any], offset: float) -> dict[str, Any]:
    out = dict(w)
    for key in ("start", "end"):
        v = out.get(key)
        if isinstance(v, (int, float)):
            out[key] = float(v) + offset
    return out


def _text_from_segments(segments: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for seg in segments:
        t = (seg.get("text") or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


def merge_transcriptions(
    skeleton: dict[str, Any],
    transcribed: list[TranscribedChunk],
    *,
    include_words: bool = False,
) -> dict[str, Any]:
    """
    Merge sorted chunk responses into one verbose_json-like dict.
    Assumes each chunk response uses times relative to chunk audio (start at 0).
    """
    ordered = sorted(transcribed, key=lambda c: c.start_offset)
    texts: list[str] = []
    all_segments: list[dict[str, Any]] = []
    all_words: list[dict[str, Any]] = []

    seg_id = 0
    for tc in ordered:
        resp = tc.response
        chunk_text = (resp.get("text") or "").strip()
        if chunk_text:
            texts.append(chunk_text)
        offset = tc.start_offset
        chunk_duration_sec = max(0.0, float(tc.end_offset) - float(tc.start_offset))
        for seg in resp.get("segments") or []:
            if not isinstance(seg, dict):
                continue
            s = _shift_segment(
                seg,
                offset,
                chunk_duration_sec=chunk_duration_sec,
            )
            s["id"] = seg_id
            seg_id += 1
            all_segments.append(s)
        if include_words:
            for w in resp.get("words") or []:
                if isinstance(w, dict):
                    wn = _normalize_time_fields_ms_heuristic(w, chunk_duration_sec)
                    all_words.append(_shift_word(wn, offset))

    out = deepcopy(skeleton)
    out["text"] = " ".join(texts).strip()
    out["segments"] = all_segments
    if include_words:
        out["words"] = all_words

    filled_from_segments = False
    if not out["text"] and all_segments:
        out["text"] = _text_from_segments(all_segments)
        filled_from_segments = bool(out["text"])

    langs = [tc.response.get("language") for tc in ordered if tc.response.get("language")]
    if langs:
        out["language"] = langs[0]

    if not out["text"] and not all_segments:
        raise MergeError(
            "Merged result is empty — all chunk responses missing text and segments"
        )

    log.info(
        "merge | chunks_merged=%d | text_chars=%d | segments=%d | text_from_segments=%s",
        len(ordered),
        len(out["text"]),
        len(all_segments),
        filled_from_segments,
    )
    return out
