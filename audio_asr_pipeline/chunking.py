"""Build time-bounded chunks from speech spans."""

from __future__ import annotations

import uuid

from audio_asr_pipeline.config import PipelineConfig
from audio_asr_pipeline.models import AudioChunk, AudioFileTask, TimeSpan


def build_chunks_from_spans(
    task: AudioFileTask,
    spans: list[TimeSpan],
    config: PipelineConfig,
) -> list[TimeSpan]:
    """
    Split long continuous spans into sub-spans of at most max_chunk_duration_sec.
    Drops spans shorter than min_chunk_duration_sec unless merged into neighbors.
    """
    max_d = config.max_chunk_duration_sec
    min_d = config.min_chunk_duration_sec
    out: list[TimeSpan] = []
    for sp in sorted(spans, key=lambda s: s.start):
        dur = sp.end - sp.start
        if dur < min_d:
            continue
        if dur <= max_d:
            out.append(sp)
            continue
        cur = sp.start
        while cur < sp.end:
            nxt = min(cur + max_d, sp.end)
            piece = TimeSpan(cur, nxt)
            plen = piece.end - piece.start
            if plen >= min_d:
                out.append(piece)
            elif out and abs(cur - out[-1].end) < 1e-3:
                out[-1] = TimeSpan(out[-1].start, nxt)
            elif out:
                out[-1] = TimeSpan(out[-1].start, nxt)
            cur = nxt
    return out


def spans_to_audio_chunks(
    task: AudioFileTask,
    span_list: list[TimeSpan],
) -> list[AudioChunk]:
    """Wrap TimeSpan rows as AudioChunk placeholders (bytes filled later)."""
    chunks: list[AudioChunk] = []
    for sp in span_list:
        cid = f"{task.file_id}_{uuid.uuid4().hex[:12]}"
        chunks.append(
            AudioChunk(
                chunk_id=cid,
                file_id=task.file_id,
                start=sp.start,
                end=sp.end,
                audio_bytes=None,
                sample_rate=0,
                num_samples=0,
                source_path=task.source_path,
            )
        )
    return chunks
