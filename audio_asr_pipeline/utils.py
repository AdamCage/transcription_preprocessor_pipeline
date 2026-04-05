"""Small shared utilities."""

from __future__ import annotations

from audio_asr_pipeline.models import TimeSpan


def merge_nearby_spans(spans: list[TimeSpan], max_gap_sec: float) -> list[TimeSpan]:
    """Merge consecutive spans if the gap between them is <= max_gap_sec."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: s.start)
    out: list[TimeSpan] = [ordered[0]]
    for cur in ordered[1:]:
        prev = out[-1]
        gap = cur.start - prev.end
        if gap <= max_gap_sec and cur.start >= prev.start:
            out[-1] = TimeSpan(prev.start, max(prev.end, cur.end))
        else:
            out.append(cur)
    return out
