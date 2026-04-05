"""Unit tests for merge timestamp shifting."""

from audio_asr_pipeline.merge import merge_transcriptions
from audio_asr_pipeline.models import TranscribedChunk


def test_merge_shifts_segments():
    sk = {
        "text": "",
        "task": "transcribe",
        "language": None,
        "duration": 100.0,
        "segments": [],
        "words": [],
        "pipeline_meta": {},
    }
    tc = [
        TranscribedChunk(
            chunk_id="a",
            file_id="f",
            start_offset=10.0,
            end_offset=20.0,
            response={
                "text": "hello",
                "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            },
        ),
        TranscribedChunk(
            chunk_id="b",
            file_id="f",
            start_offset=25.0,
            end_offset=35.0,
            response={
                "text": "world",
                "segments": [{"start": 0.5, "end": 1.5, "text": "world"}],
            },
        ),
    ]
    out = merge_transcriptions(sk, tc)
    assert out["text"] == "hello world"
    assert len(out["segments"]) == 2
    assert out["segments"][0]["start"] == 10.0
    assert out["segments"][0]["end"] == 11.0
    assert out["segments"][1]["start"] == 25.5
    assert out["segments"][1]["end"] == 26.5


def test_merge_text_fallback_from_segments_when_chunk_text_empty():
    """vLLM may omit top-level text but fill segments[].text."""
    sk = {
        "text": "",
        "task": "transcribe",
        "language": None,
        "duration": 50.0,
        "segments": [],
        "words": [],
        "pipeline_meta": {},
    }
    tc = [
        TranscribedChunk(
            chunk_id="a",
            file_id="f",
            start_offset=0.0,
            end_offset=5.0,
            response={
                "text": "",
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "hello"},
                    {"start": 2.0, "end": 5.0, "text": "there"},
                ],
            },
        ),
        TranscribedChunk(
            chunk_id="b",
            file_id="f",
            start_offset=10.0,
            end_offset=15.0,
            response={
                "segments": [{"start": 0.0, "end": 3.0, "text": "world"}],
            },
        ),
    ]
    out = merge_transcriptions(sk, tc)
    assert out["text"] == "hello there world"
    assert len(out["segments"]) == 3


def test_merge_ms_timestamps_heuristic():
    """Large segment times relative to chunk length treated as ms."""
    sk = {
        "text": "",
        "task": "transcribe",
        "language": None,
        "duration": 60.0,
        "segments": [],
        "words": [],
        "pipeline_meta": {},
    }
    tc = [
        TranscribedChunk(
            chunk_id="a",
            file_id="f",
            start_offset=7.0,
            end_offset=37.0,
            response={
                "text": "x",
                "segments": [{"start": 0.0, "end": 5000.0, "text": "a"}],
            },
        ),
    ]
    out = merge_transcriptions(sk, tc)
    assert out["segments"][0]["start"] == 7.0
    assert out["segments"][0]["end"] == 12.0
