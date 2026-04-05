"""Core dataclasses / types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

SegmentLabel = Literal["speech", "music", "noise", "silence", "unknown"]


@dataclass(frozen=True)
class AudioFileTask:
    source_path: Path
    file_id: str
    language: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class TimeSpan:
    start: float
    end: float


@dataclass
class LabeledSegment:
    start: float
    end: float
    label: SegmentLabel
    score: float | None = None


@dataclass
class AudioChunk:
    chunk_id: str
    file_id: str
    start: float
    end: float
    audio_bytes: bytes | None
    sample_rate: int
    num_samples: int
    source_path: Path | None = None


@dataclass
class TranscribedChunk:
    chunk_id: str
    file_id: str
    start_offset: float
    end_offset: float
    response: dict[str, Any]


@dataclass
class PipelineResult:
    file_id: str
    source_path: Path
    text: str
    verbose_json: dict[str, Any]
    stats: dict[str, Any] = field(default_factory=dict)
    # If set, the file failed; text/stats may be empty or partial.
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class LoadedAudio:
    """Mono float32 waveform and metadata after normalization."""

    samples: NDArray[np.float32]
    sample_rate: int
    source_path: Path
    duration_sec: float
