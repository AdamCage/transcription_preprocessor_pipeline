"""Load and normalize audio; extract chunk bytes."""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from audio_asr_pipeline.errors import AudioLoadError
from audio_asr_pipeline.models import LoadedAudio


def load_normalized_audio(
    path: Path,
    *,
    target_sr: int = 16000,
    mono: bool = True,
) -> LoadedAudio:
    """Load file with librosa (ffmpeg-backed formats), mono, target sample rate."""
    try:
        y, sr = librosa.load(str(path), sr=target_sr, mono=mono)
    except Exception as e:  # noqa: BLE001
        raise AudioLoadError(f"Failed to load audio: {path}") from e
    if y.size == 0:
        raise AudioLoadError(f"Empty audio: {path}")
    duration = float(len(y) / target_sr)
    return LoadedAudio(
        samples=np.asarray(y, dtype=np.float32),
        sample_rate=target_sr,
        source_path=path.resolve(),
        duration_sec=duration,
    )


def extract_span_to_wav_bytes(
    samples: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> tuple[bytes, int]:
    """Slice [start_sec, end_sec) and encode as WAV PCM_16 in memory."""
    n = len(samples)
    i0 = max(0, int(start_sec * sample_rate))
    i1 = min(n, int(end_sec * sample_rate))
    if i1 <= i0:
        segment = np.zeros(1, dtype=np.float32)
    else:
        segment = samples[i0:i1]
    # peak normalize slightly to avoid clipping when converting to int16
    peak = float(np.max(np.abs(segment))) if segment.size else 0.0
    if peak > 1.0:
        segment = segment / peak
    pcm = (np.clip(segment, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue(), int(pcm.shape[0])


def split_stereo_channels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Split a 2-channel waveform into left/right float32 1-D arrays.

    librosa may return stereo as ``(n_samples, 2)`` or ``(2, n_samples)`` (channels first)
    depending on version/build; this accepts both.
    """
    if y.ndim != 2:
        raise AudioLoadError(f"stereo requires 2-D array, got shape {y.shape}")
    s0, s1 = int(y.shape[0]), int(y.shape[1])
    if s0 == 2 and s1 > s0:
        ch0 = np.asarray(y[0], dtype=np.float32)
        ch1 = np.asarray(y[1], dtype=np.float32)
        n = s1
    elif s1 == 2 and s0 > s1:
        ch0 = np.asarray(y[:, 0], dtype=np.float32)
        ch1 = np.asarray(y[:, 1], dtype=np.float32)
        n = s0
    elif s0 == 2 and s1 == 2:
        ch0 = np.asarray(y[:, 0], dtype=np.float32)
        ch1 = np.asarray(y[:, 1], dtype=np.float32)
        n = s0
    else:
        raise AudioLoadError(
            f"expected exactly 2 channels (shape (n,2) or (2,n)); got {y.shape}"
        )
    return ch0, ch1, n


def write_mono_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write float32 mono waveform in [-1, 1] as WAV PCM_16 on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seg = np.asarray(samples, dtype=np.float32).ravel()
    peak = float(np.max(np.abs(seg))) if seg.size else 0.0
    if peak > 1.0:
        seg = seg / peak
    pcm = (np.clip(seg, -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(str(path), pcm, sample_rate, format="WAV", subtype="PCM_16")
