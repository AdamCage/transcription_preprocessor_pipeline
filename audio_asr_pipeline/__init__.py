"""Audio preprocessing and OpenAI-compatible chunked STT pipeline."""

from audio_asr_pipeline.config import PipelineConfig, VLLMTranscribeConfig
from audio_asr_pipeline.errors import (
    AudioAsrPipelineError,
    AudioLoadError,
    MergeError,
    SegmentationError,
    TranscriptionRequestError,
    VADProcessingError,
)
from audio_asr_pipeline.merge import (
    build_verbose_json_skeleton,
    merge_transcriptions,
)
from audio_asr_pipeline.models import (
    AudioChunk,
    AudioFileTask,
    LabeledSegment,
    LoadedAudio,
    PipelineResult,
    TimeSpan,
    TranscribedChunk,
)
from audio_asr_pipeline.pipeline import (
    AudioTranscriptionPipeline,
    process_file_sync,
    process_files_sync,
)
from audio_asr_pipeline.preprocess import preprocess_audio

__all__ = [
    "AudioAsrPipelineError",
    "AudioChunk",
    "AudioFileTask",
    "AudioLoadError",
    "AudioTranscriptionPipeline",
    "LabeledSegment",
    "LoadedAudio",
    "MergeError",
    "PipelineConfig",
    "PipelineResult",
    "SegmentationError",
    "TimeSpan",
    "TranscribedChunk",
    "TranscriptionRequestError",
    "VADProcessingError",
    "VLLMTranscribeConfig",
    "build_verbose_json_skeleton",
    "merge_transcriptions",
    "preprocess_audio",
    "process_file_sync",
    "process_files_sync",
]
