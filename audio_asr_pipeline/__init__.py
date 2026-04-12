"""Audio preprocessing and OpenAI-compatible chunked STT pipeline."""

from audio_asr_pipeline.config import (
    CoarseBackendName,
    GemmaApiStyle,
    PipelineConfig,
    STTBackendName,
    VADBackendName,
    VLLMTranscribeConfig,
)
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
from audio_asr_pipeline.remote_clients import RemoteSegmentationClient, RemoteVADClient

__all__ = [
    "AudioAsrPipelineError",
    "CoarseBackendName",
    "GemmaApiStyle",
    "AudioChunk",
    "AudioFileTask",
    "AudioLoadError",
    "AudioTranscriptionPipeline",
    "LabeledSegment",
    "LoadedAudio",
    "MergeError",
    "PipelineConfig",
    "PipelineResult",
    "RemoteSegmentationClient",
    "RemoteVADClient",
    "STTBackendName",
    "SegmentationError",
    "TimeSpan",
    "TranscribedChunk",
    "TranscriptionRequestError",
    "VADBackendName",
    "VADProcessingError",
    "VLLMTranscribeConfig",
    "build_verbose_json_skeleton",
    "merge_transcriptions",
    "preprocess_audio",
    "process_file_sync",
    "process_files_sync",
]
