"""Typed errors for the pipeline."""


class AudioAsrPipelineError(Exception):
    """Base class for pipeline errors."""


class AudioLoadError(AudioAsrPipelineError):
    """Failed to load or decode audio."""


class SegmentationError(AudioAsrPipelineError):
    """Coarse segmentation failed."""


class VADProcessingError(AudioAsrPipelineError):
    """VAD refinement failed."""


class TranscriptionRequestError(AudioAsrPipelineError):
    """STT HTTP request failed after retries."""


class MergeError(AudioAsrPipelineError):
    """Merging transcription chunks failed."""
