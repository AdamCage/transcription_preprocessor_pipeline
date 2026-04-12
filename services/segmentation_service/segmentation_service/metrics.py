"""Prometheus metrics for the segmentation service."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

INFERENCE_DURATION = Histogram(
    "segmentation_inference_duration_seconds",
    "Time spent on GPU inference (excludes I/O decode).",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)

REQUESTS_TOTAL = Counter(
    "segmentation_requests_total",
    "Total /segment requests by status.",
    ["status"],
)

AUDIO_DURATION = Histogram(
    "segmentation_audio_duration_seconds",
    "Duration of input audio files.",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
)

GPU_MEMORY_USED = Gauge(
    "segmentation_gpu_memory_used_bytes",
    "GPU memory currently allocated by PyTorch.",
)

SEMAPHORE_WAITERS = Gauge(
    "segmentation_semaphore_waiters",
    "Number of requests waiting on the concurrency semaphore.",
)
