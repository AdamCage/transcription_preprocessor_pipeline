"""Prometheus metrics for the VAD service (OTel-compatible via /metrics scrape)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "vad_requests_total",
    "Total POST /refine requests",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "vad_request_duration_seconds",
    "End-to-end /refine latency",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

SPANS_IN = Counter(
    "vad_spans_input_total",
    "Cumulative input spans received",
)

SPANS_OUT = Counter(
    "vad_spans_output_total",
    "Cumulative output spans returned",
)

GPU_MEM_BYTES = Gauge(
    "vad_gpu_memory_bytes",
    "GPU memory currently allocated by this process",
)
