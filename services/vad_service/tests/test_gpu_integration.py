#!/usr/bin/env python
"""GPU integration benchmark for the VAD service.

Starts the service, exercises all endpoints with real audio from
``test_audio_mad``, and writes latency/throughput metrics to a JSON file
under ``tests/results/``.

Usage::

    uv run --extra dev python tests/test_gpu_integration.py
    uv run --extra dev python tests/test_gpu_integration.py --workers 4
    uv run --extra dev python tests/test_gpu_integration.py --device cpu
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

import httpx
import soundfile as sf

AUDIO_DIR = pathlib.Path(__file__).resolve().parents[2] / ".." / "test_audio_mad"
RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"

DEFAULT_PORT = 18002  # non-standard port to avoid clashing with a running service


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_prometheus(text: str) -> dict[str, float]:
    """Extract vad_* metric values from Prometheus exposition text."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "vad_" not in line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0]
            key = re.sub(r'[{}"]', "", key).replace(",", "_").replace("=", "_")
            try:
                out[key] = float(parts[1])
            except ValueError:
                pass
    return out


def _load_audio_files() -> list[pathlib.Path]:
    audio_dir = AUDIO_DIR.resolve()
    if not audio_dir.is_dir():
        sys.exit(f"Audio directory not found: {audio_dir}")
    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        sys.exit(f"No .wav files in {audio_dir}")
    return wavs


def _audio_duration(path: pathlib.Path) -> float:
    data, sr = sf.read(str(path))
    return len(data) / sr


def _refine_request(
    client: httpx.Client,
    base: str,
    wav_path: pathlib.Path,
    spans: list[dict],
) -> dict:
    """Send a single /refine request and return a result dict."""
    raw = wav_path.read_bytes()
    req_json = json.dumps({"spans": spans})
    t0 = time.perf_counter()
    r = client.post(
        f"{base}/refine",
        files={"audio": (wav_path.name, raw, "audio/wav")},
        data={"request": req_json},
        timeout=300,
    )
    latency = time.perf_counter() - t0
    body = r.json() if r.status_code == 200 else {}
    return {
        "file": wav_path.name,
        "status": r.status_code,
        "latency_sec": round(latency, 4),
        "spans_in": len(spans),
        "spans_out": len(body.get("spans", [])),
    }


async def _refine_request_async(
    client: httpx.AsyncClient,
    base: str,
    wav_path: pathlib.Path,
    spans: list[dict],
) -> dict:
    raw = wav_path.read_bytes()
    req_json = json.dumps({"spans": spans})
    t0 = time.perf_counter()
    r = await client.post(
        f"{base}/refine",
        files={"audio": (wav_path.name, raw, "audio/wav")},
        data={"request": req_json},
        timeout=300,
    )
    latency = time.perf_counter() - t0
    body = r.json() if r.status_code == 200 else {}
    return {
        "file": wav_path.name,
        "status": r.status_code,
        "latency_sec": round(latency, 4),
        "spans_in": len(spans),
        "spans_out": len(body.get("spans", [])),
    }


# ------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------

def run_sequential(client: httpx.Client, base: str, wavs: list[pathlib.Path]) -> list[dict]:
    """Each WAV sent one at a time with a single full-file span."""
    results = []
    for wav in wavs:
        dur = _audio_duration(wav)
        res = _refine_request(client, base, wav, [{"start": 0.0, "end": round(dur, 2)}])
        res["audio_duration_sec"] = round(dur, 2)
        results.append(res)
        print(f"  sequential | {wav.name} | {res['latency_sec']:.3f}s | spans_out={res['spans_out']}")
    return results


def run_multi_span(client: httpx.Client, base: str, wavs: list[pathlib.Path], n_splits: int = 4) -> list[dict]:
    """Each WAV sent with duration split into n_splits equal spans."""
    results = []
    for wav in wavs:
        dur = _audio_duration(wav)
        chunk = dur / n_splits
        spans = [
            {"start": round(i * chunk, 2), "end": round(min((i + 1) * chunk, dur), 2)}
            for i in range(n_splits)
        ]
        res = _refine_request(client, base, wav, spans)
        res["audio_duration_sec"] = round(dur, 2)
        results.append(res)
        print(f"  multi_span | {wav.name} | {res['latency_sec']:.3f}s | spans_in={n_splits} spans_out={res['spans_out']}")
    return results


async def run_concurrent(base: str, wavs: list[pathlib.Path]) -> dict:
    """All WAVs sent in parallel via asyncio.gather."""
    async with httpx.AsyncClient() as client:
        tasks = []
        for wav in wavs:
            dur = _audio_duration(wav)
            spans = [{"start": 0.0, "end": round(dur, 2)}]
            tasks.append(_refine_request_async(client, base, wav, spans))

        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_latency = time.perf_counter() - t0

    for r in results:
        print(f"  concurrent | {r['file']} | {r['latency_sec']:.3f}s | spans_out={r['spans_out']}")
    return {
        "total_latency_sec": round(total_latency, 4),
        "requests": list(results),
    }


# ------------------------------------------------------------------
# Service lifecycle
# ------------------------------------------------------------------

def _start_service(device: str, workers: int, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["VAD_DEVICE"] = device
    env["VAD_EXECUTOR_WORKERS"] = str(workers)
    env["VAD_PORT"] = str(port)
    env["VAD_LOG_LEVEL"] = "info"
    env["VAD_LOG_DIR"] = "logs"

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "vad_service.app:app",
            "--host", "127.0.0.1",
            "--port", str(port),
        ],
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


def _wait_for_health(base: str, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base}/health", timeout=5)
            if r.status_code == 200:
                return r.json()
        except httpx.ConnectError:
            pass
        time.sleep(1)
    raise TimeoutError(f"Service did not become healthy within {timeout}s")


def _stop_service(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="VAD service GPU integration benchmark")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0 or cpu)")
    parser.add_argument("--workers", type=int, default=1, help="executor_workers")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Service port")
    parser.add_argument("--url", default="", help="Skip starting service; connect to this URL instead")
    args = parser.parse_args()

    wavs = _load_audio_files()
    print(f"Found {len(wavs)} WAV files in {AUDIO_DIR.resolve()}\n")

    base = args.url.rstrip("/") if args.url else f"http://127.0.0.1:{args.port}"
    proc = None

    try:
        if not args.url:
            print(f"Starting service: device={args.device} workers={args.workers} port={args.port}")
            proc = _start_service(args.device, args.workers, args.port)
            health = _wait_for_health(base)
            print(f"Service healthy: {json.dumps(health)}\n")
        else:
            health = httpx.get(f"{base}/health", timeout=10).json()
            print(f"Connected to {base}: {json.dumps(health)}\n")

        report: dict = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "device": health.get("device", args.device),
            "executor_workers": args.workers,
            "gpu_memory_used_mb": health.get("gpu_memory_used_mb"),
            "gpu_memory_total_mb": health.get("gpu_memory_total_mb"),
            "audio_files": [w.name for w in wavs],
            "scenarios": {},
        }

        with httpx.Client() as client:
            print("--- Sequential (1 full-file span per request) ---")
            report["scenarios"]["sequential"] = run_sequential(client, base, wavs)
            print()

            print(f"--- Multi-span (4 spans per request, workers={args.workers}) ---")
            report["scenarios"]["multi_span"] = run_multi_span(client, base, wavs, n_splits=4)
            print()

        print("--- Concurrent (all files in parallel) ---")
        report["scenarios"]["concurrent"] = asyncio.run(run_concurrent(base, wavs))
        print()

        # Prometheus snapshot
        r = httpx.get(f"{base}/metrics", timeout=10)
        report["prometheus_snapshot"] = _parse_prometheus(r.text)

        # Final health (GPU memory after load)
        health_after = httpx.get(f"{base}/health", timeout=10).json()
        report["gpu_memory_after_mb"] = health_after.get("gpu_memory_used_mb")

    finally:
        if proc is not None:
            _stop_service(proc)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{ts}_w{args.workers}_{args.device.replace(':', '')}_metrics.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results written to {out_path}")

    # Summary
    print("\n=== Summary ===")
    for scenario_name, scenario_data in report["scenarios"].items():
        if isinstance(scenario_data, list):
            total = sum(r["latency_sec"] for r in scenario_data)
            print(f"  {scenario_name}: {len(scenario_data)} requests, total {total:.3f}s")
        elif isinstance(scenario_data, dict):
            reqs = scenario_data.get("requests", [])
            print(f"  {scenario_name}: {len(reqs)} requests, wall {scenario_data['total_latency_sec']:.3f}s")


if __name__ == "__main__":
    main()
