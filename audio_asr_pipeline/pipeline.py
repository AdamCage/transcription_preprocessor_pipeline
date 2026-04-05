"""High-level async transcription orchestration."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx
import numpy as np

from audio_asr_pipeline.chunking import build_chunks_from_spans, spans_to_audio_chunks
from audio_asr_pipeline.config import PipelineConfig
from audio_asr_pipeline.errors import MergeError, TranscriptionRequestError
from audio_asr_pipeline.io import extract_span_to_wav_bytes
from audio_asr_pipeline.merge import build_verbose_json_skeleton, merge_transcriptions
from audio_asr_pipeline.models import (
    AudioChunk,
    AudioFileTask,
    LabeledSegment,
    PipelineResult,
    TranscribedChunk,
)
from audio_asr_pipeline.preprocess import preprocess_audio
from audio_asr_pipeline.transcribe import VLLMTranscriptionClient

log = logging.getLogger(__name__)


def _pipeline_meta_from_coarse(
    labeled: list[LabeledSegment],
    *,
    loaded_duration: float,
    kept_spans_duration: float,
    num_chunks: int,
    config: PipelineConfig,
) -> dict:
    music = sum(1 for L in labeled if L.label == "music")
    noise = sum(1 for L in labeled if L.label == "noise")
    silence = sum(1 for L in labeled if L.label in ("silence", "unknown"))
    pct = (
        100.0 * (1.0 - kept_spans_duration / loaded_duration)
        if loaded_duration > 0
        else 0.0
    )
    return {
        "num_coarse_segments": len(labeled),
        "coarse_music_segments": music,
        "coarse_noise_segments": noise,
        "coarse_silence_or_unknown_segments": silence,
        "num_speech_chunks": num_chunks,
        "total_original_duration_sec": loaded_duration,
        "total_kept_speech_duration_sec": round(kept_spans_duration, 3),
        "percent_timeline_not_in_kept_spans_approx": round(pct, 2),
        "coarse_segmenter_backend": config.coarse_segmenter_backend,
        "vad_backend": config.vad_backend,
    }


def _stats_block(
    duration_sec: float,
    meta: dict[str, Any],
    transcription_sec: float,
) -> dict[str, Any]:
    stats = {
        "coarse_segmentation_sec": meta["timings_sec"].get("coarse_segmentation_sec", 0.0),
        "vad_sec": meta["timings_sec"].get("vad_sec", 0.0),
        "chunking_sec": meta["timings_sec"].get("chunking_sec", 0.0),
        "transcription_sec": transcription_sec,
        "preprocess_sec": (
            meta["timings_sec"].get("coarse_segmentation_sec", 0.0)
            + meta["timings_sec"].get("vad_sec", 0.0)
            + meta["timings_sec"].get("chunking_sec", 0.0)
        ),
    }
    preprocess_wall = stats["preprocess_sec"]
    if preprocess_wall > 0:
        stats["rt_audio_sec_per_preprocess_sec"] = duration_sec / preprocess_wall
    else:
        stats["rt_audio_sec_per_preprocess_sec"] = None
    if transcription_sec > 0:
        stats["rt_audio_sec_per_transcription_sec"] = duration_sec / transcription_sec
    else:
        stats["rt_audio_sec_per_transcription_sec"] = None
    return stats


class AudioTranscriptionPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        executor: Executor | None = None,
    ) -> None:
        self.config = config
        self._client = VLLMTranscriptionClient(config.vllm)
        self._file_sem = asyncio.Semaphore(config.max_concurrent_files)
        self._global_stt_sem = asyncio.Semaphore(config.max_in_flight_requests)
        if executor is None:
            self._executor: Executor = ThreadPoolExecutor(
                max_workers=max(1, config.max_concurrent_files),
            )
            self._owns_executor = True
        else:
            self._executor = executor
            self._owns_executor = False
        self._http_client: httpx.AsyncClient | None = None
        self._http_lock = asyncio.Lock()

    async def _ensure_http_client(self) -> httpx.AsyncClient:
        if self._http_client is not None:
            return self._http_client
        async with self._http_lock:
            if self._http_client is None:
                vcfg = self.config.vllm
                self._http_client = httpx.AsyncClient(
                    base_url=vcfg.base_url.rstrip("/"),
                    timeout=httpx.Timeout(
                        vcfg.request_timeout_sec,
                        connect=vcfg.connect_timeout_sec,
                    ),
                    trust_env=vcfg.trust_env,
                )
        assert self._http_client is not None
        return self._http_client

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._owns_executor:
            self._executor.shutdown(wait=True)

    async def __aenter__(self) -> AudioTranscriptionPipeline:
        await self._ensure_http_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _prepare_file_cpu(
        self,
        path: Path,
    ) -> tuple[AudioFileTask, float, list[LabeledSegment], list[AudioChunk], dict]:
        """Sync CPU path: load, segment, VAD, chunk, encode WAV bytes."""
        path = path.resolve()
        task = AudioFileTask(source_path=path, file_id=path.stem)
        fid = task.file_id
        log.info("pipeline | file_id=%s | stage=prepare_cpu_start | path=%s", fid, path)

        loaded, spans, timings, labeled = preprocess_audio(path, self.config)
        log.info(
            "pipeline | file_id=%s | stage=preprocess_summary | coarse_sec=%.3f | vad_sec=%.3f | speech_spans=%d",
            fid,
            timings.get("coarse_segmentation_sec", 0.0),
            timings.get("vad_sec", 0.0),
            len(spans),
        )

        t_chunk0 = time.perf_counter()
        chunk_spans = build_chunks_from_spans(task, spans, self.config)
        chunks = spans_to_audio_chunks(task, chunk_spans)
        loaded_duration_sec = float(loaded.duration_sec)
        samples = np.asarray(loaded.samples, dtype=np.float32)
        sr = int(loaded.sample_rate)
        total_wav_bytes = 0
        for ch in chunks:
            b, n = extract_span_to_wav_bytes(samples, sr, ch.start, ch.end)
            ch.audio_bytes = b
            ch.sample_rate = sr
            ch.num_samples = n
            total_wav_bytes += len(b) if b else 0
        del samples
        del loaded
        timings["chunking_sec"] = time.perf_counter() - t_chunk0
        log.info(
            "pipeline | file_id=%s | stage=chunking_done | chunks=%d | wav_bytes_total=%d | wall_sec=%.3f",
            fid,
            len(chunks),
            total_wav_bytes,
            timings["chunking_sec"],
        )

        kept_dur = sum(s.end - s.start for s in spans)
        meta = _pipeline_meta_from_coarse(
            labeled,
            loaded_duration=loaded_duration_sec,
            kept_spans_duration=kept_dur,
            num_chunks=len(chunks),
            config=self.config,
        )
        meta["timings_sec"] = timings

        return task, loaded_duration_sec, labeled, chunks, meta

    async def process_file(self, path: Path) -> PipelineResult:
        results = await self.process_files([path])
        return results[0]

    async def process_files(self, paths: list[Path]) -> list[PipelineResult]:
        tasks = [self._guard_file(p, self._file_sem) for p in paths]
        return await asyncio.gather(*tasks)

    async def _guard_file(self, path: Path, sem: asyncio.Semaphore) -> PipelineResult:
        async with sem:
            return await self._process_one(path)

    def _result_failed(
        self,
        *,
        path: Path,
        file_id: str,
        message: str,
        duration_sec: float = 0.0,
        meta: dict[str, Any] | None = None,
    ) -> PipelineResult:
        meta = dict(meta or {})
        meta.setdefault(
            "timings_sec",
            {"coarse_segmentation_sec": 0.0, "vad_sec": 0.0, "chunking_sec": 0.0},
        )
        skeleton = build_verbose_json_skeleton(duration=duration_sec, pipeline_meta=meta)
        skeleton["text"] = ""
        return PipelineResult(
            file_id=file_id,
            source_path=path,
            text="",
            verbose_json=skeleton,
            stats=_stats_block(duration_sec, meta, transcription_sec=0.0),
            error=message,
        )

    async def _process_one(self, path: Path) -> PipelineResult:
        path = path.resolve()
        fid = path.stem
        try:
            loop = asyncio.get_running_loop()
            task, duration_sec, _labeled, chunks, meta = await loop.run_in_executor(
                self._executor,
                self._prepare_file_cpu,
                path,
            )
        except Exception as e:
            log.exception(
                "pipeline | file_id=%s | stage=prepare_failed | path=%s",
                fid,
                path,
            )
            if self.config.fail_fast:
                raise
            return self._result_failed(path=path, file_id=fid, message=f"{type(e).__name__}: {e}")

        if not chunks:
            msg = f"No chunks to transcribe for {path}"
            if self.config.fail_fast:
                raise MergeError(msg)
            log.warning("pipeline | file_id=%s | stage=no_chunks | %s", task.file_id, msg)
            skeleton = build_verbose_json_skeleton(duration=duration_sec, pipeline_meta=meta)
            return PipelineResult(
                file_id=task.file_id,
                source_path=task.source_path,
                text="",
                verbose_json=skeleton,
                stats=_stats_block(duration_sec, meta, transcription_sec=0.0),
                error=msg,
            )

        tfid = task.file_id
        log.info(
            "pipeline | file_id=%s | stage=transcribe_start | num_chunks=%d | concurrency_cap=%d",
            tfid,
            len(chunks),
            min(self.config.max_concurrent_chunks, self.config.max_in_flight_requests),
        )
        try:
            t_tr0 = time.perf_counter()
            transcribed = await self._transcribe_all(chunks)
            transcription_sec = time.perf_counter() - t_tr0
            meta["timings_sec"]["transcription_sec"] = transcription_sec
            log.info(
                "pipeline | file_id=%s | stage=transcribe_done | chunks_ok=%d | wall_sec=%.3f",
                tfid,
                len(transcribed),
                transcription_sec,
            )

            skeleton = build_verbose_json_skeleton(duration=duration_sec, pipeline_meta=meta)
            try:
                verbose = merge_transcriptions(
                    skeleton,
                    transcribed,
                    include_words=self.config.vllm.include_word_timestamps,
                )
                nseg = len(verbose.get("segments") or [])
                log.info(
                    "pipeline | file_id=%s | stage=merge_ok | text_chars=%d | segments=%d",
                    tfid,
                    len(verbose.get("text") or ""),
                    nseg,
                )
            except MergeError as e:
                log.warning(
                    "pipeline | file_id=%s | stage=merge_failed fallback empty | reason=%s",
                    tfid,
                    e,
                )
                if self.config.fail_fast:
                    raise
                verbose = skeleton
                verbose["text"] = ""
                verbose["pipeline_meta"] = meta

            stats = _stats_block(duration_sec, meta, transcription_sec)
            return PipelineResult(
                file_id=task.file_id,
                source_path=task.source_path,
                text=verbose.get("text") or "",
                verbose_json=verbose,
                stats=stats,
                error=None,
            )
        except Exception as e:
            log.exception("pipeline | file_id=%s | stage=transcribe_or_merge_failed", tfid)
            if self.config.fail_fast:
                raise
            return self._result_failed(
                path=task.source_path,
                file_id=tfid,
                message=f"{type(e).__name__}: {e}",
                duration_sec=duration_sec,
                meta=meta,
            )

    async def _transcribe_all(
        self,
        chunks: list[AudioChunk],
    ) -> list[TranscribedChunk]:
        cfg = self.config
        chunk_sem = asyncio.Semaphore(
            min(cfg.max_concurrent_chunks, cfg.max_in_flight_requests)
        )
        http = await self._ensure_http_client()

        async def one(ch: AudioChunk) -> TranscribedChunk | None:
            async with self._global_stt_sem:
                async with chunk_sem:
                    try:
                        resp = await self._client.transcribe_chunk(ch, client=http)
                        return TranscribedChunk(
                            chunk_id=ch.chunk_id,
                            file_id=ch.file_id,
                            start_offset=ch.start,
                            end_offset=ch.end,
                            response=resp,
                        )
                    except TranscriptionRequestError:
                        if cfg.skip_failed_chunks:
                            return None
                        raise

        results: list[TranscribedChunk | None] = await asyncio.gather(*[one(c) for c in chunks])
        out = [r for r in results if r is not None]
        if not out and chunks:
            raise TranscriptionRequestError("All chunk transcriptions failed")
        return out


def _run_async_in_fresh_thread(factory):
    """Run ``asyncio.run(factory())`` in this thread, or in a worker if a loop is already running."""

    def runner():
        return asyncio.run(factory())

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return runner()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(runner).result()


def process_file_sync(path: Path | str, config: PipelineConfig) -> PipelineResult:
    p = Path(path)

    async def _main() -> PipelineResult:
        async with AudioTranscriptionPipeline(config) as pipeline:
            return await pipeline.process_file(p)

    return _run_async_in_fresh_thread(lambda: _main())


def process_files_sync(paths: list[Path | str], config: PipelineConfig) -> list[PipelineResult]:
    pp = [Path(x) for x in paths]

    async def _main() -> list[PipelineResult]:
        async with AudioTranscriptionPipeline(config) as pipeline:
            return await pipeline.process_files(pp)

    return _run_async_in_fresh_thread(lambda: _main())
