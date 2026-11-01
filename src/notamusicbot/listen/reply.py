"""Reply synthesis for listen mode: sentence -> in-memory 48 kHz stereo PCM.

Wraps :class:`~notamusicbot.speech.pipeline.SpeechPipeline` (clone TTS ->
postprocess), reads ``final.wav`` into memory and removes the job artifact, so
a spoken reply never outlives its synthesis call on disk. There is no
passthrough fallback: if the pipeline fails :class:`ReplySynthesisError` is
raised and the CALLER says nothing (owner decision, plan section 10).
``synthesize_sentence`` returns a whole clip; ``stream_sentence`` yields the PCM
while the model is still generating, so playback can start on the first piece.
Discord-free.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import shutil
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import soundfile as sf

from notamusicbot.domain.speech import AudioArtifact
from notamusicbot.domain.voice_profiles import to_processing_profile, tts_request_for
from notamusicbot.speech.artifacts import FAILED_MARKER_NAME, cleanup_temp_artifact
from notamusicbot.speech.postprocess import PeakTracker, StreamingPostprocess

if TYPE_CHECKING:
    from notamusicbot.domain.speech import AudioProcessingProfile, TtsRequest
    from notamusicbot.domain.voice_profiles import VoiceProfile

logger = logging.getLogger(__name__)

# discord.PCMAudio format: 48 kHz, 2 channels, signed 16-bit little-endian.
REPLY_SAMPLE_RATE_HZ = 48000
REPLY_CHANNELS = 2
_BYTES_PER_FRAME = 2 * REPLY_CHANNELS

# Backchannel clip trimming. Bump BACKCHANNEL_TRIM_VERSION (or change a value
# below) and every cached file gets a new key, so stale untrimmed clips are
# never reused.
BACKCHANNEL_TRIM_VERSION = 1
BACKCHANNEL_SILENCE_DBFS = -45.0
BACKCHANNEL_FADE_OUT_S = 0.08
BACKCHANNEL_MAX_S = 1.2

# TTS length cap (guards against over-generation): max(floor, words * per_word).
TTS_MAX_SECONDS_PER_WORD = 1.0
TTS_CAP_FLOOR_SECONDS = 2.0
TTS_CAP_FADE_OUT_S = 0.08

# Reply tail trim: the model rings down after the last word (a low whine on the
# clone); anything below this level after the last loud frame is faded out.
REPLY_TAIL_SILENCE_DBFS = -45.0
REPLY_TAIL_FADE_S = 0.08


class ReplySynthesisError(RuntimeError):
    """A reply could not be synthesized; the caller drops it (says nothing)."""


@dataclass(frozen=True, slots=True)
class ReplyAudio:
    """Playable reply: 48 kHz stereo s16le PCM, ready for ``discord.PCMAudio``."""

    pcm: bytes
    duration_s: float
    # True when the TTS length cap cut the reply short.
    truncated: bool = False


_BYTES_PER_SECOND = REPLY_SAMPLE_RATE_HZ * _BYTES_PER_FRAME


class StreamedReply:
    """One chunk's PCM that is still being synthesized.

    The producer ``push``es pieces and calls ``end`` (always, even on failure);
    the player iterates ``pieces``. ``duration_s`` grows as audio arrives.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.duration_s = 0.0

    def push(self, pcm: bytes) -> None:
        self.duration_s += len(pcm) / _BYTES_PER_SECOND
        self._queue.put_nowait(pcm)

    def end(self) -> None:
        self._queue.put_nowait(None)

    async def pieces(self) -> AsyncIterator[bytes]:
        while (piece := await self._queue.get()) is not None:
            yield piece


def tts_cap_seconds(
    text: str,
    *,
    seconds_per_word: float = TTS_MAX_SECONDS_PER_WORD,
    floor_seconds: float = TTS_CAP_FLOOR_SECONDS,
) -> float:
    """Audio-length cap for ``text``: ``max(floor, words * seconds_per_word)``."""
    return max(floor_seconds, len(text.split()) * seconds_per_word)


def _cap_audio(
    audio: ReplyAudio, cap_s: float, *, fade_out_s: float = TTS_CAP_FADE_OUT_S
) -> ReplyAudio:
    """Cut ``audio`` at ``cap_s`` with a linear fade-out; flag it truncated.

    Audio at or under the cap (within one frame) is returned unchanged.
    """
    cap_frames = int(round(cap_s * REPLY_SAMPLE_RATE_HZ))
    frames = np.frombuffer(audio.pcm, dtype="<i2").reshape(-1, REPLY_CHANNELS)
    if frames.shape[0] <= cap_frames + 1:
        return audio
    clip = frames[:cap_frames].astype(np.float64)
    fade = min(int(round(fade_out_s * REPLY_SAMPLE_RATE_HZ)), cap_frames)
    if fade > 0:
        clip[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
    pcm = np.ascontiguousarray(np.rint(clip), dtype="<i2").tobytes()
    return ReplyAudio(pcm=pcm, duration_s=cap_frames / REPLY_SAMPLE_RATE_HZ, truncated=True)


class _Pipeline(Protocol):
    def stream(self, request: TtsRequest) -> AsyncIterator[tuple[Any, int]]: ...

    async def run(
        self,
        request: TtsRequest,
        processing: AudioProcessingProfile | None = None,
    ) -> tuple[AudioArtifact, Any]: ...


def _within_cap(pcm: bytes, sent: int, cap_frames: int) -> tuple[bytes, int, bool]:
    """Trim ``pcm`` so the reply holds at most ``cap_frames`` frames.

    Returns (pcm to emit, total frames after it, whether the cap cut it). A cut
    piece gets the same linear fade-out as :func:`_cap_audio`.
    """
    frames = len(pcm) // _BYTES_PER_FRAME
    if sent + frames <= cap_frames:
        return pcm, sent + frames, False
    keep = max(0, cap_frames - sent)
    clip = np.frombuffer(pcm, dtype="<i2").reshape(-1, REPLY_CHANNELS)[:keep].astype(np.float64)
    fade = min(int(round(TTS_CAP_FADE_OUT_S * REPLY_SAMPLE_RATE_HZ)), keep)
    if fade > 0:
        clip[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
    return np.ascontiguousarray(np.rint(clip), dtype="<i2").tobytes(), cap_frames, True


def _trim_leading_silence(frames: np.ndarray, *, silence_dbfs: float) -> np.ndarray:
    """Drop leading near-silence: the model's warm-up hiss before the first word.

    Keeps a short natural lead-in (the first loud frame minus 120 ms) so words
    do not start abrubtly against the previous chunk.
    """
    threshold = 32768.0 * 10.0 ** (silence_dbfs / 20.0)
    loud = np.flatnonzero(np.abs(frames.astype(np.int32)).max(axis=1) >= threshold)
    if loud.size == 0:
        return frames
    start = max(0, int(loud[0]) - int(0.12 * REPLY_SAMPLE_RATE_HZ))
    return frames[start:]


def _trim_tail_silence(frames: np.ndarray, *, silence_dbfs: float, fade_s: float) -> np.ndarray:
    """Drop trailing near-silence (TTS ring-down / whine) with a short fade.

    Keeps at least one frame. Operates on a float copy; caller converts back.
    """
    threshold = 32768.0 * 10.0 ** (silence_dbfs / 20.0)
    loud = np.flatnonzero(np.abs(frames.astype(np.int32)).max(axis=1) >= threshold)
    keep = int(loud[-1]) + 1 if loud.size else frames.shape[0]
    if keep >= frames.shape[0]:
        return frames
    faded = frames.copy().astype(np.float64)
    fade = min(int(round(fade_s * REPLY_SAMPLE_RATE_HZ)), frames.shape[0] - keep)
    if fade > 0:
        faded[keep : keep + fade] *= np.linspace(1.0, 0.0, fade)[:, None]
    return faded[: keep + fade]


def _read_pcm(path: Path) -> ReplyAudio:
    """Load a final WAV as s16le stereo 48 kHz bytes (blocking; run in a thread)."""
    data, rate = sf.read(path, dtype="int16", always_2d=True)
    if rate != REPLY_SAMPLE_RATE_HZ:
        msg = f"final.wav is {rate} Hz, expected {REPLY_SAMPLE_RATE_HZ} Hz"
        raise ReplySynthesisError(msg)
    if data.shape[1] == 1:
        data = np.repeat(data, REPLY_CHANNELS, axis=1)
    elif data.shape[1] != REPLY_CHANNELS:
        msg = f"final.wav has {data.shape[1]} channels, expected {REPLY_CHANNELS}"
        raise ReplySynthesisError(msg)
    data = _trim_tail_silence(
        data, silence_dbfs=REPLY_TAIL_SILENCE_DBFS, fade_s=REPLY_TAIL_FADE_S
    )
    data = _trim_leading_silence(data, silence_dbfs=REPLY_TAIL_SILENCE_DBFS)
    data = data.astype("<i2")
    pcm = np.ascontiguousarray(data, dtype="<i2").tobytes()
    return ReplyAudio(pcm=pcm, duration_s=data.shape[0] / REPLY_SAMPLE_RATE_HZ)


def _job_dirs(root: Path | None) -> set[Path]:
    if root is None or not root.is_dir():
        return set()
    return {p for p in root.iterdir() if p.is_dir()}


class ReplySynthesizer:
    """Turns one sentence into :class:`ReplyAudio` through the speech pipeline."""

    def __init__(
        self,
        pipeline: _Pipeline,
        voice_profile: VoiceProfile,
        *,
        sample_rate_hz: int = REPLY_SAMPLE_RATE_HZ,
        channels: int = REPLY_CHANNELS,
        artifact_root: Path | None = None,
        tts_max_seconds_per_word: float = TTS_MAX_SECONDS_PER_WORD,
        tts_cap_floor_seconds: float = TTS_CAP_FLOOR_SECONDS,
    ) -> None:
        self._tts_seconds_per_word = tts_max_seconds_per_word
        self._tts_cap_floor_s = tts_cap_floor_seconds
        self._pipeline = pipeline
        self._profile = voice_profile
        self._processing = to_processing_profile(
            voice_profile, sample_rate_hz=sample_rate_hz, channels=channels
        )
        # The pipeline's output_dir. Optional: when set, job directories that a
        # FAILED call leaves behind (the pipeline keeps them for diagnosis)
        # are removed, so a dropped reply leaves no files.
        self._artifact_root = artifact_root
        self._peaks = PeakTracker()

    @property
    def can_stream(self) -> bool:
        """True when replies can be played while still being generated."""
        pipeline = self._pipeline
        return (
            bool(getattr(pipeline, "can_stream", False))
            and self._processing.speed == 1.0  # the time-stretch needs the whole clip
            and self._processing.target_sample_rate_hz == REPLY_SAMPLE_RATE_HZ
            and self._processing.channels == REPLY_CHANNELS
        )

    @property
    def profile_id(self) -> str:
        return self._profile.id

    async def synthesize_sentence(self, text: str) -> ReplyAudio:
        """Synthesize ``text``; raise :class:`ReplySynthesisError` on any failure."""
        cap_s = tts_cap_seconds(
            text,
            seconds_per_word=self._tts_seconds_per_word,
            floor_seconds=self._tts_cap_floor_s,
        )
        request = tts_request_for(self._profile, text, max_audio_seconds=cap_s)
        before = _job_dirs(self._artifact_root)
        artifact: AudioArtifact | None = None
        try:
            artifact, _metrics = await self._pipeline.run(request, self._processing)
            audio = await asyncio.to_thread(_read_pcm, artifact.path)
            capped = _cap_audio(audio, cap_s)
            if capped.truncated:
                logger.warning(
                    "TTS over-generated: %d words, %.2fs audio cut to %.2fs cap",
                    len(text.split()),
                    audio.duration_s,
                    cap_s,
                )
            return capped
        except asyncio.CancelledError:
            raise
        except ReplySynthesisError:
            raise
        except Exception as exc:
            logger.warning("Reply synthesis failed (%s); reply dropped", type(exc).__name__)
            msg = f"reply synthesis failed: {exc}"
            raise ReplySynthesisError(msg) from exc
        finally:
            if artifact is not None:
                await asyncio.to_thread(cleanup_temp_artifact, artifact.path)
            await asyncio.to_thread(self._remove_failed_jobs, before)

    async def stream_sentence(self, text: str) -> AsyncGenerator[bytes, None]:
        """Yield ``text`` as 48 kHz stereo s16le pieces while it is being generated.

        Same length cap and failure contract as :meth:`synthesize_sentence`
        (:class:`ReplySynthesisError`; cancellation propagates). Leaving the
        iterator early stops generation and frees the GPU.
        """
        cap_s = tts_cap_seconds(
            text,
            seconds_per_word=self._tts_seconds_per_word,
            floor_seconds=self._tts_cap_floor_s,
        )
        request = tts_request_for(self._profile, text, max_audio_seconds=cap_s)
        cap_frames = int(round(cap_s * REPLY_SAMPLE_RATE_HZ))
        post = StreamingPostprocess(self._processing, self._peaks)
        source = self._pipeline.stream(request)
        sent = 0
        head_trimmed = False
        started = time.perf_counter()
        try:
            async for audio, sample_rate in source:
                pcm = post.push(audio, sample_rate)
                if not pcm:
                    continue
                if sent == 0:
                    logger.info("TTS first audio after %.2fs", time.perf_counter() - started)
                piece, sent, capped = _within_cap(pcm, sent, cap_frames)
                if piece and not head_trimmed:
                    # The first piece carries the model's warm-up hiss; trim it
                    # exactly once, same rule as the whole-clip path.
                    head_trimmed = True
                    frames = np.frombuffer(piece, dtype="<i2").reshape(-1, REPLY_CHANNELS)
                    trimmed = _trim_leading_silence(
                        frames, silence_dbfs=REPLY_TAIL_SILENCE_DBFS
                    ).astype("<i2")
                    piece = np.ascontiguousarray(trimmed).tobytes()
                if piece:
                    yield piece
                if capped:
                    logger.warning(
                        "TTS over-generated: %d words cut to the %.2fs cap",
                        len(text.split()),
                        cap_s,
                    )
                    return
            tail, sent, _capped = _within_cap(post.finish(), sent, cap_frames)
            if tail:
                # Same tail trim as the whole-clip path: fade the ring-down after
                # the last loud frame instead of playing it as a whine.
                frames = np.frombuffer(tail, dtype="<i2").reshape(-1, REPLY_CHANNELS)
                trimmed = _trim_tail_silence(
                    frames, silence_dbfs=REPLY_TAIL_SILENCE_DBFS, fade_s=REPLY_TAIL_FADE_S
                ).astype("<i2")
                yield np.ascontiguousarray(trimmed).tobytes()
        except asyncio.CancelledError:
            raise
        except ReplySynthesisError:
            raise
        except Exception as exc:
            logger.warning("Reply synthesis failed (%s); reply dropped", type(exc).__name__)
            msg = f"reply synthesis failed: {exc}"
            raise ReplySynthesisError(msg) from exc
        finally:
            await source.aclose()  # type: ignore[attr-defined]

    def _remove_failed_jobs(self, before: set[Path]) -> None:
        for job_dir in _job_dirs(self._artifact_root) - before:
            if (job_dir / FAILED_MARKER_NAME).is_file():
                shutil.rmtree(job_dir, ignore_errors=True)


def trim_backchannel(
    audio: ReplyAudio,
    *,
    silence_dbfs: float = BACKCHANNEL_SILENCE_DBFS,
    fade_out_s: float = BACKCHANNEL_FADE_OUT_S,
    max_s: float = BACKCHANNEL_MAX_S,
) -> ReplyAudio:
    """Cut leading / trailing silence, cap the length and fade the end out.

    Silence is every frame whose peak (over both channels) is below
    ``silence_dbfs``. An all-silent clip comes back unchanged.
    """
    frames = np.frombuffer(audio.pcm, dtype="<i2").reshape(-1, REPLY_CHANNELS)
    threshold = 32768.0 * 10.0 ** (silence_dbfs / 20.0)
    loud = np.flatnonzero(np.abs(frames.astype(np.int32)).max(axis=1) >= threshold)
    if loud.size == 0:
        return audio
    clip = frames[loud[0] : loud[-1] + 1]
    clip = clip[: int(round(max_s * REPLY_SAMPLE_RATE_HZ))]
    fade = min(int(round(fade_out_s * REPLY_SAMPLE_RATE_HZ)), clip.shape[0])
    out = clip.astype(np.float64)
    if fade > 0:
        out[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
    pcm = np.ascontiguousarray(np.rint(out), dtype="<i2").tobytes()
    return ReplyAudio(pcm=pcm, duration_s=clip.shape[0] / REPLY_SAMPLE_RATE_HZ)


class BackchannelCache:
    """Pre-rendered short acknowledgements ("Hm?", "Oh yeah?") for instant feedback.

    Files live under ``directory`` keyed by a hash of (text, voice profile id,
    trim version and parameters), so a changed phrase, voice or trim setting
    re-renders and an unchanged one never calls the pipeline again. Each stored
    clip is already trimmed (silence cut, 80 ms fade-out, 1.2 s cap).
    """

    def __init__(
        self, directory: Path, texts: list[str] | tuple[str, ...], synthesizer: ReplySynthesizer
    ) -> None:
        self._directory = Path(directory)
        self._texts = tuple(texts)
        self._synthesizer = synthesizer
        self._clips: dict[str, ReplyAudio] = {}

    def _path_for(self, text: str) -> Path:
        trim = (
            f"v{BACKCHANNEL_TRIM_VERSION}:{BACKCHANNEL_SILENCE_DBFS}:"
            f"{BACKCHANNEL_FADE_OUT_S}:{BACKCHANNEL_MAX_S}"
        )
        raw = f"{text}\x00{self._synthesizer.profile_id}\x00{trim}"
        key = hashlib.sha256(raw.encode()).hexdigest()
        return self._directory / f"{key[:32]}.pcm"

    async def prerender(self) -> int:
        """Load or render every phrase; return how many were newly rendered.

        A phrase whose synthesis fails is skipped (logged), not fatal.
        """
        rendered = 0
        for text in self._texts:
            path = self._path_for(text)
            cached = await asyncio.to_thread(self._load, path)
            if cached is None:
                try:
                    cached = trim_backchannel(await self._synthesizer.synthesize_sentence(text))
                except ReplySynthesisError:
                    logger.warning("Backchannel %r could not be rendered; skipped", text)
                    continue
                await asyncio.to_thread(self._store, path, cached)
                rendered += 1
            self._clips[text] = cached
        return rendered

    @staticmethod
    def _load(path: Path) -> ReplyAudio | None:
        try:
            pcm = path.read_bytes()
        except OSError:
            return None
        if not pcm or len(pcm) % _BYTES_PER_FRAME:
            return None
        return ReplyAudio(pcm=pcm, duration_s=len(pcm) / _BYTES_PER_FRAME / REPLY_SAMPLE_RATE_HZ)

    @staticmethod
    def _store(path: Path, audio: ReplyAudio) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(audio.pcm)
        tmp.replace(path)

    def __len__(self) -> int:
        return len(self._clips)

    def get_random(self, rng: random.Random | None = None) -> ReplyAudio:
        """A random pre-rendered clip; raises LookupError when none is ready."""
        if not self._clips:
            msg = "no backchannel clips rendered"
            raise LookupError(msg)
        return (rng or random).choice(list(self._clips.values()))
