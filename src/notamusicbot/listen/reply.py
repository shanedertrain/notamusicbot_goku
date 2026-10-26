"""Reply synthesis for listen mode: sentence -> in-memory 48 kHz stereo PCM.

Wraps :class:`~notamusicbot.speech.pipeline.SpeechPipeline` (TTS -> RVC ->
postprocess), reads ``final.wav`` into memory and removes the job artifact, so
a spoken reply never outlives its synthesis call on disk. There is no
passthrough fallback: if the pipeline fails (RVC unavailable or anything else)
:class:`ReplySynthesisError` is raised and the CALLER says nothing (owner
decision, plan section 10). Discord-free.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import soundfile as sf

from notamusicbot.domain.speech import AudioArtifact, TtsRequest, VoiceConversionProfile
from notamusicbot.domain.voice_profiles import to_conversion_profile, to_processing_profile
from notamusicbot.speech.artifacts import FAILED_MARKER_NAME, cleanup_temp_artifact

if TYPE_CHECKING:
    from notamusicbot.domain.speech import AudioProcessingProfile
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

# Streaming gain. A stream cannot be peak-normalised (postprocess scales each
# sentence so its peak is target_peak_dbfs), so use one fixed gain instead.
# Measured live on 8 real fast-TTS (Ryan) sentences against the postprocessed
# WAV path: at +3 dB the stream RMS was a median 3.7 dB below the WAV (range
# -6.8 .. +0.8), peaks a median 1.5 dB lower; at +5 dB the median RMS was
# 0.6 dB below (per-sentence spread +-4 dB, inherent to a fixed gain), so +4 dB
# splits the difference and keeps loud peaks off the knee below. The knee
# (linear to 0.8, tanh above, never reaching full scale) keeps those peaks from
# hard clipping.
STREAM_GAIN_DB = 4.0
# Gain for the RVC stream path. RVC output is already near full scale, so the
# stream needs slightly LESS than unity vs the peak-normalised (-1 dBFS) WAV
# path. Measured 2026-10-26 on 7 real TTS sentences (unity gain): stream RMS a
# median +0.9 dB above the WAV (range +0.3 .. +1.4), peak +0.6 dB (+0.4 .. +0.8).
RVC_STREAM_GAIN_DB = -0.8
_STREAM_KNEE = 0.8
# Peak the gain was tuned for (the shipped target_peak_dbfs); a profile with a
# different target shifts the gain by the difference.
_STREAM_TUNED_PEAK_DBFS = -1.0
_STREAM_SOURCE_RATE_HZ = 24000

_WARMUP_SECONDS = 0.5
_WARMUP_SOURCE_RATE_HZ = 24000


class ReplySynthesisError(RuntimeError):
    """A reply could not be synthesized; the caller drops it (says nothing)."""


@dataclass(frozen=True, slots=True)
class ReplyAudio:
    """Playable reply: 48 kHz stereo s16le PCM, ready for ``discord.PCMAudio``."""

    pcm: bytes
    duration_s: float
    # True when the TTS length cap cut the reply short.
    truncated: bool = False


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


def _soft_limit(y: np.ndarray) -> np.ndarray:
    """Linear up to ``_STREAM_KNEE``, then a tanh knee that stays below full scale."""
    mag = np.abs(y)
    over = mag > _STREAM_KNEE
    room = 1.0 - _STREAM_KNEE
    knee = _STREAM_KNEE + room * np.tanh((mag - _STREAM_KNEE) / room)
    return np.where(over, np.sign(y) * knee, y)


class _Upsampler2x:
    """Stateful 24 kHz -> 48 kHz linear upsampler (exact factor 2, no boundary clicks).

    Output is ``x0, (x0+x1)/2, x1, ...``; the last sample of each chunk is held
    back until the next chunk (or :meth:`flush`) supplies its neighbour, so the
    stream is one continuous interpolation and N input samples give exactly 2N.
    """

    def __init__(self) -> None:
        self._held: np.float32 | None = None

    def push(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        seq = x if self._held is None else np.concatenate(([self._held], x))
        if seq.size == 0:
            return seq
        self._held = seq[-1]
        out = np.empty(2 * (seq.size - 1), dtype=np.float32)
        out[0::2] = seq[:-1]
        out[1::2] = (seq[:-1] + seq[1:]) * 0.5
        return out

    def flush(self) -> np.ndarray:
        if self._held is None:
            return np.empty(0, dtype=np.float32)
        out = np.array([self._held, self._held], dtype=np.float32)
        self._held = None
        return out


class _Pipeline(Protocol):
    async def run(
        self,
        request: TtsRequest,
        conversion: VoiceConversionProfile,
        processing: AudioProcessingProfile | None = None,
    ) -> tuple[AudioArtifact, Any]: ...


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
        streaming_tts: Any | None = None,
        tts_max_seconds_per_word: float = TTS_MAX_SECONDS_PER_WORD,
        tts_cap_floor_seconds: float = TTS_CAP_FLOOR_SECONDS,
        stream_converter: Any | None = None,
    ) -> None:
        self._tts_seconds_per_word = tts_max_seconds_per_word
        self._tts_cap_floor_s = tts_cap_floor_seconds
        self._pipeline = pipeline
        self._profile = voice_profile
        self._conversion: VoiceConversionProfile = to_conversion_profile(voice_profile)
        self._processing = to_processing_profile(
            voice_profile, sample_rate_hz=sample_rate_hz, channels=channels
        )
        # The pipeline's output_dir. Optional: when set, job directories that a
        # FAILED call leaves behind (the pipeline keeps them for diagnosis)
        # are removed, so a dropped reply leaves no files.
        self._artifact_root = artifact_root
        # The TTS synthesizer that may stream (supports_streaming / stream_pcm);
        # by default the one inside the speech pipeline.
        self._tts = streaming_tts or getattr(pipeline, "_synthesizer", None)
        # The voice converter that may stream (supports_streaming /
        # convert_stream); by default the one inside the speech pipeline.
        self._stream_converter = stream_converter or getattr(pipeline, "_converter", None)
        self._stream_gain = 10.0 ** (
            (STREAM_GAIN_DB + self._processing.target_peak_dbfs - _STREAM_TUNED_PEAK_DBFS) / 20.0
        )
        self._stream_gain_rvc = 10.0 ** (
            (RVC_STREAM_GAIN_DB + self._processing.target_peak_dbfs - _STREAM_TUNED_PEAK_DBFS)
            / 20.0
        )

    def _converter_streams(self) -> bool:
        """True when the stream converter can convert this voice's stream."""
        converter = self._stream_converter
        if converter is None or not callable(getattr(converter, "convert_stream", None)):
            return False
        supports = getattr(converter, "supports_streaming", None)
        if not callable(supports):
            return False
        try:
            return supports(self._conversion) is True
        except Exception:
            logger.warning("supports_streaming check failed", exc_info=True)
            return False

    @property
    def can_stream(self) -> bool:
        """True when :meth:`stream_sentence` is equivalent to the whole-sentence path.

        Needs a passthrough voice profile, or an RVC profile whose converter
        supports streaming, plus a plain 48 kHz stereo 1.0x postprocess profile
        and a TTS that reports streaming.
        """
        p = self._processing
        backend = self._conversion.backend
        return (
            (backend == "passthrough" or (backend == "rvc" and self._converter_streams()))
            and p.speed == 1.0
            and p.target_sample_rate_hz == REPLY_SAMPLE_RATE_HZ
            and p.channels == REPLY_CHANNELS
            and getattr(self._tts, "supports_streaming", False) is True
            and callable(getattr(self._tts, "stream_pcm", None))
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
        request = TtsRequest(
            text=text,
            language=self._profile.tts.language,
            speaker=self._profile.tts.speaker,
            max_audio_seconds=cap_s,
        )
        before = _job_dirs(self._artifact_root)
        artifact: AudioArtifact | None = None
        try:
            artifact, _metrics = await self._pipeline.run(
                request, self._conversion, self._processing
            )
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

    async def stream_sentence(self, text: str) -> AsyncIterator[bytes]:
        """Yield ``text`` as s16le 48 kHz stereo PCM chunks while the TTS is still generating.

        Only valid when :attr:`can_stream`. Applies the fixed stream gain, the
        same TTS length cap as :meth:`synthesize_sentence` (an 80 ms fade-out
        ends the cut audio) and closes the TTS generator on exit, cancel or
        error. Raises :class:`ReplySynthesisError` on any failure.
        """
        if not self.can_stream:
            msg = "streaming is not available for this voice profile / synthesizer"
            raise ReplySynthesisError(msg)
        cap_s = tts_cap_seconds(
            text,
            seconds_per_word=self._tts_seconds_per_word,
            floor_seconds=self._tts_cap_floor_s,
        )
        request = TtsRequest(
            text=text,
            language=self._profile.tts.language,
            speaker=self._profile.tts.speaker,
            max_audio_seconds=cap_s,
        )
        cap_frames = int(round(cap_s * REPLY_SAMPLE_RATE_HZ))
        fade_frames = min(int(round(TTS_CAP_FADE_OUT_S * REPLY_SAMPLE_RATE_HZ)), cap_frames)
        rvc = self._conversion.backend == "rvc"
        gain = self._stream_gain_rvc if rvc else self._stream_gain
        up = _Upsampler2x()
        emitted = 0
        truncated = False

        def encode(mono: np.ndarray) -> bytes | None:
            # Gain, cap (with the fade over the last fade_frames before the cap),
            # hard clip, mono -> stereo, s16le. None once the cap is reached.
            nonlocal emitted, truncated
            room = cap_frames - emitted
            if mono.size > room:
                # The upsampler's held tail can overshoot an exact cap by 2 frames.
                truncated = truncated or mono.size > room + 2
                mono = mono[: max(room, 0)]
            if mono.size == 0:
                return None if truncated or room <= 0 else b""
            y = mono.astype(np.float32) * gain
            if emitted + mono.size > cap_frames - fade_frames:
                # Linear 1 -> 0 ramp over the last fade_frames before the cap.
                idx = emitted + np.arange(mono.size)
                y *= np.clip((cap_frames - idx) / max(fade_frames, 1), 0.0, 1.0).astype(np.float32)
            emitted += mono.size
            pcm = np.rint(_soft_limit(y) * 32767.0).astype("<i2")
            return np.repeat(pcm, REPLY_CHANNELS).tobytes()

        try:
            if rvc:
                # Exit order matters: the converter generator closes first, then the TTS.
                async with (
                    aclosing(self._tts.stream_pcm(request)) as tts_chunks,  # type: ignore[union-attr]
                    aclosing(
                        self._stream_converter.convert_stream(tts_chunks, self._conversion)  # type: ignore[union-attr]
                    ) as converted,
                ):
                    async for chunk in converted:
                        data = encode(chunk)
                        if data:
                            yield data
                        if truncated or data is None:
                            break
            else:
                async with aclosing(self._tts.stream_pcm(request)) as chunks:  # type: ignore[union-attr]
                    async for chunk in chunks:
                        data = encode(up.push(chunk))
                        if data:
                            yield data
                        if truncated or data is None:
                            break  # cap reached: stop pulling (and close) the TTS
                    else:
                        data = encode(up.flush())
                        if data:
                            yield data
            if truncated:
                logger.warning(
                    "TTS over-generated: %d words cut to %.2fs cap", len(text.split()), cap_s
                )
        except (asyncio.CancelledError, GeneratorExit, ReplySynthesisError):
            raise
        except Exception as exc:
            logger.warning("Reply streaming failed (%s); reply dropped", type(exc).__name__)
            msg = f"reply streaming failed: {exc}"
            raise ReplySynthesisError(msg) from exc

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


async def warm_voice_converter(
    pipeline_or_router: Any, profile: VoiceProfile | VoiceConversionProfile
) -> bool:
    """Run one short conversion so the RVC model is resident before the first reply.

    Cold job 1.9 s vs warm 0.14 s (measured). There is no public warm-up API on
    the router or the pipeline, so this drives the same ``convert`` call a real
    job makes: accepts a ``VoiceConverterRouter`` / converter directly, or a
    ``SpeechPipeline`` (its converter is read from ``_converter``). Passthrough
    profiles need no warming. Returns True when a conversion ran; failures are
    logged and return False (the first real reply will then surface the error).
    When the converter also supports streaming for this profile, its
    ``warm_stream`` runs after the conversion warm-up; a failure there is
    logged and never changes the return value.
    """
    conversion = profile if hasattr(profile, "backend") else to_conversion_profile(profile)
    if conversion.backend == "passthrough":
        return False
    converter = getattr(pipeline_or_router, "_converter", pipeline_or_router)
    convert = getattr(converter, "convert", None)
    if convert is None:
        logger.warning("Warm-up skipped: %r has no convert()", type(pipeline_or_router).__name__)
        return False
    n = int(_WARMUP_SOURCE_RATE_HZ * _WARMUP_SECONDS)
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / _WARMUP_SOURCE_RATE_HZ)
    with tempfile.TemporaryDirectory(prefix="listen_warm_") as tmp:
        src_path = Path(tmp) / "warm_src.wav"
        out_path = Path(tmp) / "warm_out.wav"
        try:
            await asyncio.to_thread(sf.write, src_path, tone, _WARMUP_SOURCE_RATE_HZ, "PCM_16")
            source = AudioArtifact(
                path=src_path, sample_rate_hz=_WARMUP_SOURCE_RATE_HZ, duration_s=_WARMUP_SECONDS
            )
            await convert(source, conversion, out_path)
        except Exception:
            logger.warning("Voice converter warm-up failed", exc_info=True)
            return False
    warm_stream = getattr(converter, "warm_stream", None)
    supports = getattr(converter, "supports_streaming", None)
    if callable(warm_stream) and callable(supports):
        try:
            if supports(conversion) is True:
                await warm_stream(conversion)
        except Exception:
            logger.warning("Voice converter stream warm-up failed", exc_info=True)
    return True
