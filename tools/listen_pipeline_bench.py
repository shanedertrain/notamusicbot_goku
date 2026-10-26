"""P10-B: end-to-end listen-turn latency with the REAL components (no Discord).

Chain, as Application wires it, with a paced fake packet source and a fake
player at the two Discord edges:

    PacedPacketSource (20 ms packets, REAL TIME) -> ListenIngress (pass-through
    decoder) -> Endpointer + SileroVad (own thread) -> TurnManager ->
    Parakeet TDT 0.6B int8 (CPU, defaults, build_transcriber) ->
    the production listen chat provider (Application.listen_content_provider:
    FallbackChatProvider = SparkChatProvider over the loopback tunnel, streamed
    into StreamChunker caps 8/12; the in-process Qwen3-4B is NOT loaded) ->
    ReplySynthesizer (Qwen3-TTS + RVC goku, rvc_unavailable_policy error,
    artifact_root set) -> RecordingPlayer.

Local only: ``enforce_offline_models()`` runs before any model loads, nothing
is downloaded; the only network is the loopback Spark tunnel (127.0.0.1:8000).
Stops with an error if GET /v1/models fails. After the 30 turns a separate
FALLBACK DRILL points the provider at 127.0.0.1:9 (nothing listens), expects
one WARNING and a lazy in-process load, then checks that a healthy Spark makes
``ensure_spark_preferred`` unload it, and that a non-loopback base_url is
refused before any request. User audio is synthetic TTS held in
memory; the pre-rendered backchannel clips go under a temp dir in ``var/``
that is deleted afterwards. Needs the models in the local HF cache and the RVC
runtime (same as tools/listen_bench_models.py). Not part of pytest.

    .venv-main/bin/python tools/listen_pipeline_bench.py [--out PATH] [--turns 30]

Per turn (from the TurnManager's own metrics line, 10 ms resolution):
eos->asr, asr->llm, llm->tts1, tts1->audio and total (end of speech to first
real reply audio). "End of speech" is the last voiced frame, so eos->asr
includes the endpointer's 600 ms silence wait. The first backchannel time is
read from the recording player. Percentiles are nearest rank (see
tools/listen_bench.py). nvidia-smi memory.used is sampled every 250 ms for the
whole run in a thread.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from listen_bench import percentile  # noqa: E402

from notamusicbot.listen.offline import enforce_offline_models  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "evidence" / "listen_turn_latency_n20_fast.json"
PRIOR_AFTER_LOAD_MIB = 11802  # evidence/listen_turn_latency_n30.json (Qwen3-4B resident)
PRIOR_BASELINE_MIB = 1676

QUESTIONS = [
    "Hey Goku, what's your favorite food, and how much of it can you eat?",
    "Goku, who was the toughest opponent you have ever fought?",
    "Hey Goku, how do you stay so strong while training every single day?",
    "What would you do if you had one whole day with nothing to fight?",
    "Hey Goku, can you tell me about your first time turning Super Saiyan?",
    "Goku, what is the hardest thing you have ever had to learn?",
    "Hey Goku, do you ever get nervous before a really big fight?",
    "Tell me, Goku, what was it like training with Master Roshi as a kid?",
]
# Emoji / markdown that must never reach TTS (checked on every synthesize_sentence call).
_EMOJI_RE = re.compile("[\U0001f000-\U0001ffff\u2600-\u27bf\u2b00-\u2bff\ufe0f\u200d]")
_MARKDOWN_RE = re.compile(r"[*_`#~]|^\s*[-+>]\s|\[[^\]]*\]\([^)]*\)", re.MULTILINE)
BOT_ID = 999
USER_ID = 1
STRANGER_ID = 2
STREAM_PREBUFFER_S = 0.12  # VoicePlayer.PREBUFFER_FRAMES (6) x 20 ms
PACKET_FRAMES = 960  # 20 ms at 48 kHz
RATE_48K = 48000
LEAD_SILENCE_S = 0.3
TAIL_SILENCE_S = 1.0  # must exceed vad.end_silence_ms (600 ms) for endpointing
SMI_PERIOD_S = 0.25
STAGES = ("eos->asr", "asr->llm", "llm->tts1", "tts1->audio", "total")
STAGE_LABELS = {
    "eos->asr": "asr (end of speech to transcript)",
    "asr->llm": "time to first chunk (transcript to first chunk text)",
    "llm->tts1": "tts first chunk (first chunk text to its audio)",
    "tts1->audio": "audio (audio ready to playback start)",
    "total": "end of speech to first reply audio",
}
SPARK_MODELS_URL = "http://127.0.0.1:8000/v1/models"
BUDGET = {"total_p50_s": 6.0, "total_p90_s": 9.0, "backchannel_s": 0.8}

NOTE = (
    "Input is synthetic clean TTS speech fed in real time (20 ms packets, one "
    "thread), not real microphones. Replies are played in real time by the fake "
    "player (sleep for the PCM duration), so a turn takes its full audio length. "
    "Reply path is the production TurnManager: streamed Spark text through "
    "FallbackChatProvider (listen.llm defaults, model auto), StreamChunker caps "
    "from listen.reply (8/12), the TTS length cap, the early backchannel and the "
    "trimmed BackchannelCache; max_new_tokens comes from "
    "listen.reply.max_new_tokens via the production provider (no override). The "
    "in-process LLM is not loaded at startup. Stage keys keep their old names but "
    "now mean: asr->llm = ASR done to first chunk text ready (time to first chunk);"
    " llm->tts1 = first chunk ready to its TTS+RVC audio ready; tts1->audio = audio"
    " ready to playback start (waits for the backchannel clip to finish); total = "
    "end of speech to first reply audio. Loaders get hub ids under "
    "HF_HUB_OFFLINE=1."
)


def smi_used_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return int(out.strip().splitlines()[0])


class VramSampler:
    """Samples nvidia-smi memory.used every 250 ms, tagged with the run phase."""

    def __init__(self) -> None:
        self.phase = "baseline"
        self.samples: list[tuple[float, int, str]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vram-sampler", daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.samples.append((started, smi_used_mib(), self.phase))
            except Exception:  # sampling must never kill the run
                pass
            self._stop.wait(max(0.0, SMI_PERIOD_S - (time.monotonic() - started)))

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(5.0)

    def peak(self, phase: str | None = None) -> int | None:
        values = [u for _, u, p in self.samples if phase is None or p == phase]
        return max(values) if values else None

    def summary(self) -> dict:
        phases = sorted(
            {p for _, _, p in self.samples},
            key=lambda p: next(t for t, _, q in self.samples if q == p),
        )
        return {
            "period_s": SMI_PERIOD_S,
            "n_samples": len(self.samples),
            "peak_overall_mib": self.peak(),
            "per_phase": {
                p: {
                    "n": sum(1 for _, _, q in self.samples if q == p),
                    "peak_mib": self.peak(p),
                    "min_mib": min(u for _, u, q in self.samples if q == p),
                }
                for p in phases
            },
        }


class PassThroughDecoder:
    """Packet payload already is 48 kHz stereo s16 PCM."""

    def __init__(self, counter: list[int]) -> None:
        self._counter = counter

    def decode(self, data: bytes) -> bytes:
        self._counter[0] += 1
        return data


class PacedPacketSource:
    """Delivers one user's 20 ms packets on an absolute 20 ms schedule (real time)."""

    def __init__(self, user_id: int, payloads: list[bytes], realtime: bool = True) -> None:
        self.user_id = user_id
        self.payloads = payloads
        self.realtime = realtime

    def deliver(self, callback) -> int:  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        for i, payload in enumerate(self.payloads):
            if self.realtime:
                wait = start + i * 0.02 - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            callback(self.user_id, payload)
        return len(self.payloads)


class RecordingPlayer:
    """ReplyPlayerPort fake: records PCM + start time, then 'plays' it in real time."""

    def __init__(self) -> None:
        self.plays: list[tuple[float, bytes]] = []
        self.ends: list[float] = []  # monotonic time each play() returned
        self.stop_calls = 0
        self.streams: list[dict] = []  # one entry per play_stream() call

    async def play(self, pcm: bytes) -> None:
        self.plays.append((time.monotonic(), pcm))
        try:
            await asyncio.sleep(len(pcm) / (RATE_48K * 4))
        finally:
            # Always append (also on a barge-in cancel) so ends[k] stays plays[k].
            self.ends.append(time.monotonic())

    async def play_stream(self, chunks) -> None:  # type: ignore[no-untyped-def]
        """Consume a PCM stream the way VoicePlayer does, in real time.

        Models the production player: playback starts once STREAM_PREBUFFER_S of
        audio is buffered (or the stream ends), then runs at 1x. A chunk that
        arrives after the buffered audio has been played out is an UNDERRUN
        (production pads it with silence). Records the first-chunk time, total
        audio seconds and the gaps between chunk arrivals; ``plays``/``ends``
        get one entry per reply so the per-turn code works unchanged.
        """
        t_first: float | None = None
        t_prev: float | None = None
        t_start: float | None = None
        play_end = 0.0
        buffered_s = 0.0
        total_bytes = 0
        parts: list[bytes] = []
        gaps: list[float] = []
        underruns: list[float] = []
        try:
            async for chunk in chunks:
                now = time.monotonic()
                dur = len(chunk) / (RATE_48K * 4)
                parts.append(chunk)
                total_bytes += len(chunk)
                if t_first is None:
                    t_first = now
                else:
                    gaps.append(now - (t_prev or now))
                t_prev = now
                if t_start is None:
                    buffered_s += dur
                    if buffered_s >= STREAM_PREBUFFER_S:
                        t_start, play_end = now, now + buffered_s
                else:
                    if now > play_end:
                        underruns.append(now - play_end)
                        play_end = now
                    play_end += dur
            if t_first is None:
                return
            if t_start is None:  # short reply: plays when the stream ends
                t_start = time.monotonic()
                play_end = t_start + buffered_s
            await asyncio.sleep(max(0.0, play_end - time.monotonic()))
        finally:
            if t_first is not None:
                self.plays.append((t_first, b"".join(parts)))
                self.ends.append(time.monotonic())
                self.streams.append(
                    {
                        "first_chunk_t": t_first,
                        "play_start_t": t_start,
                        "n_chunks": len(gaps) + 1,
                        "audio_s": round(total_bytes / (RATE_48K * 4), 3),
                        "chunk_gaps_s": [round(g, 3) for g in gaps],
                        "underruns_s": [round(u, 3) for u in underruns],
                    }
                )

    def stop(self) -> None:
        self.stop_calls += 1

    def is_playing(self) -> bool:
        return False


class CapWarningCounter(logging.Handler):
    """Counts the reply synthesizer's 'TTS over-generated' warnings (cap hit)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "over-generated" in record.getMessage():
            self.count += 1


class RecordingSynth:
    """Proxy over ReplySynthesizer: records the text and TTS-cap outcome of every call."""

    def __init__(self, inner, cap_kwargs: dict) -> None:  # type: ignore[no-untyped-def]
        self._inner = inner
        self._cap_kwargs = cap_kwargs
        self.cap_warnings = CapWarningCounter()  # "TTS over-generated" from reply.py
        logging.getLogger("notamusicbot.listen.reply").addHandler(self.cap_warnings)
        self.calls: list[str] = []
        self.results: list[dict] = []

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._inner, name)

    async def stream_sentence(self, text: str):  # type: ignore[no-untyped-def]
        """Pass-through of the streaming path; records bytes, cap and truncation."""
        from contextlib import aclosing

        from notamusicbot.listen.reply import tts_cap_seconds

        self.calls.append(text)
        n_bytes = 0
        warn0 = self.cap_warnings.count
        try:
            async with aclosing(self._inner.stream_sentence(text)) as chunks:
                async for chunk in chunks:
                    n_bytes += len(chunk)
                    yield chunk
        finally:
            # Always append (also when cancelled or failed) so results[k] stays calls[k].
            self.results.append(
                {
                    "words": len(text.split()),
                    "audio_s": round(n_bytes / (RATE_48K * 4), 3),
                    "cap_s": tts_cap_seconds(text, **self._cap_kwargs),
                    "truncated": self.cap_warnings.count > warn0,
                    "completed": n_bytes > 0,
                    "streamed": True,
                }
            )

    async def synthesize_sentence(self, text: str):  # type: ignore[no-untyped-def]
        from notamusicbot.listen.reply import tts_cap_seconds

        self.calls.append(text)
        audio = None
        try:
            audio = await self._inner.synthesize_sentence(text)
            return audio
        finally:
            # Always append (also when cancelled or failed) so results[k] stays calls[k].
            self.results.append(
                {
                    "words": len(text.split()),
                    "audio_s": None if audio is None else round(audio.duration_s, 3),
                    "cap_s": tts_cap_seconds(text, **self._cap_kwargs),
                    "truncated": bool(audio is not None and audio.truncated),
                    "completed": audio is not None,
                }
            )


class CountingTranscriber:
    """Wraps the real transcriber: counts calls and keeps per-call text and time."""

    def __init__(self, inner) -> None:  # type: ignore[no-untyped-def]
        self._inner = inner
        self.calls: list[dict] = []

    async def transcribe(self, audio):  # type: ignore[no-untyped-def]
        t = time.perf_counter()
        result = await self._inner.transcribe(audio)
        self.calls.append(
            {
                "audio_s": round(audio.duration_s, 3),
                "elapsed_s": round(time.perf_counter() - t, 3),
                "text": result.text,
            }
        )
        return result

    async def shutdown(self) -> None:
        await self._inner.shutdown()


class RecordingProvider:
    """Wraps the production chat provider: raw reply, first-delta and total time.

    Exposes ``stream`` (the TurnManager prefers it) and ``generate``; both
    delegate to the wrapped FallbackChatProvider unchanged.
    """

    def __init__(self, inner) -> None:  # type: ignore[no-untyped-def]
        self._inner = inner
        self.calls: list[dict] = []
        self.local_calls = 0  # incremented by the fallback hook below

    async def generate(self, request):  # type: ignore[no-untyped-def]
        t = time.perf_counter()
        text = await self._inner.generate(request)
        self.calls.append(
            {"first_delta_s": None, "gen_s": round(time.perf_counter() - t, 3), "text": text}
        )
        return text

    async def stream(self, request):  # type: ignore[no-untyped-def]
        from contextlib import aclosing

        t = time.perf_counter()
        local0 = self.local_calls
        first: float | None = None
        parts: list[str] = []
        try:
            async with aclosing(self._inner.stream(request)) as deltas:
                async for delta in deltas:
                    if first is None and delta:
                        first = time.perf_counter() - t
                    parts.append(delta)
                    yield delta
        finally:
            self.calls.append(
                {
                    "first_delta_s": None if first is None else round(first, 3),
                    "gen_s": round(time.perf_counter() - t, 3),
                    "text": "".join(parts),
                    "used_local": self.local_calls > local0,
                }
            )


class WarningCapture(logging.Handler):
    """Collects WARNING+ records from the provider loggers."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(f"{record.name}: {record.getMessage()}")


def spark_models(url: str = SPARK_MODELS_URL) -> list[str]:
    """Model ids the Spark reports (loopback only); raises if the tunnel is down."""
    from urllib.request import urlopen

    with urlopen(url, timeout=5) as resp:  # noqa: S310 - fixed loopback URL
        return [str(m["id"]) for m in json.load(resp)["data"]]


@dataclass
class TurnRecord:
    index: int
    question: str
    clip_s: float
    metrics_line: str | None = None
    t_eos: float | None = None
    endpoint_wait_s: float | None = None
    eos_to_backchannel_s: float | None = None
    first_reply_play_minus_eos_s: float | None = None
    plays: list[dict] = field(default_factory=list)
    synth_texts: list[str] = field(default_factory=list)
    synth_results: list[dict] = field(default_factory=list)
    llm_call: dict | None = None
    n_utterances: int = 1
    n_asr_calls: int = 1
    asr_call: dict | None = None
    warnings: list[str] = field(default_factory=list)
    dead_air_gaps_s: list[float] = field(default_factory=list)
    streams: list[dict] = field(default_factory=list)
    failure: str | None = None


def build_packets(pcm_48k_stereo: bytes) -> list[bytes]:
    """Lead silence + the clip + tail silence, split into 20 ms packets."""
    stereo = np.frombuffer(pcm_48k_stereo, dtype="<i2").reshape(-1, 2)
    lead = np.zeros((int(LEAD_SILENCE_S * RATE_48K), 2), dtype="<i2")
    tail = np.zeros((int(TAIL_SILENCE_S * RATE_48K), 2), dtype="<i2")
    full = np.concatenate([lead, stereo, tail])
    return [full[i : i + PACKET_FRAMES].tobytes() for i in range(0, len(full), PACKET_FRAMES)]


async def fallback_drill(settings, profile, sampler: VramSampler) -> dict:
    """3 CHAT turns with the Spark pointed at 127.0.0.1:9 (nothing listens).

    Uses the production wiring (a second Application's listen provider). Checks:
    exactly one WARNING on the first call, a lazy in-process load, replies from
    the local model, the cold-load time; then that a healthy Spark makes
    ``ensure_spark_preferred`` unload the idle local model; then that a
    non-loopback base_url is refused before any request.
    """
    import httpx

    from notamusicbot.app import Application
    from notamusicbot.config import ListenLlmSettings, SecretsSettings
    from notamusicbot.content.base import ContentKind, ContentRequest
    from notamusicbot.content.fallback import FallbackChatProvider, LazyLocalProvider
    from notamusicbot.content.spark_chat import SparkChatProvider

    sampler.phase = "drill_setup"
    drill: dict = {"unreachable_base_url": "http://127.0.0.1:9/v1"}
    dead_listen = settings.listen.model_copy(
        update={"llm": settings.listen.llm.model_copy(update={"base_url": "http://127.0.0.1:9/v1"})}
    )
    dead_app = Application(
        settings.model_copy(update={"listen": dead_listen}), SecretsSettings(_env_file=None)
    )
    provider = dead_app.listen_content_provider
    assert isinstance(provider, FallbackChatProvider) and provider.fallback is not None
    capture = WarningCapture()
    for name in ("notamusicbot.content.fallback", "notamusicbot.content.spark_chat"):
        logging.getLogger(name).addHandler(capture)
    drill["local_loaded_before"] = provider.fallback.loaded
    drill["vram_before_mib"] = smi_used_mib()
    questions = [
        "hey goku how are you today",
        "goku what is your favorite food",
        "who is the toughest fighter you know",
    ]
    calls = []
    sampler.phase = "drill_turns"
    for i, question in enumerate(questions, 1):
        request = ContentRequest(
            kind=ContentKind.CHAT,
            persona_description=profile.persona.description,
            requester_name="User 1",
            transcript_text=question,
        )
        n_warn = len(capture.records)
        t = time.perf_counter()
        first = None
        text = ""
        async for delta in provider.stream(request):
            if first is None and delta:
                first = time.perf_counter() - t
            text += delta
        calls.append(
            {
                "call": i,
                "question": question,
                "first_delta_s": None if first is None else round(first, 3),
                "total_s": round(time.perf_counter() - t, 3),
                "new_warnings": capture.records[n_warn:],
                "local_loaded_after": provider.fallback.loaded,
                "reply": text,
                "vram_after_mib": smi_used_mib(),
            }
        )
    drill["calls"] = calls
    drill["warning_count_total"] = len(capture.records)
    drill["warning_count_first_call"] = len(calls[0]["new_warnings"])
    drill["warnings"] = capture.records
    local = provider.fallback._provider
    info = getattr(local, "load_info", None)
    drill["local_load_info"] = (
        {
            "model_id": info.model_id,
            "load_s": info.load_s,
            "vram_used_mib": info.vram_used_mib,
        }
        if info is not None
        else None
    )
    drill["cold_load_s_reported_by_loader"] = info.load_s if info is not None else None
    drill["first_call_total_s"] = calls[0]["total_s"]
    drill["second_call_total_s"] = calls[1]["total_s"]
    drill["first_minus_second_s"] = round(calls[0]["total_s"] - calls[1]["total_s"], 3)
    drill["vram_with_local_loaded_mib"] = smi_used_mib()

    # Healthy Spark again: the idle local model must be unloaded.
    sampler.phase = "drill_unload"
    healthy = FallbackChatProvider(
        SparkChatProvider(settings.listen.llm, max_new_tokens=settings.listen.reply.max_new_tokens),
        provider.fallback,
        settings.listen.llm.health_interval_s,
    )
    t = time.perf_counter()
    unloaded = await healthy.ensure_spark_preferred()
    await asyncio.sleep(1.0)
    drill["ensure_spark_preferred"] = {
        "unloaded": unloaded,
        "elapsed_s": round(time.perf_counter() - t, 3),
        "local_loaded_after": provider.fallback.loaded,
        "vram_after_mib": smi_used_mib(),
    }

    # Non-loopback base_url: refused before any request.
    sampler.phase = "drill_nonloopback"
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "x"}]})

    refusals: dict = {}
    bad_url = "http://10.0.0.5:8000/v1"
    try:
        ListenLlmSettings(base_url=bad_url)
        refusals["config_validator"] = "NOT REFUSED"
    except ValueError as exc:  # pydantic ValidationError subclasses ValueError
        refusals["config_validator"] = f"{type(exc).__name__}: " + str(exc).splitlines()[-2].strip()
    bypass = ListenLlmSettings.model_construct(base_url=bad_url)
    try:
        SparkChatProvider(bypass, transport=httpx.MockTransport(handler))
        refusals["provider_ctor"] = "NOT REFUSED"
    except ValueError as exc:
        refusals["provider_ctor"] = f"{type(exc).__name__}: {exc}"
    mutated = ListenLlmSettings()
    spark = SparkChatProvider(mutated, transport=httpx.MockTransport(handler))
    mutated.base_url = bad_url  # changed after construction: the per-call guard must catch it
    factory_calls: list[int] = []
    guarded = FallbackChatProvider(
        spark, LazyLocalProvider(lambda: factory_calls.append(1) or None)
    )
    request = ContentRequest(
        kind=ContentKind.CHAT,
        persona_description=profile.persona.description,
        requester_name="User 1",
        transcript_text="hello",
    )
    try:
        async for _ in guarded.stream(request):
            pass
        refusals["provider_call_guard"] = "NOT REFUSED"
    except ValueError as exc:
        refusals["provider_call_guard"] = f"{type(exc).__name__}: {exc}"
    refusals["requests_sent_to_transport"] = len(sent)
    refusals["local_factory_calls"] = len(factory_calls)
    drill["non_loopback"] = refusals

    for name in ("notamusicbot.content.fallback", "notamusicbot.content.spark_chat"):
        logging.getLogger(name).removeHandler(capture)
    sampler.phase = "drill_done"
    return drill


async def run(args: argparse.Namespace) -> dict:
    import soxr

    from notamusicbot.app import Application
    from notamusicbot.config import SecretsSettings, load_app_settings
    from notamusicbot.content.base import ContentKind, ContentRequest
    from notamusicbot.content.fallback import FallbackChatProvider
    from notamusicbot.content.llm import LocalQwenLlmProvider
    from notamusicbot.domain.listen import (
        AudioChunk,
        ConsentState,
        SpeechStarted,
        UtteranceEvent,
    )
    from notamusicbot.listen.conversation import Conversation, StreamChunker
    from notamusicbot.listen.endpointer import Endpointer, EndpointerThread
    from notamusicbot.listen.ingress import ListenIngress
    from notamusicbot.listen.metrics import parse_turn_line
    from notamusicbot.listen.reply import (
        ReplySynthesizer,
        warm_voice_converter,
    )
    from notamusicbot.listen.turn_taking import MonotonicClock, TurnManager
    from notamusicbot.speech.asr.factory import build_transcriber
    from notamusicbot.speech.vad.silero import SileroVad

    changed = enforce_offline_models()
    result: dict = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "note": NOTE,
        "offline_env_changed": changed,
        "budget": BUDGET,
    }
    sampler = VramSampler()
    baseline = smi_used_mib()
    result["baseline_vram_used_mib"] = baseline
    sampler.start()
    sampler.phase = "load"

    settings = load_app_settings()
    # Production wiring: listen on, listen.llm defaults (spark, auto, loopback :8000).
    # Owner's intended settings (B4): mirror config/app.yaml's listen block (asserted
    # below) and pass the two not-yet-in-config values through settings:
    # Parakeet fp32 on CUDA and a 400 ms end-of-speech silence.
    owner = settings.listen
    mirrored = {
        "llm.thinking": owner.llm.thinking == "low",
        "voice_profile": owner.voice_profile == "goku_raw",
        "reply.backchannel_enabled": owner.reply.backchannel_enabled is False,
        "reply.max_sentences": owner.reply.max_sentences == 1,
        "reply.first_chunk_max_words": owner.reply.first_chunk_max_words == 40,
        "reply.later_chunk_max_words": owner.reply.later_chunk_max_words == 40,
    }
    result["owner_config_mirror"] = {
        **mirrored,
        "owner_asr": owner.asr.model_dump(),
        "owner_vad_end_silence_ms": owner.vad.end_silence_ms,
    }
    if not all(mirrored.values()):
        raise RuntimeError(f"config/app.yaml listen block differs from the intended: {mirrored}")
    settings = settings.model_copy(
        update={
            "listen": owner.model_copy(
                update={
                    "enabled": True,
                    "asr": owner.asr.model_copy(update={"device": "cuda", "quantization": "fp32"}),
                    "vad": owner.vad.model_copy(update={"end_silence_ms": 400}),
                }
            )
        }
    )
    listen = settings.listen
    if listen.llm.provider != "spark" or listen.llm.model != "auto":
        raise RuntimeError(f"listen.llm is not the production default: {listen.llm!r}")
    result["listen_llm_settings"] = listen.llm.model_dump()
    result["listen_reply_settings"] = listen.reply.model_dump()
    # Stop early if the tunnel is down (nothing else is loaded yet).
    try:
        result["spark_models_before"] = await asyncio.to_thread(spark_models)
    except Exception as exc:
        raise RuntimeError(f"Spark tunnel down: GET {SPARK_MODELS_URL} failed: {exc!r}") from exc
    # Loaders must resolve hub ids themselves under HF_HUB_OFFLINE=1 (B3); no
    # snapshot-directory workaround here.
    result["llm_model_id"] = settings.content.model_id
    result["tts_model_id"] = settings.speech.tts_model_id
    # Do not read .env: nothing here needs a token.
    app = Application(settings, SecretsSettings(_env_file=None))
    profile = app.voice_profiles[listen.voice_profile]
    bc_dir = REPO_ROOT / "var" / "listen_bench_backchannel"
    shutil.rmtree(bc_dir, ignore_errors=True)
    transcriber = None
    manager = None
    poller = None
    try:
        t = time.perf_counter()
        # Production startup call: with listen on and provider spark it skips the
        # eager in-process load.
        await asyncio.to_thread(app.load_content_model)
        provider = app.listen_content_provider
        if not isinstance(provider, FallbackChatProvider):
            raise RuntimeError(f"listen provider is {type(provider).__name__}, not Fallback")
        music_llm = app.build_content_provider()
        local_loaded_at_start = bool(
            (isinstance(music_llm, LocalQwenLlmProvider) and music_llm.is_loaded())
            or (provider.fallback is not None and provider.fallback.loaded)
        )
        if local_loaded_at_start:
            raise RuntimeError("in-process LLM is loaded at startup (must not be)")
        result["llm_startup_s"] = round(time.perf_counter() - t, 3)
        result["inprocess_llm_loaded_at_startup"] = local_loaded_at_start
        result["llm_max_new_tokens"] = listen.reply.max_new_tokens
        t = time.perf_counter()
        pipeline = await asyncio.to_thread(
            lambda: (app.load_speech_model(), app.build_pipeline())[1]
        )
        result["tts_load_s"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        await app.converter_router.startup()
        result["rvc_startup_s"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        transcriber = build_transcriber(listen.asr, app.coordinator)
        await asyncio.to_thread(transcriber.load)  # type: ignore[attr-defined]
        result["asr_load_s"] = round(time.perf_counter() - t, 2)
        result["asr_backend"] = listen.asr.backend
        result["asr_settings"] = listen.asr.model_dump()

        artifact_root = Path(settings.speech.output_dir)
        cap_kwargs = {
            "seconds_per_word": listen.reply.tts_max_seconds_per_word,
            "floor_seconds": listen.reply.tts_cap_floor_seconds,
        }
        synth = ReplySynthesizer(
            pipeline,
            profile,
            artifact_root=artifact_root,
            tts_max_seconds_per_word=cap_kwargs["seconds_per_word"],
            tts_cap_floor_seconds=cap_kwargs["floor_seconds"],
        )
        plain_profile = profile.model_copy(
            update={
                "voice_conversion": profile.voice_conversion.model_copy(
                    update={"backend": "passthrough"}
                )
            }
        )
        # Input clips are the bench's own questions: never cap them.
        plain_synth = ReplySynthesizer(
            pipeline,
            plain_profile,
            artifact_root=artifact_root,
            tts_max_seconds_per_word=5.0,
            tts_cap_floor_seconds=20.0,
        )
        sampler.phase = "load_done_idle"
        await asyncio.sleep(1.0)  # let the sampler record the resident level
        result["after_load_vram_used_mib"] = smi_used_mib()
        result["vram_check"] = {
            "baseline_mib": baseline,
            "after_load_mib": result["after_load_vram_used_mib"],
            "resident_above_baseline_mib": result["after_load_vram_used_mib"] - baseline,
            "prior_baseline_mib": PRIOR_BASELINE_MIB,
            "prior_after_load_mib": PRIOR_AFTER_LOAD_MIB,
            "prior_resident_above_baseline_mib": PRIOR_AFTER_LOAD_MIB - PRIOR_BASELINE_MIB,
            "after_load_minus_prior_after_load_mib": result["after_load_vram_used_mib"]
            - PRIOR_AFTER_LOAD_MIB,
            "resident_drop_vs_prior_mib": (PRIOR_AFTER_LOAD_MIB - PRIOR_BASELINE_MIB)
            - (result["after_load_vram_used_mib"] - baseline),
        }

        # ---- input clips: plain TTS (no VC), in memory -------------------------
        sampler.phase = "prepare_clips"
        clips = []
        for text in QUESTIONS:
            audio = await plain_synth.synthesize_sentence(text)
            clips.append({"text": text, "pcm": audio.pcm, "duration_s": audio.duration_s})
        result["clips"] = [
            {"text": c["text"], "duration_s": round(c["duration_s"], 2)} for c in clips
        ]

        # ---- warm-ups (not timed): RVC, TTS, ASR, LLM, backchannel render ------
        sampler.phase = "warmup"
        result["rvc_warmed"] = await warm_voice_converter(pipeline, profile)
        await synth.synthesize_sentence("Warm up sentence number one.")
        await synth.synthesize_sentence("Warm up sentence number two, a little longer.")
        samples16 = np.frombuffer(clips[0]["pcm"], dtype="<i2").reshape(-1, 2).mean(axis=1)
        a16 = soxr.resample(samples16.astype(np.float32) / 32768.0, RATE_48K, 16000)
        await transcriber.transcribe(AudioChunk(np.ascontiguousarray(a16, dtype=np.float32)))
        warn_capture = WarningCapture()
        for name in ("notamusicbot.content.fallback", "notamusicbot.content.spark_chat"):
            logging.getLogger(name).addHandler(warn_capture)
        warm_text = ""
        async for delta in provider.stream(
            ContentRequest(
                kind=ContentKind.CHAT,
                persona_description=profile.persona.description,
                requester_name="User 1",
                transcript_text="hey goku how are you today",
            )
        ):
            warm_text += delta
        result["spark_warmup_reply"] = warm_text
        result["spark_model_id_resolved"] = provider.primary._model
        rec_synth = RecordingSynth(synth, cap_kwargs)
        backchannel = None  # reply.backchannel_enabled is false: no clips, as in production
        result["backchannel"] = {"enabled": listen.reply.backchannel_enabled}
        result["can_stream"] = bool(rec_synth.can_stream)
        result["tts_fast_model"] = type(getattr(pipeline, "_synthesizer", None)).__name__
        result["tts_supports_streaming"] = bool(
            getattr(getattr(pipeline, "_synthesizer", None), "supports_streaming", False)
        )
        bc_pcms: set[bytes] = set()
        sampler.phase = "idle_before_turns"
        await asyncio.sleep(1.0)

        # ---- the real chain -----------------------------------------------------
        sampler.phase = "turns"
        loop = asyncio.get_running_loop()
        consent = ConsentState(bot_user_id=BOT_ID)
        consent.opt_in(USER_ID)  # STRANGER_ID is never opted in
        decode_total = [0]  # every decode() call by any user

        def decoder_factory() -> PassThroughDecoder:
            return PassThroughDecoder(decode_total)

        ingress = ListenIngress(consent, BOT_ID, decoder_factory, listen.vad.ring_buffer_s)
        clock = MonotonicClock()
        player = RecordingPlayer()
        counting_asr = CountingTranscriber(transcriber)
        recording_llm = RecordingProvider(provider)
        if provider.fallback is not None:  # count every call that reaches the local model
            local_generate = provider.fallback.generate

            async def counted_local_generate(request):  # type: ignore[no-untyped-def]
                recording_llm.local_calls += 1
                return await local_generate(request)

            provider.fallback.generate = counted_local_generate  # type: ignore[method-assign]
        metrics_lines: list[str] = []
        manager = TurnManager(
            listen,
            clock,
            consent,
            counting_asr,
            Conversation(listen.reply.history_turns),
            recording_llm,
            rec_synth,
            backchannel,
            player,
            metrics_lines.append,
            BOT_ID,
            guild_id=1,
            persona_description=profile.persona.description,
        )
        utterances: list[dict] = []
        dispatch_tasks: set[asyncio.Task] = set()
        utterance_dispatched = asyncio.Event()

        async def dispatch(event) -> None:  # type: ignore[no-untyped-def]
            if isinstance(event, SpeechStarted):
                await manager.on_speech_started(event)
                return
            await manager.on_utterance(event)
            utterance_dispatched.set()

        def handle(event) -> None:  # type: ignore[no-untyped-def]
            if isinstance(event, UtteranceEvent):
                utterances.append(
                    {"user_id": event.user_id, "t_eos": event.t_eos, "t_emit": time.monotonic()}
                )
            task = loop.create_task(dispatch(event))
            dispatch_tasks.add(task)
            task.add_done_callback(dispatch_tasks.discard)

        endpointer = Endpointer(
            SileroVad,  # factory: one stateful detector per user
            listen.vad,
            clock,
            lambda ev: loop.call_soon_threadsafe(handle, ev),
        )
        poller = EndpointerThread(ingress, endpointer, consent.opted_in)
        poller.start()

        records: list[TurnRecord] = []
        for turn in range(args.turns):
            clip = clips[turn % len(clips)]
            rec = TurnRecord(index=turn + 1, question=clip["text"], clip_s=clip["duration_s"])
            records.append(rec)
            n_utt, n_met, n_plays = len(utterances), len(metrics_lines), len(player.plays)
            n_synth = len(rec_synth.calls)
            n_streams = len(player.streams)
            n_llm_calls, n_asr_calls, n_warn = (
                len(recording_llm.calls),
                len(counting_asr.calls),
                len(warn_capture.records),
            )
            utterance_dispatched.clear()
            source = PacedPacketSource(USER_ID, build_packets(clip["pcm"]), realtime=True)
            await asyncio.to_thread(source.deliver, ingress.accept_packet)
            try:
                await asyncio.wait_for(utterance_dispatched.wait(), timeout=30.0)
                await asyncio.wait_for(manager.wait_idle(), timeout=120.0)
            except TimeoutError:
                rec.failure = "timeout waiting for utterance/turn"
            await asyncio.sleep(0.5)  # let a late event settle before the next turn
            if len(utterances) <= n_utt:
                rec.failure = rec.failure or "no utterance endpointed"
                continue
            # A clip with an internal pause can endpoint twice (a fragment, then the
            # rest). The reply belongs to the LAST utterance, so anchor on it.
            rec.n_utterances = len(utterances) - n_utt
            utt = utterances[-1]
            rec.t_eos = utt["t_eos"]
            rec.endpoint_wait_s = round(utt["t_emit"] - utt["t_eos"], 3)
            new_plays = player.plays[n_plays:]
            new_ends = player.ends[n_plays:]
            rec.synth_texts = rec_synth.calls[n_synth:]
            rec.synth_results = rec_synth.results[n_synth:]
            rec.warnings = warn_capture.records[n_warn:]
            new_llm = recording_llm.calls[n_llm_calls:]
            new_asr = counting_asr.calls[n_asr_calls:]
            rec.llm_call = new_llm[-1] if new_llm else None
            rec.asr_call = new_asr[-1] if new_asr else None
            rec.n_asr_calls = len(new_asr)
            rec.streams = player.streams[n_streams:]
            for k, (t_play, pcm) in enumerate(new_plays):
                t_end = new_ends[k] if k < len(new_ends) else None
                rec.plays.append(
                    {
                        "at_eos_plus_s": round(t_play - utt["t_eos"], 3),
                        "end_at_eos_plus_s": (
                            round(t_end - utt["t_eos"], 3) if t_end is not None else None
                        ),
                        "audio_s": round(len(pcm) / (RATE_48K * 4), 2),
                        "backchannel": pcm in bc_pcms,
                    }
                )
            real_plays = [p for p in rec.plays if not p["backchannel"]]
            for prev, nxt in zip(real_plays, real_plays[1:], strict=False):
                if prev["end_at_eos_plus_s"] is not None:
                    rec.dead_air_gaps_s.append(
                        round(max(0.0, nxt["at_eos_plus_s"] - prev["end_at_eos_plus_s"]), 3)
                    )
            bcs = [p for p in rec.plays if p["backchannel"]]
            reals = [p for p in rec.plays if not p["backchannel"]]
            # A split utterance can play a clip for the dropped fragment (negative
            # offset); the clip of the anchored (last) utterance is the first >= 0.
            anchored = [p for p in bcs if p["at_eos_plus_s"] >= 0]
            rec.eos_to_backchannel_s = anchored[0]["at_eos_plus_s"] if anchored else None
            rec.first_reply_play_minus_eos_s = reals[0]["at_eos_plus_s"] if reals else None
            if len(metrics_lines) > n_met:
                rec.metrics_line = metrics_lines[n_met]
            else:
                rec.failure = rec.failure or "no metrics line (turn dropped)"

        # ---- the not-opted-in user: one extra turn, fed at full speed -----------
        n_calls = len(counting_asr.calls)
        n_met, n_plays, n_llm = len(metrics_lines), len(player.plays), len(recording_llm.calls)
        n_decoded = decode_total[0]
        stranger = PacedPacketSource(STRANGER_ID, build_packets(clips[0]["pcm"]), realtime=False)
        await asyncio.to_thread(stranger.deliver, ingress.accept_packet)
        await asyncio.sleep(2.0)  # long enough for any (wrongly) endpointed turn to show up
        await manager.wait_idle()
        stranger_result = {
            "packets_sent": len(stranger.payloads),
            "decoder_calls": decode_total[0] - n_decoded,
            "buffered_samples": ingress.pending_samples(STRANGER_ID),
            "transcribe_calls": len(counting_asr.calls) - n_calls,
            "llm_calls": len(recording_llm.calls) - n_llm,
            "metrics_lines": len(metrics_lines) - n_met,
            "plays": len(player.plays) - n_plays,
        }
        stranger_result["passed"] = all(
            stranger_result[k] == 0
            for k in (
                "decoder_calls",
                "buffered_samples",
                "transcribe_calls",
                "llm_calls",
                "metrics_lines",
                "plays",
            )
        )
        result["not_opted_in_turn"] = stranger_result

        # ---- per-turn table, replies, percentiles -------------------------------
        rows = []
        for rec in records:
            parsed = parse_turn_line(rec.metrics_line) if rec.metrics_line else None
            rows.append(
                {
                    "turn": rec.index,
                    "question": rec.question,
                    "clip_s": round(rec.clip_s, 2),
                    "failure": rec.failure,
                    "endpoint_wait_s": rec.endpoint_wait_s,
                    "metrics_line": rec.metrics_line,
                    "stages_s": {k: parsed[k] for k in STAGES} if parsed else None,
                    "eos_to_backchannel_s": rec.eos_to_backchannel_s,
                    "first_reply_play_minus_eos_s": rec.first_reply_play_minus_eos_s,
                    "plays": rec.plays,
                    "dead_air_gaps_s": rec.dead_air_gaps_s,
                    "first_pcm_chunk_s": rec.first_reply_play_minus_eos_s,
                    "streams": [
                        {k: v for k, v in st.items() if k not in ("first_chunk_t", "play_start_t")}
                        | {
                            "play_start_at_eos_plus_s": (
                                round(st["play_start_t"] - rec.t_eos, 3)
                                if st["play_start_t"] and rec.t_eos
                                else None
                            ),
                            "max_chunk_gap_s": max(st["chunk_gaps_s"], default=0.0),
                        }
                        for st in rec.streams
                    ],
                    "synth_texts": rec.synth_texts,
                    "chunk_word_counts": [len(t.split()) for t in rec.synth_texts],
                    "tts_results": rec.synth_results,
                    "n_utterances_in_turn": rec.n_utterances,
                    "n_asr_calls_in_turn": rec.n_asr_calls,
                    "chunks_truncated_by_tts_cap": [
                        {"text": t, **r}
                        for t, r in zip(rec.synth_texts, rec.synth_results, strict=False)
                        if r["truncated"]
                    ],
                    "spark_first_delta_s": rec.llm_call["first_delta_s"] if rec.llm_call else None,
                    "used_local_model": bool(rec.llm_call and rec.llm_call.get("used_local")),
                    "provider_warnings": rec.warnings,
                    "emoji_or_markdown_in_tts": [
                        t for t in rec.synth_texts if _EMOJI_RE.search(t) or _MARKDOWN_RE.search(t)
                    ],
                }
            )
        ok_rows = [r for r in rows if r["stages_s"] is not None]
        replies = []
        spoken_ok = True
        for rec in records:
            if rec.llm_call is None or rec.asr_call is None:
                continue
            raw = rec.llm_call["text"]
            chunker = StreamChunker(
                listen.reply.max_sentences,
                listen.reply.first_chunk_max_words,
                listen.reply.later_chunk_max_words,
            )
            planned = chunker.feed_chunks(raw) + chunker.finish_chunks()
            n_spoken = len({c.sentence_index for c in planned})
            raw_sentences = len(re.findall(r"[.!?]+(?:\s|$)", raw or ""))
            n_chunks_played = len([p for p in rec.plays if not p["backchannel"]])
            replies.append(
                {
                    "turn": rec.index,
                    "asr_text": rec.asr_call["text"],
                    "asr_s": rec.asr_call["elapsed_s"],
                    "llm_gen_s": rec.llm_call["gen_s"],
                    "llm_first_delta_s": rec.llm_call["first_delta_s"],
                    "llm_raw": raw,
                    "raw_sentences": raw_sentences,
                    "raw_chars": len(raw),
                    "spoken_sentences": n_spoken,
                    "chunks_synthesized": len(rec.synth_texts),
                    "chunks_played": n_chunks_played,
                    "spoken_text": " ".join(c.text for c in planned),
                    "within_max_sentences": n_spoken <= listen.reply.max_sentences,
                }
            )
            spoken_ok &= n_spoken <= listen.reply.max_sentences
        summary: dict = {}
        for key in STAGES:
            vals = [r["stages_s"][key] for r in ok_rows if r["stages_s"][key] is not None]
            summary[key] = (
                {
                    "n": len(vals),
                    "p50_s": percentile(vals, 50),
                    "p90_s": percentile(vals, 90),
                    "min_s": min(vals),
                    "max_s": max(vals),
                }
                if vals
                else None
            )
        bc_vals = [r["eos_to_backchannel_s"] for r in rows if r["eos_to_backchannel_s"] is not None]
        ep_vals = [r["endpoint_wait_s"] for r in rows if r["endpoint_wait_s"] is not None]
        summary["eos->backchannel"] = (
            {
                "n": len(bc_vals),
                "p50_s": percentile(bc_vals, 50),
                "p90_s": percentile(bc_vals, 90),
                "max_s": max(bc_vals),
            }
            if bc_vals
            else None
        )
        summary["endpoint_wait (eos->utterance emitted)"] = (
            {"n": len(ep_vals), "p50_s": percentile(ep_vals, 50), "p90_s": percentile(ep_vals, 90)}
            if ep_vals
            else None
        )
        gaps = [g for r in records for g in r.dead_air_gaps_s]
        summary["dead_air_gap_between_chunks"] = (
            {
                "n": len(gaps),
                "p50_s": percentile(gaps, 50),
                "p90_s": percentile(gaps, 90),
                "max_s": max(gaps),
            }
            if gaps
            else None
        )
        fra = [
            r["first_reply_play_minus_eos_s"]
            for r in rows
            if r["first_reply_play_minus_eos_s"] is not None
        ]
        summary["first_reply_audio (player, eos->first real chunk)"] = (
            {
                "n": len(fra),
                "p50_s": percentile(fra, 50),
                "p90_s": percentile(fra, 90),
                "max_s": max(fra),
            }
            if fra
            else None
        )
        summary["time_to_first_spark_delta"] = (
            {
                "n": len(fd),
                "p50_s": percentile(fd, 50),
                "p90_s": percentile(fd, 90),
                "max_s": max(fd),
            }
            if (
                fd := [
                    r["spark_first_delta_s"] for r in rows if r["spark_first_delta_s"] is not None
                ]
            )
            else None
        )
        all_gaps = [g for r in rows for st in r["streams"] for g in st["chunk_gaps_s"]]
        all_underruns = [u for r in rows for st in r["streams"] for u in st["underruns_s"]]
        summary["chunk_gaps_within_stream"] = (
            {
                "n": len(all_gaps),
                "p50_s": percentile(all_gaps, 50),
                "p90_s": percentile(all_gaps, 90),
                "max_s": max(all_gaps),
            }
            if all_gaps
            else None
        )
        summary["underruns"] = {
            "count": len(all_underruns),
            "turns": sorted({r["turn"] for r in rows for st in r["streams"] if st["underruns_s"]}),
            "total_s": round(sum(all_underruns), 3),
            "max_s": max(all_underruns, default=0.0),
            "streams_played": sum(len(r["streams"]) for r in rows),
            "turns_played_non_streaming": [
                r["turn"] for r in rows if r["plays"] and not r["streams"]
            ],
        }
        truncated_chunks = [
            {"turn": r["turn"], **c} for r in rows for c in r["chunks_truncated_by_tts_cap"]
        ]
        summary["chunks_truncated_by_tts_cap"] = {
            "count": len(truncated_chunks),
            "of_chunks": sum(len(r["synth_texts"]) for r in rows),
            "which": truncated_chunks,
        }
        summary["turns_with_more_than_one_utterance"] = [
            r["turn"] for r in rows if r["n_utterances_in_turn"] > 1
        ]
        summary["max_chunk_words"] = max(
            (w for r in rows for w in r["chunk_word_counts"]), default=None
        )
        summary["turns_fell_back_to_local"] = sum(1 for r in rows if r["used_local_model"])
        summary["backchannel_fraction_over_0.8s"] = (
            round(sum(1 for v in bc_vals if v > BUDGET["backchannel_s"]) / len(bc_vals), 3)
            if bc_vals
            else None
        )
        summary["backchannel_n_over_0.8s"] = sum(1 for v in bc_vals if v > BUDGET["backchannel_s"])
        summary["emoji_or_markdown_turns"] = [
            r["turn"] for r in rows if r["emoji_or_markdown_in_tts"]
        ]
        total = summary["total"]
        verdict = {
            "turns_requested": args.turns,
            "turns_with_metrics": len(ok_rows),
            "total_p50_within_6s": bool(total and total["p50_s"] <= BUDGET["total_p50_s"]),
            "total_p90_within_9s": bool(total and total["p90_s"] <= BUDGET["total_p90_s"]),
            "backchannel_p50_within_0.8s": bool(
                summary["eos->backchannel"]
                and summary["eos->backchannel"]["p50_s"] <= BUDGET["backchannel_s"]
            ),
            "backchannel_max_within_0.8s": bool(
                summary["eos->backchannel"]
                and summary["eos->backchannel"]["max_s"] <= BUDGET["backchannel_s"]
            ),
            "all_replies_within_max_sentences": spoken_ok,
            "max_sentences": listen.reply.max_sentences,
        }
        result.update(
            {
                "percentile_method": "nearest rank: value at 1-based rank ceil(p/100*n)",
                "stage_labels": STAGE_LABELS,
                "spark_models_after": await asyncio.to_thread(spark_models),
                "provider_warnings_during_turns": list(warn_capture.records),
                "inprocess_llm_loaded_after_turns": bool(
                    provider.fallback is not None and provider.fallback.loaded
                ),
                "turns": rows,
                "replies": replies,
                "summary": summary,
                "verdict": verdict,
            }
        )
        # ---- fallback drill (separate; after the 30 turns) -------------------------
        if args.drill:
            result["fallback_drill"] = await fallback_drill(settings, profile, sampler)
    finally:
        if poller is not None:
            poller.stop()
        sampler.phase = "shutdown"
        if transcriber is not None:
            await transcriber.shutdown()
        try:
            await app.shutdown()
        except Exception:
            traceback.print_exc()
        shutil.rmtree(bc_dir, ignore_errors=True)
        sampler.stop()
        result["vram"] = sampler.summary()
        drill_phases = {p for _, _, p in sampler.samples if p.startswith("drill_")}
        non_drill = [u for _, u, p in sampler.samples if p not in drill_phases]
        result["peak_vram_used_mib"] = max(non_drill) if non_drill else None
        result["peak_vram_including_drill_mib"] = sampler.peak()
        result["peak_vram_during_turns_mib"] = sampler.peak("turns")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument(
        "--drill", action="store_true", help="also run the Spark-down fallback drill"
    )
    args = parser.parse_args()
    try:
        result = asyncio.run(run(args))
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps({k: result[k] for k in ("summary", "verdict", "not_opted_in_turn")}, indent=2))
    print(
        "baseline MiB", result["baseline_vram_used_mib"], "peak MiB", result["peak_vram_used_mib"]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
