"""Fakes and a builder for the TurnManager tests (no GPU, no Discord, no models)."""

from __future__ import annotations

import asyncio
import heapq
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, field

import numpy as np

from conftest import FakeClock, FakeContentProvider, FakeTranscriber
from notamusicbot.config import ListenReplySettings, ListenSettings
from notamusicbot.domain.listen import (
    AudioChunk,
    SpeechStarted,
    Transcript,
    UtteranceEvent,
)
from notamusicbot.listen.conversation import Conversation
from notamusicbot.listen.reply import ReplyAudio, ReplySynthesisError
from notamusicbot.listen.turn_taking import TurnManager

BOT_ID = 999
BACKCHANNEL_PCM = b"<bc>"


class TimelineClock(FakeClock):
    """Virtual-time clock where concurrent sleeps overlap (FakeClock's add up).

    ``sleep`` parks the caller until ``run`` advances ``now`` to its wake time.
    """

    def __init__(self, start: float = 0.0) -> None:
        super().__init__(start)
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = 0

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._seq += 1
        heapq.heappush(self._sleepers, (self.now + seconds, self._seq, fut))
        await fut

    async def run(self, until: float | None = None) -> None:
        """Advance virtual time through every wake-up (up to ``until``)."""
        await spin()
        while self._sleepers:
            wake, _, fut = self._sleepers[0]
            if until is not None and wake > until + 1e-9:
                break
            heapq.heappop(self._sleepers)
            if fut.done():  # a cancelled sleep must not move time
                continue
            self.now = max(self.now, wake)
            fut.set_result(None)
            await spin()
        if until is not None:
            self.now = max(self.now, until)


class StreamingProvider:
    """Streaming ContentProvider: yields ``deltas`` ``gap_s`` apart on a TimelineClock.

    ``log`` gets ``("delta", time, text)`` per delta and ``("stream_end", time, "")``
    when the stream finishes. ``fail_after`` raises ``error`` after that many deltas.
    """

    def __init__(
        self,
        clock: TimelineClock,
        log: list[tuple[str, float, str]],
        deltas: list[str],
        gap_s: float = 0.5,
        fail_after: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.clock, self.log, self.deltas, self.gap_s = clock, log, deltas, gap_s
        self.fail_after = fail_after
        self.error = error or RuntimeError("spark down")
        self.calls: list[object] = []
        self.generate_calls = 0
        self.closed = False

    async def stream(self, request):  # type: ignore[no-untyped-def]
        self.calls.append(request)
        try:
            for i, delta in enumerate(self.deltas):
                if self.fail_after is not None and i == self.fail_after:
                    raise self.error
                await self.clock.sleep(self.gap_s)
                self.log.append(("delta", self.clock(), delta))
                yield delta
            self.log.append(("stream_end", self.clock(), ""))
        finally:
            self.closed = True

    async def generate(self, request) -> str:  # type: ignore[no-untyped-def]
        self.generate_calls += 1
        return "".join(self.deltas)


class TimedSynth:
    """Each chunk takes ``synth_s`` of virtual time to synthesize; audio plays ``play_s``."""

    def __init__(
        self,
        clock: TimelineClock,
        log: list[tuple[str, float, str]],
        synth_s: float = 1.0,
        play_s: float = 3.0,
        fail_on_call: int | None = None,
    ) -> None:
        self.clock, self.log = clock, log
        self.synth_s, self.play_s = synth_s, play_s
        self.fail_on_call = fail_on_call
        self.calls: list[str] = []

    async def synthesize_sentence(self, text: str) -> ReplyAudio:
        self.calls.append(text)
        self.log.append(("synth_start", self.clock(), text))
        await self.clock.sleep(self.synth_s)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise ReplySynthesisError("rvc gone")
        self.log.append(("synth_end", self.clock(), text))
        return ReplyAudio(pcm=text.encode(), duration_s=self.play_s)


class TimedPlayer:
    """Plays for the audio's duration of virtual time; ``stop`` ends it early."""

    def __init__(self, clock: TimelineClock, log: list[tuple[str, float, str]], play_s: float):
        self.clock, self.log, self.play_s = clock, log, play_s
        self.stop_calls = 0
        self.plays: list[bytes] = []
        self._stop = asyncio.Event()

    async def play(self, pcm: bytes) -> None:
        self.plays.append(pcm)
        self.log.append(("play_start", self.clock(), pcm.decode()))
        self._stop = asyncio.Event()
        sleeper = asyncio.ensure_future(self.clock.sleep(self.play_s))
        stopper = asyncio.ensure_future(self._stop.wait())
        try:
            await asyncio.wait([sleeper, stopper], return_when=asyncio.FIRST_COMPLETED)
        finally:
            sleeper.cancel()
            stopper.cancel()
        self.log.append(("play_end", self.clock(), pcm.decode()))

    def stop(self) -> None:
        self.stop_calls += 1
        self._stop.set()

    def is_playing(self) -> bool:
        return False


class FakePlayer:
    """Records plays; with ``hold`` a play blocks until ``stop()`` or ``release()``."""

    def __init__(self, hold: bool = False) -> None:
        self.hold = hold
        self.plays: list[bytes] = []
        self.stop_calls = 0
        self.fail_on: set[bytes] = set()
        self.stream_plays: list[list[bytes]] = []
        self._playing = False
        self._release = asyncio.Event()

    async def play_stream(self, chunks: AsyncIterator[bytes]) -> None:
        """Record the chunks of one stream (``hold`` blocks after the first, like ``play``)."""
        got: list[bytes] = []
        self.stream_plays.append(got)
        async with aclosing(chunks):
            async for pcm in chunks:
                got.append(pcm)
                if self.hold and len(got) == 1:
                    self._playing = True
                    try:
                        await self._release.wait()
                    finally:
                        self._playing = False
                        self._release = asyncio.Event()
                    return
                await asyncio.sleep(0)

    async def play(self, pcm: bytes) -> None:
        self.plays.append(pcm)
        if pcm in self.fail_on:
            raise RuntimeError("player boom")
        if not self.hold:
            await asyncio.sleep(0)
            return
        self._playing = True
        try:
            await self._release.wait()
        finally:
            self._playing = False
            self._release = asyncio.Event()

    def stop(self) -> None:
        self.stop_calls += 1
        self._release.set()

    def release(self) -> None:
        self._release.set()

    def is_playing(self) -> bool:
        return self._playing


class FakeReplySynth:
    """``synthesize_sentence`` returns the sentence text as the PCM bytes."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def synthesize_sentence(self, text: str) -> ReplyAudio:
        self.calls.append(text)
        await asyncio.sleep(0)
        if self.fail:
            raise ReplySynthesisError("rvc unavailable")
        return ReplyAudio(pcm=text.encode(), duration_s=1.0)


class FakeBackchannel:
    def __init__(self, empty: bool = False) -> None:
        self.empty = empty

    def __len__(self) -> int:
        return 0 if self.empty else 1

    def get_random(self) -> ReplyAudio:
        if self.empty:
            raise LookupError("none")
        return ReplyAudio(pcm=BACKCHANNEL_PCM, duration_s=0.5)


class UserTranscriber:
    """Transcript chosen by the user id encoded in the audio; optional per-user gates."""

    def __init__(self, texts: dict[int, str]) -> None:
        self.texts = texts
        self.gates: dict[int, asyncio.Event] = {}
        self.calls: list[int] = []

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        uid = int(audio.samples[0])
        self.calls.append(uid)
        gate = self.gates.get(uid)
        if gate is not None:
            await gate.wait()
        return Transcript(text=self.texts[uid], language="en")

    async def shutdown(self) -> None:
        return None


def utterance(
    user_id: int, t_start: float = 0.0, t_eos: float = 1.0, audio_s: float = 2.0
) -> UtteranceEvent:
    """An utterance of ``audio_s`` seconds (2.0 s: long enough for the early backchannel)."""
    audio = AudioChunk(np.full(round(16000 * audio_s), user_id, dtype=np.float32))
    return UtteranceEvent(user_id=user_id, audio=audio, t_start=t_start, t_eos=t_eos)


def speech(user_id: int, t_start: float) -> SpeechStarted:
    return SpeechStarted(user_id=user_id, t_start=t_start)


@dataclass
class Rig:
    manager: TurnManager
    clock: FakeClock
    transcriber: FakeTranscriber | UserTranscriber
    conversation: Conversation
    provider: FakeContentProvider
    synth: FakeReplySynth
    player: FakePlayer
    metrics: list[str] = field(default_factory=list)


def make_rig(
    *,
    reply: dict | None = None,
    respond_to: str = "all",
    transcriber: FakeTranscriber | UserTranscriber | None = None,
    provider: FakeContentProvider | None = None,
    synth: FakeReplySynth | None = None,
    player: FakePlayer | None = None,
    backchannel: FakeBackchannel | None = None,
    clock: FakeClock | None = None,
    names: dict[int, str] | None = None,
) -> Rig:
    reply_settings = ListenReplySettings(**({"per_user_cooldown_s": 0.0} | (reply or {})))
    settings = ListenSettings(respond_to=respond_to, reply=reply_settings)  # type: ignore[arg-type]
    clock = clock or FakeClock(start=100.0)
    tr = transcriber or FakeTranscriber(["tell me something fun"])
    conversation = Conversation(settings.reply.history_turns)
    prov = provider or FakeContentProvider(["Hey, that sounds awesome!"])
    sy = synth or FakeReplySynth()
    pl = player or FakePlayer()
    metrics: list[str] = []
    manager = TurnManager(
        settings,
        clock,
        tr,
        conversation,
        prov,
        sy,  # type: ignore[arg-type]
        backchannel,  # type: ignore[arg-type]
        pl,
        metrics.append,
        BOT_ID,
        guild_id=7,
        name_resolver=(lambda uid: (names or {}).get(uid, f"User {uid}")),
    )
    return Rig(manager, clock, tr, conversation, prov, sy, pl, metrics)


async def spin(times: int = 10) -> None:
    """Yield to the loop several times so ready tasks make progress."""
    for _ in range(times):
        await asyncio.sleep(0)


# ----------------------------------------------------------- streaming fakes


class StreamSynth:
    """Streaming ReplySynthesizer fake: ``stream_sentence`` yields ``n_chunks`` byte chunks.

    Each chunk takes ``chunk_s`` of virtual time (TimelineClock). ``log`` gets
    ``gen_start`` / ``first_pcm`` / ``gen_end`` / ``gen_closed`` entries; chunk
    payloads are ``b"<text>|<i>"`` so tests can tell sentences apart.
    """

    can_stream = True

    def __init__(
        self,
        clock: TimelineClock,
        log: list[tuple[str, float, str]],
        n_chunks: int = 3,
        chunk_s: float = 0.2,
        fail_on_call: int | None = None,
        fail_after_chunks: int = 0,
    ) -> None:
        self.clock, self.log = clock, log
        self.n_chunks, self.chunk_s = n_chunks, chunk_s
        self.fail_on_call, self.fail_after_chunks = fail_on_call, fail_after_chunks
        self.calls: list[str] = []
        self.closed = 0

    async def synthesize_sentence(self, text: str) -> ReplyAudio:  # not used when can_stream
        raise AssertionError("whole-sentence path used while streaming")

    async def stream_sentence(self, text: str) -> AsyncIterator[bytes]:
        self.calls.append(text)
        call = len(self.calls)
        self.log.append(("gen_start", self.clock(), text))
        try:
            for i in range(self.n_chunks):
                if self.fail_on_call == call and i == self.fail_after_chunks:
                    raise ReplySynthesisError("tts gone")
                await self.clock.sleep(self.chunk_s)
                if i == 0:
                    self.log.append(("first_pcm", self.clock(), text))
                yield f"{text}|{i}".encode()
            self.log.append(("gen_end", self.clock(), text))
        finally:
            self.closed += 1
            self.log.append(("gen_closed", self.clock(), text))


class StreamPlayer:
    """Records ``play_stream`` chunks; each chunk "plays" for ``chunk_play_s`` of virtual time.

    ``stop`` ends the stream early and closes the iterator, like VoicePlayer.
    """

    def __init__(
        self, clock: TimelineClock, log: list[tuple[str, float, str]], chunk_play_s: float = 0.5
    ) -> None:
        self.clock, self.log, self.chunk_play_s = clock, log, chunk_play_s
        self.stop_calls = 0
        self.plays: list[bytes] = []
        self.chunks: list[bytes] = []
        self._stop = asyncio.Event()

    async def play(self, pcm: bytes) -> None:
        self.plays.append(pcm)
        await asyncio.sleep(0)

    async def play_stream(self, chunks: AsyncIterator[bytes]) -> None:
        self._stop = asyncio.Event()
        async with aclosing(chunks):
            async for pcm in chunks:
                if self._stop.is_set():
                    break
                if not self.chunks:
                    self.log.append(("stream_start", self.clock(), pcm.decode()))
                self.chunks.append(pcm)
                self.log.append(("play_chunk", self.clock(), pcm.decode()))
                await self.clock.sleep(self.chunk_play_s)
        self.log.append(("stream_end", self.clock(), ""))

    def stop(self) -> None:
        self.stop_calls += 1
        self._stop.set()

    def is_playing(self) -> bool:
        return False
