"""Listen-mode turn manager (plan sections 5 and 7).

One manager per guild owns the whole spoken-reply pipeline: consent re-checks,
ASR, the noise / echo / wake-word filters, the backchannel clip, the LLM call,
sentence-pipelined synthesis and playback, barge-in and the rate limits.
Everything here is local and in memory; nothing is written to disk and
nothing imports discord.py (the glue owns the voice client and calls in).

States: IDLE, LISTENING(user), THINKING(user), SPEAKING(reply_id). Only one
reply task exists at a time. Utterances that arrive while their turn is still
being assembled (any ASR in flight, or the turn not yet at the LLM) are merged
into ONE LLM call; later ones wait in a depth-1 pending slot.

The reply streams: the LLM text is cut into chunks as it arrives (StreamChunker),
the first chunk is synthesized while the LLM is still writing, and playback
drains the synthesized chunks back to back. The backchannel clip starts at end
of speech, before ASR, for utterances that are long enough to deserve a reply.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from enum import Enum, auto
from time import monotonic
from typing import TYPE_CHECKING, Any, Protocol

from notamusicbot.domain.listen import ListenTurnMetrics
from notamusicbot.listen.conversation import (
    StreamChunk,
    StreamChunker,
    build_chat_request,
    clean_for_speech,
    echo_guard,
    is_noise_transcript,
    split_reply_sentences,
    split_spoken_chunks,
)
from notamusicbot.listen.metrics import format_turn_metrics
from notamusicbot.listen.reply import ReplyAudio, ReplySynthesisError

if TYPE_CHECKING:
    from notamusicbot.config import ListenSettings
    from notamusicbot.content.base import ContentProvider, ContentRequest
    from notamusicbot.domain.listen import ConsentState, SpeechStarted, Transcript, UtteranceEvent
    from notamusicbot.listen.conversation import Conversation
    from notamusicbot.listen.reply import BackchannelCache, ReplySynthesizer
    from notamusicbot.speech.asr.base import Transcriber

logger = logging.getLogger(__name__)

_END = object()  # queue sentinel: no more chunks (finished or synthesis failed)

# Plan section 5: no barge-in during the first 300 ms of a backchannel clip.
BACKCHANNEL_BARGE_GUARD_S = 0.3
# Sliding window for ``max_replies_per_minute``.
RATE_WINDOW_S = 60.0
# The echo guard only applies this long after the bot last spoke.
ECHO_WINDOW_S = 15.0
# The prompt carries the last N history turns (plan P5); the rest is trimmed.
PROMPT_HISTORY_TURNS = 2
# Merged turns cap at this many distinct speakers (newest win).
MAX_MERGED_SPEAKERS = 3
# Synthesized-but-not-yet-playing chunks held ahead of playback (pipeline depth).
SYNTH_QUEUE_DEPTH = 2
# Streamed PCM chunks buffered per sentence while an earlier sentence plays
# (a TTS chunk is a few hundred ms, so this is far more than a sentence).
STREAM_CHUNK_BUFFER = 256
# The early backchannel needs at least this much speech (shorter utterances
# are often noise or one-word asides that never get a reply).
EARLY_BACKCHANNEL_MIN_AUDIO_S = 1.2
# Float slack when comparing clock deltas.
_EPS = 1e-9


class TurnState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


class ReplyPlayerPort(Protocol):
    """The voice-client side: plays 48 kHz stereo s16le PCM into the channel."""

    async def play(self, pcm: bytes) -> None:
        """Play ``pcm``; return when it finished or was stopped."""
        ...

    async def play_stream(self, chunks: AsyncIterator[bytes]) -> None:
        """Play PCM chunks as they arrive; return when ``chunks`` ended and drained.

        Used only when the reply synthesizer reports ``can_stream``. The player
        closes ``chunks`` when it is stopped or cancelled.
        """
        ...

    def stop(self) -> None:
        """Stop whatever is playing (no-op when idle)."""
        ...

    def is_playing(self) -> bool: ...


class ClockPort(Protocol):
    """Monotonic seconds when called; ``sleep`` may be simulated in tests."""

    def __call__(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class MonotonicClock:
    """Production clock: ``time.monotonic`` and a real ``asyncio.sleep``."""

    def __call__(self) -> float:
        return monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


@dataclass(slots=True)
class _Pending:
    """One accepted transcript waiting for (or inside) a turn."""

    user_id: int
    name: str
    text: str
    t_eos: float
    t_asr_done: float


class _StreamJob:
    """One sentence's streamed PCM: chunks, then ``_END`` or a ``_Failed`` marker."""

    __slots__ = ("q",)

    def __init__(self) -> None:
        self.q: asyncio.Queue[Any] = asyncio.Queue(maxsize=STREAM_CHUNK_BUFFER)


class _Failed:
    """Queued after the last good chunk of a sentence whose generation failed."""


@dataclass(slots=True)
class _ReplyFlow:
    """Per-turn plumbing between the LLM stream, synthesis and playback."""

    chunker: StreamChunker
    # (chunk text, 1-based sentence index) then _END; unbounded so the LLM never waits.
    chunk_q: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)
    # (ReplyAudio, chunk text, sentence index) then _END; bounded = pipeline depth.
    audio_q: asyncio.Queue[Any] = field(
        default_factory=lambda: asyncio.Queue(maxsize=SYNTH_QUEUE_DEPTH)
    )
    spoken: list[str] = field(default_factory=list)
    # Streaming reply: audio_q carries (_StreamJob, text, index) instead of ReplyAudio.
    streaming: bool = False
    audio_started: bool = False
    t_first_chunk: float | None = None
    llm_task: asyncio.Task[None] | None = None


class TurnManager:
    """Drives one guild's listen turns (see the module docstring)."""

    def __init__(
        self,
        settings: ListenSettings,
        clock: ClockPort,
        consent: ConsentState,
        transcriber: Transcriber,
        conversation: Conversation,
        provider: ContentProvider,
        reply_synth: ReplySynthesizer,
        backchannel: BackchannelCache | None,
        player: ReplyPlayerPort,
        metrics_sink: Callable[[str], None] | None,
        bot_user_id: int | None,
        *,
        guild_id: int | str = 0,
        persona_description: str = "",
        system_prompt: str = "",
        name_resolver: Callable[[int], str] | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._consent = consent
        self._transcriber = transcriber
        self._conversation = conversation
        self._provider = provider
        self._synth = reply_synth
        self._backchannel = backchannel
        self._player = player
        self._metrics_sink = metrics_sink
        self._bot_user_id = bot_user_id
        self._guild_id = guild_id
        self._persona = persona_description
        self._system_prompt = system_prompt
        self._name_of = name_resolver or (lambda user_id: f"User {user_id}")

        self._state = TurnState.IDLE
        self._state_user: int | None = None
        self._reply_id: str | None = None
        self._closed = False

        self._asr_inflight = 0
        self._asr_settled = asyncio.Event()
        self._asr_settled.set()
        self._pending: dict[int, _Pending] = {}
        self._reply_task: asyncio.Task[None] | None = None
        self._turn_counter = 0

        self._bc_task: asyncio.Task[None] | None = None
        self._bc_started_at: float | None = None
        self._bc_active = False
        # True from the early clip's start until the turn it belongs to ends
        # (or every utterance of it was dropped): one clip per turn.
        self._bc_claimed = False
        self._barge_candidates: dict[int, float] = {}
        self._barge_tasks: set[asyncio.Task[None]] = set()
        # 1-based index of the sentence being played; None before audio starts.
        self._playing_sentence: int | None = None
        self._synth_task: asyncio.Task[None] | None = None
        self._bot_turn_added = False

        self._last_bot_sentence = ""
        self._last_spoke_at: float | None = None
        self._last_reply_at: dict[int, float] = {}
        self._reply_times: list[float] = []

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> TurnState:
        return self._state

    @property
    def active_user_id(self) -> int | None:
        """The user of LISTENING / THINKING, else None."""
        return self._state_user

    @property
    def reply_id(self) -> str | None:
        """The id of the reply being spoken (SPEAKING only)."""
        return self._reply_id

    @property
    def reply_task(self) -> asyncio.Task[None] | None:
        return self._reply_task

    def _set_state(self, state: TurnState, user_id: int | None = None) -> None:
        self._state = state
        self._state_user = user_id
        if state is not TurnState.SPEAKING:
            self._reply_id = None

    def _reply_active(self) -> bool:
        return self._reply_task is not None and not self._reply_task.done()

    def _settle_state(self) -> None:
        """Pick the resting state when nothing is speaking."""
        if self._reply_active():
            return
        if not (self._asr_inflight or self._pending):
            self._bc_claimed = False  # nothing left that the clip could belong to
        if self._asr_inflight or self._pending:
            user = next(reversed(self._pending), self._state_user)
            self._set_state(TurnState.THINKING, user)
        elif self._barge_candidates:
            self._set_state(TurnState.LISTENING, next(iter(self._barge_candidates)))
        else:
            self._set_state(TurnState.IDLE)

    async def wait_idle(self) -> None:
        """Wait until no reply task is running (for tests and shutdown paths)."""
        while self._reply_active():
            task = self._reply_task
            assert task is not None
            await asyncio.wait([task])
            await asyncio.sleep(0)

    # ----------------------------------------------------------------- events

    async def on_speech_started(self, event: SpeechStarted) -> None:
        """Speech onset: LISTENING, and barge-in once the speech is sustained."""
        if self._closed or not self._allowed(event.user_id):
            return
        self._barge_candidates[event.user_id] = event.t_start
        if self._state is TurnState.IDLE:
            self._set_state(TurnState.LISTENING, event.user_id)
        if not self._can_barge_in():
            return
        remaining = self._barge_remaining(event.t_start)
        if remaining <= _EPS:
            self._barge_in(event.user_id)
            return
        task = asyncio.get_running_loop().create_task(
            self._barge_in_after(event.user_id, event.t_start, remaining)
        )
        self._barge_tasks.add(task)
        task.add_done_callback(self._barge_tasks.discard)

    async def on_utterance(self, event: UtteranceEvent) -> None:
        """One endpointed utterance: ASR, filters, then queue it for a turn."""
        user_id = event.user_id
        # Synchronous prefix: the user stopped talking, so the barge-in
        # candidate is over before any await.
        self._barge_candidates.pop(user_id, None)
        if self._closed:
            return
        # Round-2 finding: check consent BEFORE any ASR work.
        if not self._allowed(user_id):
            self._settle_state()
            return
        self._start_early_backchannel(event)
        if not self._reply_active():
            self._set_state(TurnState.THINKING, user_id)
        self._asr_inflight += 1
        self._asr_settled.clear()
        t_asr_start = self._clock()
        try:
            transcript = await self._transcribe(event)
        finally:
            self._asr_inflight -= 1
            if not self._asr_inflight:
                self._asr_settled.set()
        try:
            if transcript is None or self._closed:
                return
            t_asr_done = self._clock()
            text = transcript.text.strip()
            # Consent can be revoked while ASR ran: drop, never reply to it.
            accepted = self._accept(user_id, text, t_asr_done)
            self._log_asr_timing(event, transcript, t_asr_start, t_asr_done)
            if not accepted:
                return
            if self._settings.privacy.log_transcripts:
                logger.debug("listen transcript user=%s text=%r", user_id, text)
            entry = _Pending(user_id, self._name_of(user_id), text, event.t_eos, t_asr_done)
            self._pending.pop(user_id, None)  # same user replaces; order = arrival
            self._pending[user_id] = entry
            while len(self._pending) > MAX_MERGED_SPEAKERS:
                del self._pending[next(iter(self._pending))]
            if not self._reply_active():
                self._start_turn()
        finally:
            self._settle_state()

    def _log_asr_timing(
        self, event: UtteranceEvent, transcript: Transcript, t_asr_start: float, t_asr_done: float
    ) -> None:
        """One diagnostic INFO line per utterance that reached ASR (no text).

        ``asr_compute`` is the transcriber call as the loop sees it;
        ``asr_wait`` is the part of it outside the transcriber's own timer
        (thread-pool or loop-resume lag). ``emit->asr_start`` is the loop hop.
        """
        t_accepted = self._clock()
        t_emit = event.t_emit
        eos_to_emit = f"{t_emit - event.t_eos:.3f}" if t_emit is not None else "n/a"
        emit_to_start = f"{t_asr_start - t_emit:.3f}" if t_emit is not None else "n/a"
        compute = t_asr_done - t_asr_start
        logger.info(
            "listen asr timing user=%s eos->emit=%s emit->asr_start=%s asr_wait=%.3f "
            "asr_compute=%.3f asr_done->accepted=%.3f audio_s=%.3f",
            event.user_id,
            eos_to_emit,
            emit_to_start,
            max(0.0, compute - transcript.elapsed_s) if transcript.elapsed_s else 0.0,
            compute,
            t_accepted - t_asr_done,
            event.audio.duration_s,
        )

    async def _transcribe(self, event: UtteranceEvent) -> Transcript | None:
        try:
            return await asyncio.wait_for(
                self._transcriber.transcribe(event.audio), self._settings.asr.timeout_s
            )
        except TimeoutError:
            logger.warning("ASR timed out; utterance dropped")
        except Exception:
            logger.warning("ASR failed; utterance dropped", exc_info=True)
        return None

    # ---------------------------------------------------------------- filters

    def _allowed(self, user_id: int) -> bool:
        """Consent, plus the bot's own id even if ConsentState was built without it."""
        return user_id != self._bot_user_id and self._consent.allowed(user_id)

    def _accept(self, user_id: int, text: str, now: float) -> bool:
        if not self._allowed(user_id):
            return False
        if not text or is_noise_transcript(text):
            return False
        recently_spoke = self._last_spoke_at is not None and (
            now - self._last_spoke_at <= ECHO_WINDOW_S
        )
        if recently_spoke and echo_guard(text, self._last_bot_sentence):
            logger.info("Echo transcript dropped")
            return False
        if self._settings.respond_to == "wake_word" and not self._addressed(text, now):
            return False
        if self._cooldown_blocks(user_id, now):
            logger.info("Per-user cooldown: utterance dropped")
            return False
        return True

    def _addressed(self, text: str, now: float) -> bool:
        wake = self._settings.wake_word.strip().lower()
        if wake and wake in text.lower():
            return True
        return self._within_follow_up(now)

    def _within_follow_up(self, now: float) -> bool:
        window = self._settings.follow_up_window_s
        return self._last_spoke_at is not None and now - self._last_spoke_at <= window

    def _rate_limited(self, now: float) -> bool:
        self._reply_times = [t for t in self._reply_times if now - t < RATE_WINDOW_S]
        return len(self._reply_times) >= self._settings.reply.max_replies_per_minute

    def _cooldown_blocks(self, user_id: int, now: float) -> bool:
        cooldown = self._settings.reply.per_user_cooldown_s
        last = self._last_reply_at.get(user_id)
        return cooldown > 0.0 and last is not None and now - last < cooldown

    # ---------------------------------------------------------------- barge-in

    def _can_barge_in(self) -> bool:
        return self._state is TurnState.SPEAKING or self._bc_active

    def _barge_remaining(self, t_start: float) -> float:
        """Seconds until the speech starting at ``t_start`` counts as sustained."""
        start = t_start
        if self._bc_active and self._bc_started_at is not None:
            start = max(start, self._bc_started_at + BACKCHANNEL_BARGE_GUARD_S)
        needed = self._settings.reply.barge_in_min_speech_ms / 1000.0
        return start + needed - self._clock()

    async def _barge_in_after(self, user_id: int, t_start: float, delay: float) -> None:
        await self._clock.sleep(delay)
        if self._closed or self._barge_candidates.get(user_id) != t_start:
            return  # the user stopped (or a newer onset replaced this one)
        if not self._can_barge_in() or not self._allowed(user_id):
            return
        if self._barge_remaining(t_start) > _EPS:
            return
        self._barge_in(user_id)

    def _barge_in(self, user_id: int) -> None:
        """Stop playback exactly once, cancel the reply and go to LISTENING."""
        self._player.stop()
        if self._bot_turn_added and self._playing_sentence is not None:
            self._conversation.mark_interrupted(self._playing_sentence)
        if self._synth_task is not None and not self._synth_task.done():
            self._synth_task.cancel()  # no further synthesis after a barge-in
        task = self._reply_task
        if task is not None and not task.done():
            task.cancel()
        self._pending.clear()  # whatever was queued is stale; the user is talking
        self._bc_active = False
        self._set_state(TurnState.LISTENING, user_id)
        logger.info("Barge-in: reply stopped")

    # ------------------------------------------------------------------- turns

    def _start_turn(self) -> None:
        self._turn_counter += 1
        task = asyncio.get_running_loop().create_task(self._turn(f"t{self._turn_counter}"))
        self._reply_task = task

    async def _turn(self, turn_id: str) -> None:
        interrupted = False
        try:
            await self._run_turn(turn_id)
        except asyncio.CancelledError:
            interrupted = True
            raise
        except Exception:
            logger.warning("Listen turn failed", exc_info=True)
        finally:
            self._finish_turn(interrupted)

    def _finish_turn(self, cancelled: bool) -> None:
        if self._bc_task is not None and not self._bc_task.done():
            self._bc_task.cancel()
        self._bc_task = None
        self._bc_active = False
        self._bc_claimed = False
        self._playing_sentence = None
        self._bot_turn_added = False
        if self._state is TurnState.SPEAKING:
            self._set_state(TurnState.IDLE)
        # A depth-1 pending slot filled during this turn becomes the next turn,
        # unless the user is mid-sentence (barge-in) or we are shutting down.
        if self._pending and not self._closed and not cancelled:
            self._start_turn()
            return
        self._reply_task = None
        self._settle_state()

    async def _run_turn(self, turn_id: str) -> None:
        # Let every in-flight ASR finish so near-simultaneous speakers merge.
        while self._asr_inflight:
            await self._asr_settled.wait()
        lines = [p for p in self._pending.values() if self._allowed(p.user_id)]
        self._pending.clear()
        if not lines:
            return
        now = self._clock()
        if self._rate_limited(now):
            logger.info("Reply rate limit reached; turn dropped")
            return
        self._reply_times.append(now)
        for line in lines:
            self._last_reply_at[line.user_id] = now
        self._set_state(TurnState.THINKING, lines[-1].user_id)

        newest = max(lines, key=lambda p: p.t_eos)
        t_eos, t_asr_done = newest.t_eos, max(p.t_asr_done for p in lines)

        multi = len(lines) > 1
        speaker = " and ".join(p.name for p in lines)
        heard = "\n".join(f"{p.name}: {p.text}" for p in lines) if multi else lines[0].text
        request = build_chat_request(
            self._persona,
            speaker,
            heard,
            self._conversation.history()[-PROMPT_HISTORY_TURNS:],
            self._system_prompt,
        )
        for p in lines:
            self._conversation.add_user(p.name, p.text)

        # Three stages run concurrently: the LLM stream is cut into chunks,
        # chunks are synthesized in order, and this task plays the audio.
        reply = self._settings.reply
        flow = _ReplyFlow(
            chunker=StreamChunker(
                reply.max_sentences, reply.first_chunk_max_words, reply.later_chunk_max_words
            )
        )
        flow.streaming = bool(getattr(self._synth, "can_stream", False))
        if flow.streaming:
            # Jobs are enqueued once their first PCM exists and never block on the
            # player, so generation of sentence k+1 follows k's immediately.
            flow.audio_q = asyncio.Queue()
        loop = asyncio.get_running_loop()
        flow.llm_task = loop.create_task(self._consume_llm(request, flow))
        self._synth_task = loop.create_task(self._synthesize_stream(flow))
        try:
            first = await flow.audio_q.get()
            if first is _END:
                return  # LLM failed / said nothing / first chunk could not be synthesized
            # Streaming: the first job is queued with its first PCM chunk, so this
            # is "first PCM chunk ready" (llm_to_tts1_s) and tts1_to_audio is ~0.
            t_tts1_done = self._clock()
            assert flow.t_first_chunk is not None  # audio only exists after a chunk
            stamps = (t_eos, t_asr_done, flow.t_first_chunk, t_tts1_done)
            await self._speak(turn_id, flow, first, lines, stamps)
        finally:
            tasks = [t for t in (flow.llm_task, self._synth_task) if t is not None]
            self._synth_task = None
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _consume_llm(self, request: ContentRequest, flow: _ReplyFlow) -> None:
        """Producer: LLM text -> chunker -> ``flow.chunk_q`` (ends with ``_END``)."""
        try:
            await self._pump_llm(request, flow)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            logger.warning("LLM timed out; %s", "rest dropped" if flow.spoken else "turn dropped")
        except Exception:
            logger.warning(
                "LLM failed; %s",
                "rest dropped" if flow.spoken else "turn dropped",
                exc_info=True,
            )
        # A failure after chunks were queued keeps what is queued and drops the
        # unfinished remainder; a failure before any chunk drops the reply.
        flow.chunk_q.put_nowait(_END)

    async def _pump_llm(self, request: ContentRequest, flow: _ReplyFlow) -> None:
        timeout_s = self._settings.reply.llm_timeout_s
        stream = getattr(self._provider, "stream", None)
        if stream is None:
            async with asyncio.timeout(timeout_s):
                raw = await self._provider.generate(request)
            self._queue_chunks(flow, flow.chunker.feed_chunks(raw or ""))
        else:
            # The provider owns the first-delta and total deadlines; timeout_s
            # only guards a stream that goes silent between two deltas.
            async with aclosing(stream(request)) as deltas:
                it = deltas.__aiter__()
                while not flow.chunker.done:
                    try:
                        async with asyncio.timeout(timeout_s):
                            delta = await anext(it)
                    except StopAsyncIteration:
                        break
                    self._queue_chunks(flow, flow.chunker.feed_chunks(delta))
        self._queue_chunks(flow, flow.chunker.finish_chunks())

    def _queue_chunks(self, flow: _ReplyFlow, chunks: list[StreamChunk]) -> None:
        for chunk in chunks:
            if flow.t_first_chunk is None:
                flow.t_first_chunk = self._clock()
            flow.spoken.append(chunk.text)
            if self._bot_turn_added:
                self._conversation.extend_bot(chunk.text)
            flow.chunk_q.put_nowait((chunk.text, chunk.sentence_index))

    async def _synthesize_stream(self, flow: _ReplyFlow) -> None:
        """Producer: synthesize queued chunks in order into ``flow.audio_q``."""
        delivered = False
        try:
            while True:
                entry = await flow.chunk_q.get()
                if entry is _END:
                    break
                chunk, index = entry
                if flow.streaming:
                    await self._generate_stream(flow, chunk, index)
                    delivered = flow.audio_started
                    continue
                audio = await self._synth.synthesize_sentence(chunk)
                await flow.audio_q.put((audio, chunk, index))
                delivered = True
        except ReplySynthesisError:
            self._warn_synthesis("failed", delivered or flow.audio_started)
            self._stop_llm(flow)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._warn_synthesis("crashed", delivered or flow.audio_started, exc_info=True)
            self._stop_llm(flow)
        await flow.audio_q.put(_END)

    async def _generate_stream(self, flow: _ReplyFlow, chunk: str, index: int) -> None:
        """Generate one sentence's PCM into a job (sets ``flow.audio_started``).

        The job joins ``flow.audio_q`` with its first chunk. A failure after that
        marks the job failed (playback ends there, the rest of the reply is
        dropped) and re-raises for the caller's single warning.
        """
        job = _StreamJob()
        queued = False
        try:
            async with aclosing(self._synth.stream_sentence(chunk)) as pcm_chunks:
                async for pcm in pcm_chunks:
                    await job.q.put(pcm)
                    if not queued:
                        queued = flow.audio_started = True
                        flow.audio_q.put_nowait((job, chunk, index))
            if queued:
                job.q.put_nowait(_END)
        except asyncio.CancelledError:
            raise
        except Exception:
            if queued:
                with contextlib.suppress(asyncio.QueueFull):
                    job.q.put_nowait(_Failed())
            raise

    async def _stream_reply(self, flow: _ReplyFlow, first: Any) -> AsyncIterator[bytes]:
        """Every sentence's PCM back to back, as ONE stream for the player."""
        item: Any = first
        while item is not _END:
            job, _chunk, index = item
            self._playing_sentence = index
            self._last_bot_sentence = flow.chunker.sentence_text(index)
            self._last_spoke_at = self._clock()
            while True:
                pcm = await job.q.get()
                if pcm is _END:
                    break
                if isinstance(pcm, _Failed):
                    return  # generation failed mid-sentence: drop the rest
                yield pcm
            item = await flow.audio_q.get()

    @staticmethod
    def _warn_synthesis(what: str, delivered: bool, *, exc_info: bool = False) -> None:
        tail = "mid-reply; rest dropped" if delivered else "; reply dropped"
        logger.warning("Reply synthesis %s %s", what, tail, exc_info=exc_info)

    @staticmethod
    def _stop_llm(flow: _ReplyFlow) -> None:
        if flow.llm_task is not None and not flow.llm_task.done():
            flow.llm_task.cancel()  # nothing more will be synthesized

    async def _speak(
        self,
        turn_id: str,
        flow: _ReplyFlow,
        first: Any,
        lines: list[_Pending],
        stamps: tuple[float, float, float, float],
    ) -> None:
        t_eos, t_asr_done, t_first_chunk, t_tts1_done = stamps
        # The backchannel must finish before the reply audio starts.
        if self._bc_task is not None:
            await self._bc_task
            self._bc_task = None
        self._bc_active = False
        t_audio = self._clock()
        self._set_state(TurnState.SPEAKING)
        self._reply_id = turn_id
        # The history holds the spoken text: every chunk produced so far, and
        # each later chunk is appended as the LLM stream delivers it.
        self._conversation.add_bot(" ".join(flow.spoken))
        self._bot_turn_added = True

        user_id = lines[-1].user_id
        m = ListenTurnMetrics(
            turn_id=turn_id,
            user_id=user_id,
            eos_to_asr_s=max(0.0, t_asr_done - t_eos),
            # Time to the first sentence (chunk) being ready, not to the whole reply.
            asr_to_llm_s=max(0.0, t_first_chunk - t_asr_done),
            llm_to_tts1_s=max(0.0, t_tts1_done - t_first_chunk),
            tts1_to_audio_s=max(0.0, t_audio - t_tts1_done),
            total_s=max(0.0, t_audio - t_eos),
        )
        self._emit_metrics(m, user_id)

        if flow.streaming:
            stream = self._stream_reply(flow, first)
            try:
                async with aclosing(stream):
                    await self._player.play_stream(stream)
            finally:
                self._last_spoke_at = self._clock()
            return

        item: Any = first
        while item is not _END:
            audio, _chunk, index = item
            self._playing_sentence = index
            self._last_bot_sentence = flow.chunker.sentence_text(index)
            self._last_spoke_at = self._clock()
            await self._player.play(audio.pcm)
            self._last_spoke_at = self._clock()
            item = await flow.audio_q.get()

    def _start_early_backchannel(self, event: UtteranceEvent) -> None:
        """Play the clip at end of speech, before ASR, for a turn that will likely reply."""
        if self._state is TurnState.SPEAKING or self._bc_claimed or self._bc_active:
            return
        if event.audio.duration_s < EARLY_BACKCHANNEL_MIN_AUDIO_S - _EPS:
            return
        now = self._clock()
        if self._cooldown_blocks(event.user_id, now) or self._rate_limited(now):
            return
        if self._settings.respond_to == "wake_word" and not self._within_follow_up(now):
            return  # the wake word is only known after ASR
        task = self._start_backchannel()
        if task is not None:
            self._bc_task = task
            self._bc_claimed = True

    def _start_backchannel(self) -> asyncio.Task[None] | None:
        if not self._settings.reply.backchannel_enabled or self._backchannel is None:
            return None
        try:
            clip = self._backchannel.get_random()
        except LookupError:
            return None
        self._bc_active = True
        self._bc_started_at = self._clock()
        return asyncio.get_running_loop().create_task(self._play_backchannel(clip))

    async def _play_backchannel(self, clip: ReplyAudio) -> None:
        try:
            await self._player.play(clip.pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Backchannel playback failed", exc_info=True)
        finally:
            self._bc_active = False

    def _emit_metrics(self, m: ListenTurnMetrics, user_id: int) -> None:
        line = format_turn_metrics(m, self._guild_id, user_id)
        if self._metrics_sink is None:
            logger.info(line)
            return
        try:
            self._metrics_sink(line)
        except Exception:
            logger.warning("Metrics sink failed", exc_info=True)

    # ---------------------------------------------------------------- shutdown

    async def shutdown(self) -> None:
        """Cancel every task, stop the player and drop the history."""
        self._closed = True
        tasks: list[asyncio.Task[Any]] = list(self._barge_tasks)
        for task in (self._reply_task, self._bc_task):
            if task is not None:
                tasks.append(task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._barge_candidates.clear()
        self._pending.clear()
        self._bc_active = False
        self._reply_task = None
        self._bc_task = None
        self._player.stop()
        self._conversation.clear()
        self._set_state(TurnState.IDLE)


def _chunk_plan(raw: str, max_sentences: int) -> list[tuple[str, int, str]]:
    """Spoken chunks as ``(chunk, 1-based sentence index, sentence text)``.

    Chunks never cross a sentence boundary, so walking the word counts of the
    sentence units assigns every chunk to the sentence it was cut from.
    """
    chunks = split_spoken_chunks(raw, max_sentences)
    units = split_reply_sentences(clean_for_speech(raw))
    if max_sentences >= 1:
        units = units[:max_sentences]
    plan: list[tuple[str, int, str]] = []
    index = 1
    used = 0  # words of units[index - 1] already covered by earlier chunks
    for chunk in chunks:
        while index < len(units) and used >= len(units[index - 1].split()):
            index += 1
            used = 0
        used += len(chunk.split())
        plan.append((chunk, index, units[index - 1] if units else chunk))
    return plan
