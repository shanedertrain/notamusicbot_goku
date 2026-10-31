"""Discord glue for listen mode (plan sections 5-7; P9, task G1).

One :class:`ListenSession` per guild joins a voice channel with a
``VoiceRecvClient`` and wires the discord-free listen modules together:
sink -> ``ListenIngress`` -> ``Endpointer`` -> ``TurnManager`` -> reply
synthesis -> the same voice client's playback. Nothing here decides policy that
the listen modules already enforce; it only connects them and keeps the
self-hearing filter at the Discord boundary:

* The sink drops ``None`` and the bot's own id BEFORE the DAVE decrypt (and
  ``ListenIngress`` drops them again before opus decode).
* A member leaving the channel has their buffered audio cleared at once.
* No audio or transcript touches disk; history lives in the in-memory
  ``Conversation`` and is cleared on stop.
* ``listen.listen_only`` (or ``!glisten transcript``) skips the reply half
  entirely and writes NOTHING to Discord: ASR still runs and each accepted
  line goes to the local transcript log (storage forced on). No TTS, no LLM.
  A listen-only start also posts no start message; the record is the log file.

discord-ext-voice-recv 0.5.2a179 has no DAVE support: with ``wants_opus()``
the frames it hands over are transport-decrypted but still DAVE-encrypted, so
:func:`dave_decrypt` runs the davey session decrypt itself (P0a spike).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any

import davey
import discord
from discord.ext import voice_recv

from notamusicbot.domain.listen import UtteranceEvent
from notamusicbot.listen.addressing import AddressJudge
from notamusicbot.listen.conversation import Conversation
from notamusicbot.listen.endpointer import Endpointer, EndpointerThread, ListenEvent
from notamusicbot.listen.ingress import ListenIngress
from notamusicbot.listen.offline import enforce_offline_models
from notamusicbot.listen.reply import BackchannelCache, ReplySynthesizer
from notamusicbot.listen.transcript_store import TranscriptStore
from notamusicbot.listen.turn_taking import MonotonicClock, TurnManager
from notamusicbot.speech.asr.factory import build_transcriber
from notamusicbot.speech.vad.energy import EnergyVad
from notamusicbot.speech.vad.silero import SileroVad

if TYPE_CHECKING:
    from notamusicbot.app import Application
    from notamusicbot.discord_bot.bot import NotAMusicBot

logger = logging.getLogger(__name__)

# The error davey raises for a frame that arrived unencrypted (3-9% of frames
# in the P0a spike): the payload is already plain opus, use it as it is.
PASSTHROUGH_ERROR = "UnencryptedWhenPassthroughDisabled"
# How often the endpointer loop polls the ingress when it found no audio.
ENDPOINT_POLL_INTERVAL_S = 0.01
# Voice connect timeout.
CONNECT_TIMEOUT_S = 30.0


def dave_decrypt(
    dave_session: Any | None, protocol_version: int, user_id: int, opus: bytes
) -> bytes | None:
    """Plain opus for ``opus``, or ``None`` when the frame must be dropped.

    Without a DAVE session (or at protocol version 0) the payload is already
    plain. A frame that arrived unencrypted is used as received; any other
    decrypt failure drops the frame.
    """
    if protocol_version <= 0 or dave_session is None:
        return opus
    try:
        return dave_session.decrypt(user_id, davey.MediaType.audio, opus)
    except ValueError as exc:
        if PASSTHROUGH_ERROR in str(exc):
            return opus
        return None
    except Exception:
        return None


class ListenDaveSink(voice_recv.AudioSink):
    """Reader-thread sink: self filter, DAVE decrypt, then the ingress."""

    def __init__(
        self,
        ingress: ListenIngress,
        bot_user_id: int,
        get_connection: Any,
    ) -> None:
        super().__init__()
        self._ingress = ingress
        self._bot_user_id = bot_user_id
        self._get_connection = get_connection
        self.dave_dropped = 0

    def wants_opus(self) -> bool:
        return True

    def cleanup(self) -> None:
        return None

    def write(self, user: Any, data: Any) -> None:
        # Drop only None and the bot's own id: no DAVE decrypt (and no decode) for them.
        if user is None or user.id == self._bot_user_id:
            return
        opus = data.opus
        if not opus:
            return
        conn = self._get_connection()
        payload = dave_decrypt(
            getattr(conn, "dave_session", None),
            getattr(conn, "dave_protocol_version", 0),
            user.id,
            opus,
        )
        if payload is None:
            self.dave_dropped += 1
            if self.dave_dropped == 1:
                logger.warning("listen: first undecryptable DAVE frame dropped")
            return
        self._ingress.accept_packet(user.id, payload)


# One Discord audio frame: 20 ms of 48 kHz stereo s16le.
FRAME_BYTES = 3840
# How long the source keeps emitting silence for the next chunk before it ends.
SOURCE_GRACE_FRAMES = 25


class _SourceClosed(Exception):
    """The audio source already ended; start a new one."""


def frame_align(pcm: bytes) -> bytes:
    """Whole 20 ms frames only: a partial tail is dropped (it is TTS decay, near silence).

    Padding the tail instead would put up to 19 ms of silence at every chunk seam.
    Clips shorter than one frame are padded to one.
    """
    if len(pcm) < FRAME_BYTES:
        return pcm.ljust(FRAME_BYTES, b"\x00")
    return pcm[: len(pcm) - len(pcm) % FRAME_BYTES]


class ChunkedSource(discord.AudioSource):
    """One continuous PCM stream that chunks are appended to while it plays.

    Starting a fresh ``voice_client.play`` per chunk leaves a gap at every seam
    (the previous player thread ending, the next one starting). Here the same
    source keeps running: the next chunk's frames follow the previous one's
    directly, and a short stretch of silence bridges a chunk that is a moment
    late. ``feed`` runs on the event loop, ``read`` on discord's player thread.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, *, grace_frames: int = SOURCE_GRACE_FRAMES):
        self._loop = loop
        self._grace = grace_frames
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._appended = 0
        self._consumed = 0
        self._idle = 0
        self._pending: deque[tuple[int, asyncio.Future[None]]] = deque()
        self.closed = False

    def feed(self, pcm: bytes) -> asyncio.Future[None]:
        """Queue frame-aligned ``pcm``; the future resolves when its last frame is read."""
        with self._lock:
            if self.closed:
                raise _SourceClosed
            fut: asyncio.Future[None] = self._loop.create_future()
            self._buf += pcm
            self._appended += len(pcm)
            self._idle = 0
            self._pending.append((self._appended, fut))
            return fut

    def read(self) -> bytes:
        with self._lock:
            if self._buf:
                frame = bytes(self._buf[:FRAME_BYTES])
                del self._buf[:FRAME_BYTES]
                self._consumed += FRAME_BYTES
                self._idle = 0
                while self._pending and self._pending[0][0] <= self._consumed:
                    self._settle(self._pending.popleft()[1], None)
                return frame
            self._idle += 1
            if self._idle > self._grace:
                self.closed = True
                return b""
            return b"\x00" * FRAME_BYTES

    def is_opus(self) -> bool:
        return False

    def finish(self, error: BaseException | None) -> None:
        """The player ended (or stopped): release every waiting ``play`` call."""
        with self._lock:
            self.closed = True
            self._buf.clear()
            pending, self._pending = list(self._pending), deque()
        for _end, fut in pending:
            self._settle(fut, error)

    def _settle(self, fut: asyncio.Future[None], error: BaseException | None) -> None:
        def _set() -> None:
            if fut.done():
                return
            if error is not None:
                fut.set_exception(error)
            else:
                fut.set_result(None)

        with contextlib.suppress(RuntimeError):  # loop already closed
            self._loop.call_soon_threadsafe(_set)


def _mark_ended(ended: asyncio.Future[None]) -> None:
    if not ended.done():
        ended.set_result(None)


class VoicePlayer:
    """``ReplyPlayerPort`` over the guild's voice client (the recv client itself).

    Successive ``play`` calls feed one :class:`ChunkedSource`, so a reply played in
    several chunks is one continuous stream.
    """

    def __init__(self, voice_client: Any) -> None:
        self._vc = voice_client
        self._source: ChunkedSource | None = None
        self._ended: asyncio.Future[None] | None = None

    async def _feed(self, pcm: bytes) -> asyncio.Future[None]:
        """Queue frame-aligned ``pcm`` on the live source (starting one if needed)."""
        loop = asyncio.get_running_loop()
        if self._source is not None:
            with contextlib.suppress(_SourceClosed):
                return self._source.feed(pcm)
        if self._ended is not None and not self._ended.done():
            await self._ended  # the previous player thread is still winding down
        source = ChunkedSource(loop, grace_frames=SOURCE_GRACE_FRAMES)
        ended: asyncio.Future[None] = loop.create_future()
        self._source, self._ended = source, ended
        fut = source.feed(pcm)

        def _after(error: Exception | None) -> None:
            # Runs on discord's player thread.
            source.finish(error)
            with contextlib.suppress(RuntimeError):  # loop already closed
                loop.call_soon_threadsafe(_mark_ended, ended)

        self._vc.play(source, after=_after)
        return fut

    async def play(self, pcm: bytes) -> None:
        fut = await self._feed(frame_align(pcm))
        try:
            await fut
        except asyncio.CancelledError:
            self.stop()
            raise

    async def play_stream(self, pieces: AsyncIterator[bytes]) -> None:
        """Feed ``pieces`` as they arrive; return when the last frame has been played.

        A piece rarely ends on a frame boundary, so the remainder rides along with
        the next one (nothing is dropped); only the very end is padded with silence.
        """
        carry = b""
        last: asyncio.Future[None] | None = None
        try:
            async for piece in pieces:
                data = carry + piece
                usable = len(data) - len(data) % FRAME_BYTES
                carry = data[usable:]
                if usable:
                    last = await self._feed(data[:usable])
            if carry:
                last = await self._feed(carry + b"\x00" * (FRAME_BYTES - len(carry)))
            if last is not None:
                await last
        except asyncio.CancelledError:
            self.stop()
            raise

    def stop(self) -> None:
        # VoiceRecvClient.stop() also stops RECEIVING; stop only the playback.
        stop = getattr(self._vc, "stop_playing", None) or self._vc.stop
        stop()

    def is_playing(self) -> bool:
        return bool(self._vc.is_playing())


def _new_opus_decoder() -> discord.opus.Decoder:
    return discord.opus.Decoder()


class ListenSession:
    """One guild's listen session: connection, pipeline wiring and teardown."""

    def __init__(
        self,
        app: Application,
        bot: NotAMusicBot,
        guild: discord.Guild,
        channel: discord.VoiceChannel | discord.StageChannel,
        *,
        listen_only: bool = False,
    ) -> None:
        self._app = app
        self._bot = bot
        self._guild = guild
        self._channel = channel
        # Per-call listen-only override (``!glisten transcript``); False keeps
        # the configured ``listen.listen_only`` default untouched.
        self._listen_only_override: bool | None = True if listen_only else None
        self._started = False
        self._stopped = False
        self._conversation: Conversation | None = None
        self._ingress: ListenIngress | None = None
        self._endpointer: Endpointer | None = None
        self._poller: EndpointerThread | None = None
        self._turns: TurnManager | None = None
        self._transcriber: Any = None
        self._vc: Any = None
        self._sink: ListenDaveSink | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._event_tasks: set[asyncio.Task[None]] = set()

    @property
    def channel_id(self) -> int:
        return self._channel.id

    @property
    def guild_id(self) -> int:
        return self._guild.id

    # ------------------------------------------------------------------ start

    async def start(self) -> None:
        """Load the models, join the channel, post the start message, listen."""
        if self._started:
            raise RuntimeError("listen session already started")
        self._started = True
        try:
            await self._start()
        except BaseException:
            await self.stop()
            raise

    async def _start(self) -> None:
        app = self._app
        listen = app.settings.listen
        # ``!glisten transcript`` forces listen-only regardless of the config;
        # the config flag alone keeps the standing default. The override is
        # applied ON the settings copy so the turn manager sees it too.
        if self._listen_only_override and not listen.listen_only:
            listen = listen.model_copy(update={"listen_only": True})
        listen_only = listen.listen_only
        provider = app.listen_content_provider
        pipeline = app.pipeline
        profile = app.voice_profiles.get(listen.voice_profile)
        # Listen-only needs no provider and no pipeline: it only transcribes.
        required = (profile,) if listen_only else (provider, pipeline, profile)
        if any(part is None for part in required):
            raise RuntimeError("listen mode is not ready (provider, pipeline or voice profile)")
        if self._guild.voice_client is not None:
            raise RuntimeError("the bot is already in a voice channel in this guild")

        # Local-only: no hub lookups, then the LLM, then ASR (once).
        enforce_offline_models()
        # Listen-only never replies: no provider warm-up, no TTS, no backchannel.
        if not listen_only:
            await provider.ensure_spark_preferred()  # type: ignore[attr-defined]
        self._transcriber = build_transcriber(listen.asr, app.coordinator)
        load = getattr(self._transcriber, "load", None)
        if load is not None:
            await asyncio.to_thread(load)

        reply = listen.reply
        synth: ReplySynthesizer | None = None
        backchannel = None
        if not listen_only:
            synth = ReplySynthesizer(
                pipeline,
                profile,
                artifact_root=app.settings.speech.output_dir,
                tts_max_seconds_per_word=reply.tts_max_seconds_per_word,
                tts_cap_floor_seconds=reply.tts_cap_floor_seconds,
            )
            if reply.backchannel_enabled and reply.backchannel_texts:
                directory = app.settings.speech.output_dir.parent / "listen" / "backchannel"
                backchannel = BackchannelCache(directory, reply.backchannel_texts, synth)
                try:
                    await backchannel.prerender()
                except Exception:
                    logger.warning("listen: backchannel pre-render failed", exc_info=True)

        self._vc = await self._channel.connect(cls=voice_recv.VoiceRecvClient, timeout=30.0)
        bot_id = self._bot.user.id
        self._conversation = Conversation(reply.history_turns)
        self._ingress = ListenIngress(bot_id, _new_opus_decoder, listen.vad.ring_buffer_s)
        vad = SileroVad if listen.vad.backend == "silero" else EnergyVad
        self._endpointer = Endpointer(vad, listen.vad, time.monotonic, self._on_event)
        self._poller = EndpointerThread(self._ingress, self._endpointer, ENDPOINT_POLL_INTERVAL_S)
        self._turns = TurnManager(
            listen,
            MonotonicClock(),
            self._transcriber,
            self._conversation,
            app.listen_content_provider,  # type: ignore[arg-type]
            synth,  # type: ignore[arg-type]
            backchannel,
            VoicePlayer(self._vc),
            logger.info,
            bot_id,
            guild_id=self._guild.id,
            persona_description=profile.persona.description,
            system_prompt=profile.persona.chat_prompt,
            name_resolver=self._display_name,
            transcript_store=self._open_transcript_store(force=listen_only),
            character_name=profile.persona.display_name,
            human_count=self._human_count,
            address_judge=self._build_judge(
                app.listen_content_provider, profile.persona.display_name
            ),
        )
        self._sink = ListenDaveSink(
            self._ingress, bot_id, lambda: getattr(self._vc, "_connection", None)
        )
        self._vc.listen(self._sink)
        self._loop_task = asyncio.get_running_loop().create_task(self._endpoint_loop())

        if not listen_only:
            await self._post_notice(listen.start_message)

    def _open_transcript_store(self, *, force: bool = False) -> TranscriptStore | None:
        privacy = self._app.settings.listen.privacy
        if not (force or privacy.store_transcripts):
            return None
        return TranscriptStore(Path(privacy.transcript_dir), privacy.transcript_retention_days)

    async def _post_notice(self, text: str) -> None:
        try:
            await self._channel.send(text)
        except Exception:
            logger.warning("listen: could not post the start message in %s", self._channel.id)

    # ----------------------------------------------------------- event wiring

    def _build_judge(self, provider: Any, character: str) -> Any:
        return AddressJudge(provider, character=character).addressed

    def _human_count(self) -> int:
        """Humans (non-bots) in the voice channel the bot is in."""
        channel = getattr(self._vc, "channel", None)
        return sum(1 for m in getattr(channel, "members", ()) if not m.bot)

    def _display_name(self, user_id: int) -> str:
        member = self._guild.get_member(user_id)
        return member.display_name if member is not None else f"User {user_id}"

    async def _endpoint_loop(self) -> None:
        """Poll the ingress on the loop; a failed poll is logged and polling goes on."""
        assert self._poller is not None
        while not self._stopped:
            try:
                fed = self._poller.poll_once()
            except Exception:
                logger.exception("listen endpointer poll failed; continuing")
                fed = 0
            await asyncio.sleep(0 if fed else ENDPOINT_POLL_INTERVAL_S)

    def _on_event(self, event: ListenEvent) -> None:
        """Endpointer emit (runs inside ``poll_once``, on the loop)."""
        turns = self._turns
        if turns is None or self._stopped:
            return
        if isinstance(event, UtteranceEvent):
            self._spawn(turns.on_utterance(event))
        else:
            self._spawn(turns.on_speech_started(event))

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.get_running_loop().create_task(self._guarded(coro))
        self._event_tasks.add(task)
        task.add_done_callback(self._event_tasks.discard)

    @staticmethod
    async def _guarded(coro: Coroutine[Any, Any, None]) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("listen turn handler failed")

    # ----------------------------------------------------------- member leave

    def on_member_left(self, user_id: int) -> None:
        """A member left the channel: drop their buffer and pending utterance.

        Call on the event loop (e.g. from on_voice_state_update). No-op before
        the pipeline exists or once the session has stopped.
        """
        if self._stopped or self._ingress is None or self._endpointer is None:
            return
        self._ingress.remove_user(user_id)
        self._endpointer.discard(user_id)

    def handle_voice_state_update(
        self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState
    ) -> None:
        """Voice-state hook: a member leaving clears buffers; the bot losing the channel stops."""
        left = before.channel is not None and before.channel.id == self._channel.id
        if not left or (after.channel is not None and after.channel.id == self._channel.id):
            return
        if member.id == self._bot.user.id:
            if not self._stopped:
                asyncio.get_running_loop().create_task(self.stop())
            return
        self.on_member_left(member.id)

    # ------------------------------------------------------------------ stop

    async def stop(self) -> None:
        """Stop listening, cancel every task, clear all state and disconnect (idempotent)."""
        if self._stopped:
            return
        self._stopped = True
        if self._vc is not None:
            with contextlib.suppress(Exception):
                self._vc.stop_listening()
        tasks = list(self._event_tasks)
        if self._loop_task is not None:
            tasks.append(self._loop_task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._event_tasks.clear()
        self._loop_task = None
        if self._turns is not None:
            with contextlib.suppress(Exception):
                await self._turns.shutdown()
        if self._conversation is not None:
            self._conversation.clear()
        if self._transcriber is not None:
            with contextlib.suppress(Exception):
                await self._transcriber.shutdown()
        if self._vc is not None:
            with contextlib.suppress(Exception):
                VoicePlayer(self._vc).stop()
            try:
                await self._vc.disconnect(force=True)
            except Exception:
                logger.warning("listen: voice disconnect failed", exc_info=True)

    # ---------------------------------------------------------------- status

    def status(self) -> str:
        """Channel and turn state. Never transcripts."""
        state = self._turns.state.name if self._turns is not None else "STOPPED"
        if self._stopped:
            state = "STOPPED"
        return f"Listening in {self._channel.name}. State: {state}."


class ListenSessionRegistry:
    """Active listen sessions by guild id."""

    def __init__(self) -> None:
        self._sessions: dict[int, ListenSession] = {}

    def get(self, guild_id: int) -> ListenSession | None:
        return self._sessions.get(guild_id)

    def add(self, session: ListenSession) -> None:
        guild_id = session.guild_id
        if guild_id in self._sessions:
            raise RuntimeError(f"a listen session already exists for guild {guild_id}")
        self._sessions[guild_id] = session

    def remove(self, guild_id: int) -> None:
        self._sessions.pop(guild_id, None)

    def active(self, guild_id: int) -> bool:
        return guild_id in self._sessions
