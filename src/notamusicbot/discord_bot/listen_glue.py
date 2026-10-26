"""Discord glue for listen mode (plan sections 5-7; P9, task G1).

One :class:`ListenSession` per guild joins a voice channel with a
``VoiceRecvClient`` and wires the discord-free listen modules together:
sink -> ``ListenIngress`` -> ``Endpointer`` -> ``TurnManager`` -> reply
synthesis -> the same voice client's playback. Nothing here decides policy that
the listen modules already enforce; it only connects them and keeps the
consent rules at the Discord boundary:

* The sink drops ``None``, the bot and every user who has not opted in BEFORE
  the DAVE decrypt (and ``ListenIngress`` drops them again before opus decode).
* A member leaving the channel is opted out, never just forgotten.
* No audio or transcript touches disk; history lives in the in-memory
  ``Conversation`` and is cleared on stop.

discord-ext-voice-recv 0.5.2a179 has no DAVE support: with ``wants_opus()``
the frames it hands over are transport-decrypted but still DAVE-encrypted, so
:func:`dave_decrypt` runs the davey session decrypt itself (P0a spike).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import threading
import time
from collections.abc import AsyncIterator, Coroutine
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

import davey
import discord
from discord.ext import voice_recv

from notamusicbot.domain.listen import ConsentState, UtteranceEvent
from notamusicbot.listen.conversation import Conversation
from notamusicbot.listen.endpointer import Endpointer, EndpointerThread, ListenEvent
from notamusicbot.listen.ingress import ListenIngress
from notamusicbot.listen.offline import enforce_offline_models
from notamusicbot.listen.reply import BackchannelCache, ReplySynthesizer, warm_voice_converter
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
# Voice connect timeout, and the bound on the join announcement (synthesis + play).
CONNECT_TIMEOUT_S = 30.0
ANNOUNCEMENT_TIMEOUT_S = 60.0


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
    """Reader-thread sink: consent filter, DAVE decrypt, then the ingress."""

    def __init__(
        self,
        consent: ConsentState,
        ingress: ListenIngress,
        bot_user_id: int,
        get_connection: Any,
    ) -> None:
        super().__init__()
        self._consent = consent
        self._ingress = ingress
        self._bot_user_id = bot_user_id
        self._get_connection = get_connection
        self.dave_dropped = 0

    def wants_opus(self) -> bool:
        return True

    def cleanup(self) -> None:
        return None

    def write(self, user: Any, data: Any) -> None:
        # Consent first: no DAVE decrypt (and no decode) for anyone else.
        if user is None or user.id == self._bot_user_id or not self._consent.allowed(user.id):
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


# Streamed playback: 20 ms Discord frames of 48 kHz stereo s16le.
FRAME_BYTES = 3840
# Start playing once this much is buffered (6 frames = 120 ms), so a late TTS
# chunk does not stutter the first words.
PREBUFFER_FRAMES = 6
# The producer waits while this much is queued ahead of the player (0.5 s); it
# bounds how far the turn manager's "now playing" bookkeeping can run ahead.
MAX_BUFFER_FRAMES = 25
_BUFFER_POLL_S = 0.01
_SILENCE_FRAME = b"\x00" * FRAME_BYTES


class _StreamSource(discord.AudioSource):
    """Queue-fed audio source: fed by the event loop, read by discord's player thread.

    An underrun yields a silent frame instead of ending the stream; the stream
    ends only after :meth:`finish` and once the buffer has drained.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._finished = False
        self.underruns = 0

    def feed(self, pcm: bytes) -> None:
        with self._lock:
            self._buf += pcm

    def finish(self) -> None:
        with self._lock:
            self._finished = True

    def abort(self) -> None:
        """End now: drop what is buffered."""
        with self._lock:
            self._buf.clear()
            self._finished = True

    @property
    def buffered_frames(self) -> int:
        with self._lock:
            return len(self._buf) // FRAME_BYTES

    @property
    def buffered_bytes(self) -> int:
        with self._lock:
            return len(self._buf)

    def read(self) -> bytes:
        with self._lock:
            if len(self._buf) >= FRAME_BYTES:
                frame = bytes(self._buf[:FRAME_BYTES])
                del self._buf[:FRAME_BYTES]
                return frame
            if self._finished:
                if not self._buf:
                    return b""
                frame = bytes(self._buf).ljust(FRAME_BYTES, b"\x00")  # final partial frame
                self._buf.clear()
                return frame
            self.underruns += 1
            return _SILENCE_FRAME

    def is_opus(self) -> bool:
        return False


class VoicePlayer:
    """``ReplyPlayerPort`` over the guild's voice client (the recv client itself)."""

    def __init__(self, voice_client: Any) -> None:
        self._vc = voice_client
        self._producer: asyncio.Task[None] | None = None

    async def play_stream(self, chunks: AsyncIterator[bytes]) -> None:
        """Play ``chunks`` (48 kHz stereo s16le) as they arrive; see ``ReplyPlayerPort``.

        Playback starts after a short pre-buffer, pads underruns with silence and
        ends once ``chunks`` is exhausted and the buffer drained.
        """
        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()
        source = _StreamSource()
        errors: list[BaseException] = []

        def _resolve(error: BaseException | None) -> None:
            def _finish() -> None:
                if done.done():
                    return
                if error is not None:
                    done.set_exception(error)
                else:
                    done.set_result(None)

            with contextlib.suppress(RuntimeError):  # loop already closed
                loop.call_soon_threadsafe(_finish)

        def _after(error: Exception | None) -> None:
            _resolve(error)  # runs on discord's player thread

        async def _pump() -> None:
            started = False
            try:
                async with aclosing(chunks):
                    async for pcm in chunks:
                        source.feed(pcm)
                        if not started and source.buffered_frames >= PREBUFFER_FRAMES:
                            self._vc.play(source, after=_after)
                            started = True
                        while started and source.buffered_frames > MAX_BUFFER_FRAMES:
                            await asyncio.sleep(_BUFFER_POLL_S)
                source.finish()
                if not started and source.buffered_bytes:
                    self._vc.play(source, after=_after)  # short reply: below the pre-buffer
                    started = True
            except BaseException as exc:
                source.abort()
                if not isinstance(exc, asyncio.CancelledError):
                    errors.append(exc)
                raise
            finally:
                if not started:
                    _resolve(None)  # nothing will ever play: release play_stream

        producer = loop.create_task(_pump())
        self._producer = producer
        try:
            await done
        except asyncio.CancelledError:
            self.stop()
            raise
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            if self._producer is producer:
                self._producer = None
        if errors:
            raise errors[0]

    async def play(self, pcm: bytes) -> None:
        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()

        def _after(error: Exception | None) -> None:
            # Runs on discord's player thread.
            def _finish() -> None:
                if done.done():
                    return
                if error is not None:
                    done.set_exception(error)
                else:
                    done.set_result(None)

            with contextlib.suppress(RuntimeError):  # loop already closed
                loop.call_soon_threadsafe(_finish)

        self._vc.play(discord.PCMAudio(io.BytesIO(pcm)), after=_after)
        try:
            await done
        except asyncio.CancelledError:
            self.stop()
            raise

    def stop(self) -> None:
        producer = self._producer
        if producer is not None and not producer.done():
            producer.cancel()  # a stream in flight: stop feeding it
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
    ) -> None:
        self._app = app
        self._bot = bot
        self._guild = guild
        self._channel = channel
        self._started = False
        self._stopped = False
        self._consent: ConsentState | None = None
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
        """Load the models, join the channel, announce and start listening."""
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
        provider = app.listen_content_provider
        pipeline = app.pipeline
        profile = app.voice_profiles.get(listen.voice_profile)
        if provider is None or pipeline is None or profile is None:
            raise RuntimeError("listen mode is not ready (provider, pipeline or voice profile)")
        if self._guild.voice_client is not None:
            raise RuntimeError("the bot is already in a voice channel in this guild")

        # Local-only: no hub lookups, then the LLM, then ASR (once), then RVC.
        enforce_offline_models()
        await provider.ensure_spark_preferred()  # type: ignore[attr-defined]
        self._transcriber = build_transcriber(listen.asr, app.coordinator)
        load = getattr(self._transcriber, "load", None)
        if load is not None:
            await asyncio.to_thread(load)
        await warm_voice_converter(pipeline, profile)

        reply = listen.reply
        synth = ReplySynthesizer(
            pipeline,
            profile,
            artifact_root=app.settings.speech.output_dir,
            tts_max_seconds_per_word=reply.tts_max_seconds_per_word,
            tts_cap_floor_seconds=reply.tts_cap_floor_seconds,
        )
        backchannel = None
        if reply.backchannel_enabled and reply.backchannel_texts:
            directory = app.settings.speech.output_dir.parent / "listen" / "backchannel"
            backchannel = BackchannelCache(directory, reply.backchannel_texts, synth)
            try:
                await backchannel.prerender()
            except Exception:
                logger.warning("listen: backchannel pre-render failed", exc_info=True)

        self._vc = await self._channel.connect(cls=voice_recv.VoiceRecvClient, timeout=30.0)
        bot_id = self._bot.user.id
        self._consent = consent = ConsentState(bot_user_id=bot_id)
        self._conversation = Conversation(reply.history_turns)
        self._ingress = ListenIngress(consent, bot_id, _new_opus_decoder, listen.vad.ring_buffer_s)
        vad = SileroVad if listen.vad.backend == "silero" else EnergyVad
        self._endpointer = Endpointer(vad, listen.vad, time.monotonic, self._on_event)
        self._poller = EndpointerThread(
            self._ingress, self._endpointer, consent.opted_in, ENDPOINT_POLL_INTERVAL_S
        )
        # Opt-out also forgets the user's half-finished utterance, synchronously.
        consent.add_clear_hook(self._endpointer.discard)
        self._turns = TurnManager(
            listen,
            MonotonicClock(),
            consent,
            self._transcriber,
            self._conversation,
            app.listen_content_provider,  # type: ignore[arg-type]
            synth,
            backchannel,
            VoicePlayer(self._vc),
            logger.info,
            bot_id,
            guild_id=self._guild.id,
            persona_description=profile.persona.description,
            system_prompt=profile.persona.chat_prompt,
            name_resolver=self._display_name,
        )
        self._sink = ListenDaveSink(
            consent, self._ingress, bot_id, lambda: getattr(self._vc, "_connection", None)
        )
        self._vc.listen(self._sink)
        self._loop_task = asyncio.get_running_loop().create_task(self._endpoint_loop())

        await self._post_notice(listen.announcement_text)
        if listen.announce_on_join:
            await self._announce(synth, listen.announcement_text)

    async def _post_notice(self, text: str) -> None:
        try:
            await self._channel.send(text)
        except Exception:
            logger.warning("listen: could not post the notice in %s", self._channel.id)

    async def _announce(self, synth: ReplySynthesizer, text: str) -> None:
        try:
            async with asyncio.timeout(ANNOUNCEMENT_TIMEOUT_S):
                audio = await synth.synthesize_sentence(text)
                await VoicePlayer(self._vc).play(audio.pcm)
        except Exception:
            logger.warning("listen: join announcement failed", exc_info=True)

    # ----------------------------------------------------------- event wiring

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

    # --------------------------------------------------------------- consent

    def opt_in(self, user_id: int) -> None:
        if self._consent is not None and not self._stopped:
            self._consent.opt_in(user_id)

    def opt_out(self, user_id: int) -> None:
        """Opt out and clear the user's buffer and pending utterance before returning."""
        if self._consent is not None:
            self._consent.opt_out(user_id)

    def on_member_left(self, user_id: int) -> None:
        """A member left the channel. Call on the event loop (e.g. from on_voice_state_update)."""
        self.opt_out(user_id)

    def handle_voice_state_update(
        self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState
    ) -> None:
        """Voice-state hook for the bot: leaving opts out, the bot losing the channel stops."""
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
        # Consent goes first: no new audio is admitted while the rest unwinds.
        if self._consent is not None:
            self._consent.clear()
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
        """Channel, opted-in display names and turn state. Never transcripts."""
        ids = sorted(self._consent.opted_in()) if self._consent is not None else []
        names = ", ".join(self._display_name(i) for i in ids) or "nobody"
        state = self._turns.state.name if self._turns is not None else "STOPPED"
        if self._stopped:
            state = "STOPPED"
        return f"Listening in {self._channel.name}. Opted in: {names}. State: {state}."


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
