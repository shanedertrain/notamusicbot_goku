"""ListenSession / registry / player adapter with fakes (no Discord connection, no models)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _listen_turn_fakes import FakeBackchannel, FakeReplySynth

from conftest import FakeContentProvider, FakeTranscriber
from notamusicbot.config import (
    AppSettings,
    ListenReplySettings,
    ListenSettings,
    ListenVadSettings,
    SpeechSettings,
)
from notamusicbot.discord_bot import listen_glue
from notamusicbot.discord_bot.listen_glue import (
    ListenSession,
    ListenSessionRegistry,
    VoicePlayer,
)
from notamusicbot.domain.voice_profiles import (
    AudioProfile,
    Persona,
    TtsProfile,
    VoiceConversionProfileModel,
    VoiceProfile,
)

BOT_ID = 999
PACKET_SAMPLES = 960  # 20 ms at 48 kHz


class FakeDecoder:
    """Payload ``b"T..."`` decodes to a loud tone, anything else to silence."""

    def decode(self, data: bytes) -> bytes:
        n = np.arange(PACKET_SAMPLES)
        amp = 0.4 if data.startswith(b"T") else 0.0
        mono = (amp * 32767 * np.sin(2 * np.pi * 440 * n / 48000)).astype("<i2")
        return np.repeat(mono, 2).tobytes()


class FakeVoiceClient:
    def __init__(self) -> None:
        self.sink: Any = None
        self.listening = False
        self.playing = False
        self.plays: list[bytes] = []
        self.stop_calls = 0
        self.stop_playing_calls = 0
        self.disconnects = 0
        self.hold = False
        self._after: Any = None
        self._connection = SimpleNamespace(dave_session=None, dave_protocol_version=0)

    def listen(self, sink: Any) -> None:
        self.sink = sink
        self.listening = True

    def stop_listening(self) -> None:
        self.listening = False

    def play(self, source: Any, *, after: Any) -> None:
        self.plays.append(source.read())
        self.playing = True
        if self.hold:
            self._after = after
            return
        self.playing = False
        after(None)

    def finish(self, error: Exception | None = None) -> None:
        self.playing = False
        self._after(error)

    def stop(self) -> None:
        self.stop_calls += 1
        self.stop_playing()
        self.stop_listening()

    def stop_playing(self) -> None:
        self.stop_playing_calls += 1
        if self._after is not None:
            after, self._after = self._after, None
            self.playing = False
            after(None)

    def is_playing(self) -> bool:
        return self.playing

    async def disconnect(self, *, force: bool = False) -> None:
        self.disconnects += 1


class FakeChannel:
    def __init__(self, vc: FakeVoiceClient, guild: Any) -> None:
        self.id = 55
        self.name = "general-voice"
        self.guild = guild
        self.vc = vc
        self.sent: list[str] = []

    async def connect(self, *, cls: Any, timeout: float) -> FakeVoiceClient:
        return self.vc

    async def send(self, text: str) -> None:
        self.sent.append(text)


class PrerenderBackchannel(FakeBackchannel):
    async def prerender(self) -> int:
        return 0


class FakeProvider(FakeContentProvider):
    def __init__(self) -> None:
        super().__init__(["Hey, that sounds awesome!"])
        self.preferred_calls = 0

    async def ensure_spark_preferred(self) -> bool:
        self.preferred_calls += 1
        return False


def _profile() -> VoiceProfile:
    return VoiceProfile(
        id="goku",
        persona=Persona(display_name="Goku", description="a cheerful fighter"),
        tts=TtsProfile(speaker="x"),
        voice_conversion=VoiceConversionProfileModel(),
        audio=AudioProfile(),
    )


class Rig(SimpleNamespace):
    session: ListenSession
    vc: FakeVoiceClient
    channel: FakeChannel
    provider: FakeProvider
    synth: FakeReplySynth
    transcriber: FakeTranscriber
    order: list[str]


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    order: list[str] = []
    synth = FakeReplySynth()
    transcriber = FakeTranscriber(["tell me something fun"])

    def build_transcriber(settings: Any, coordinator: Any) -> FakeTranscriber:
        order.append("asr")
        transcriber.load = lambda: order.append("asr_load")  # type: ignore[attr-defined]
        return transcriber

    async def warm(pipeline: Any, profile: Any) -> bool:
        order.append("rvc_warm")
        return True

    monkeypatch.setattr(listen_glue, "build_transcriber", build_transcriber)
    monkeypatch.setattr(listen_glue, "warm_voice_converter", warm)
    monkeypatch.setattr(listen_glue, "ReplySynthesizer", lambda *a, **k: synth)
    monkeypatch.setattr(listen_glue, "BackchannelCache", lambda *a, **k: PrerenderBackchannel())
    monkeypatch.setattr(listen_glue, "_new_opus_decoder", FakeDecoder)
    monkeypatch.setattr(listen_glue, "enforce_offline_models", lambda: order.append("offline"))
    # Park the endpointer loop after its first poll so tests control the buffers.
    monkeypatch.setattr(listen_glue, "ENDPOINT_POLL_INTERVAL_S", 3600.0)

    var = tmp_path / "var"
    var.mkdir()
    listen = ListenSettings(
        enabled=True,
        allowed_channel_ids=[55],
        vad=ListenVadSettings(backend="energy", min_speech_ms=32, end_silence_ms=64, pre_roll_ms=0),
        reply=ListenReplySettings(per_user_cooldown_s=0.0),
    )
    settings = AppSettings(listen=listen, speech=SpeechSettings(output_dir=var / "speech"))
    provider = FakeProvider()
    app = SimpleNamespace(
        settings=settings,
        coordinator=None,
        pipeline=object(),
        voice_profiles={"goku": _profile()},
        listen_content_provider=provider,
    )
    members = {1: SimpleNamespace(display_name="Alice"), 2: SimpleNamespace(display_name="Bob")}
    guild = SimpleNamespace(id=7, voice_client=None, get_member=members.get)
    vc = FakeVoiceClient()
    channel = FakeChannel(vc, guild)
    bot = SimpleNamespace(user=SimpleNamespace(id=BOT_ID))
    session = ListenSession(app, bot, guild, channel)  # type: ignore[arg-type]
    return Rig(
        session=session,
        vc=vc,
        channel=channel,
        provider=provider,
        synth=synth,
        transcriber=transcriber,
        order=order,
        var=var,
    )


def _extra_tasks() -> list[asyncio.Task[Any]]:
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]


async def test_start_orders_offline_llm_asr_rvc_then_joins_and_announces(rig: Rig) -> None:
    await rig.session.start()
    try:
        # offline guard, then ASR load, then the RVC warm-up.
        assert rig.order == ["offline", "asr", "asr_load", "rvc_warm"]
        assert rig.provider.preferred_calls == 1
        assert rig.vc.listening
        assert rig.channel.sent == [rig.session._app.settings.listen.announcement_text]
        assert rig.synth.calls == [rig.session._app.settings.listen.announcement_text]
        assert len(rig.vc.plays) == 1  # the announcement clip
        assert rig.session.channel_id == 55
    finally:
        await rig.session.stop()


async def test_announcement_is_skipped_when_disabled(rig: Rig) -> None:
    rig.session._app.settings.listen.announce_on_join = False
    await rig.session.start()
    try:
        assert rig.vc.plays == []
        assert rig.channel.sent  # the text notice is still posted
    finally:
        await rig.session.stop()


async def test_start_refuses_when_the_guild_already_has_a_voice_client(rig: Rig) -> None:
    rig.session._guild.voice_client = object()
    with pytest.raises(RuntimeError):
        await rig.session.start()
    assert rig.vc.disconnects == 0  # never joined
    assert rig.session._loop_task is None


async def test_failed_start_cleans_up(rig: Rig) -> None:
    async def boom(text: str) -> None:
        raise OSError("no channel access")

    rig.channel.send = boom  # type: ignore[method-assign]

    # A failed text notice is non-fatal; a failed connect is.
    async def bad_connect(*, cls: Any, timeout: float) -> None:
        raise OSError("connect failed")

    rig.channel.connect = bad_connect  # type: ignore[method-assign]
    with pytest.raises(OSError):
        await rig.session.start()
    assert _extra_tasks() == []


async def test_stop_is_idempotent_and_leaves_no_tasks(rig: Rig) -> None:
    await rig.session.start()
    rig.session.opt_in(1)
    await rig.session.stop()
    await rig.session.stop()
    assert rig.vc.disconnects == 1
    assert not rig.vc.listening
    assert rig.session._consent.opted_in() == frozenset()
    assert rig.session._conversation.history() == []
    assert rig.session.status().endswith("State: STOPPED.")
    assert _extra_tasks() == []


async def test_stop_before_start_is_safe(rig: Rig) -> None:
    await rig.session.stop()
    await rig.session.stop()
    assert rig.vc.disconnects == 0


def _send(rig: Rig, user_id: int, payload: bytes, count: int) -> None:
    sink = rig.vc.sink
    for _ in range(count):
        sink.write(SimpleNamespace(id=user_id), SimpleNamespace(opus=payload))


async def test_opt_out_clears_buffer_and_pending_utterance(rig: Rig) -> None:
    await rig.session.start()
    try:
        s = rig.session
        s.opt_in(1)
        _send(rig, 1, b"T", 5)
        assert s._ingress.pending_samples(1) > 0
        # A half-finished utterance sits in the endpointer.
        s._endpointer.feed(1, np.full(512 * 4, 0.5, dtype=np.float32))
        assert 1 in s._endpointer._users
        await asyncio.sleep(0)  # let the SpeechStarted handler task run
        s.on_member_left(1)
        assert s._ingress.pending_samples(1) == 0
        assert 1 not in s._endpointer._users
        # Packets after leaving are dropped (and joining again does not opt in).
        _send(rig, 1, b"T", 5)
        assert s._ingress.pending_samples(1) == 0
    finally:
        await rig.session.stop()


async def test_unconsented_users_leave_no_trace(rig: Rig) -> None:
    await rig.session.start()
    try:
        rig.session.opt_in(1)
        _send(rig, 2, b"T", 20)
        _send(rig, BOT_ID, b"T", 20)
        assert rig.session._ingress.pending_samples(2) == 0
        assert rig.session._ingress.pending_samples(BOT_ID) == 0
    finally:
        await rig.session.stop()


async def test_voice_state_update_leave_opts_out_and_bot_loss_stops(rig: Rig) -> None:
    await rig.session.start()
    here, elsewhere = SimpleNamespace(id=55), SimpleNamespace(id=56)
    s = rig.session
    s.opt_in(1)
    s.handle_voice_state_update(
        SimpleNamespace(id=2), SimpleNamespace(channel=elsewhere), SimpleNamespace(channel=here)
    )  # someone joining elsewhere: untouched
    assert s._consent.allowed(1)
    s.handle_voice_state_update(
        SimpleNamespace(id=1), SimpleNamespace(channel=here), SimpleNamespace(channel=elsewhere)
    )
    assert not s._consent.allowed(1)
    s.handle_voice_state_update(
        SimpleNamespace(id=BOT_ID), SimpleNamespace(channel=here), SimpleNamespace(channel=None)
    )
    await asyncio.sleep(0.05)
    assert rig.vc.disconnects == 1


async def test_status_lists_names_but_never_transcripts(rig: Rig) -> None:
    await rig.session.start()
    try:
        rig.session.opt_in(1)
        rig.session._conversation.add_user("Alice", "my secret words")
        text = rig.session.status()
        assert "general-voice" in text
        assert "Alice" in text
        assert "IDLE" in text
        assert "secret" not in text
    finally:
        await rig.session.stop()


async def _full_turn(rig: Rig, *, rvc_fails: bool = False) -> None:
    rig.synth.fail = rvc_fails
    await rig.session.start()
    rig.vc.plays.clear()
    rig.session.opt_in(1)
    _send(rig, 1, b"T", 15)  # 300 ms of tone
    _send(rig, 1, b"S", 10)  # then silence
    # The endpointer loop is parked; poll by hand a few times like the loop would.
    for _ in range(40):
        rig.session._poller.poll_once()
        await asyncio.sleep(0.02)
        if rig.vc.plays or rig.synth.calls[1:]:
            break
    await rig.session._turns.wait_idle()


async def test_full_fake_turn_replies_and_writes_nothing(rig: Rig) -> None:
    before = {p for p in rig.var.rglob("*")}
    await _full_turn(rig)
    try:
        assert rig.session._turns is not None
        assert rig.provider.calls  # the Spark got the transcript
        assert rig.vc.plays  # a reply was played through the voice client
        assert {p for p in rig.var.rglob("*")} == before
    finally:
        await rig.session.stop()


async def test_rvc_failure_drops_the_reply_and_the_session_continues(rig: Rig) -> None:
    await _full_turn(rig, rvc_fails=True)
    try:
        assert rig.provider.calls
        assert rig.vc.plays == []  # nothing spoken, no raw-TTS fallback
        assert rig.vc.listening
        # The next turn works once RVC is back.
        rig.synth.fail = False
        _send(rig, 1, b"T", 15)
        _send(rig, 1, b"S", 10)
        for _ in range(40):
            rig.session._poller.poll_once()
            await asyncio.sleep(0.02)
            if rig.vc.plays:
                break
        await rig.session._turns.wait_idle()
        assert rig.vc.plays
    finally:
        await rig.session.stop()


async def test_endpointer_loop_survives_a_poll_error(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(listen_glue, "ENDPOINT_POLL_INTERVAL_S", 0.005)
    await rig.session.start()
    try:
        poller = rig.session._poller
        real = poller.poll_once
        calls = {"n": 0}

        def flaky() -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("vad exploded")
            return real()

        poller.poll_once = flaky  # type: ignore[method-assign]
        await asyncio.sleep(0.1)
        assert calls["n"] > 2
        assert not rig.session._loop_task.done()
    finally:
        await rig.session.stop()


# --- player adapter -------------------------------------------------------------


async def test_player_awaits_completion_and_stop_keeps_listening() -> None:
    vc = FakeVoiceClient()
    vc.listening = True
    vc.hold = True
    player = VoicePlayer(vc)
    task = asyncio.ensure_future(player.play(b"\x00\x00" * 4))
    await asyncio.sleep(0.01)
    assert player.is_playing()
    assert not task.done()
    player.stop()
    await asyncio.wait_for(task, 1.0)
    # stop() must not call VoiceRecvClient.stop(), which also ends reception.
    assert vc.stop_calls == 0
    assert vc.stop_playing_calls == 1
    assert vc.listening


async def test_player_cancel_stops_playback() -> None:
    vc = FakeVoiceClient()
    vc.hold = True
    task = asyncio.ensure_future(VoicePlayer(vc).play(b"\x00\x00" * 4))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert vc.stop_playing_calls == 1


async def test_player_propagates_a_playback_error() -> None:
    vc = FakeVoiceClient()
    vc.hold = True
    task = asyncio.ensure_future(VoicePlayer(vc).play(b"\x00\x00" * 4))
    await asyncio.sleep(0.01)
    vc.finish(RuntimeError("ffmpeg died"))
    with pytest.raises(RuntimeError):
        await task


# --- registry -------------------------------------------------------------------


def test_registry_add_get_remove_active() -> None:
    reg = ListenSessionRegistry()
    session = SimpleNamespace(guild_id=7)
    assert reg.get(7) is None
    assert not reg.active(7)
    reg.add(session)  # type: ignore[arg-type]
    assert reg.get(7) is session
    assert reg.active(7)
    with pytest.raises(RuntimeError):
        reg.add(session)  # type: ignore[arg-type]
    reg.remove(7)
    reg.remove(7)  # removing nothing is fine
    assert not reg.active(7)


async def test_stop_clears_a_populated_conversation(rig: Rig) -> None:
    await _full_turn(rig)
    conversation = rig.session._conversation
    assert conversation is not None
    assert conversation.history(), "the fake turn should have recorded history"
    await rig.session.stop()
    assert conversation.history() == []
