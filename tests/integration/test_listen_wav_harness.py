"""P10: a WAV through the REAL listen chain with fakes at the model boundaries.

    FakePacketSource -> ListenIngress (pass-through "decoder") -> Endpointer
    (EnergyVad) -> TurnManager (scripted transcriber / content provider,
    ReplySynthesizer over the FakeSynthesizer/FakeConverter pipeline, recording
    player).

No GPU, no models, no network, no Discord, nothing written outside ``tmp_path``.

Clocks: the whole chain runs on ``time.monotonic`` (``MonotonicClock``), NOT on
the unit tests' ``FakeClock``. A FakeClock only moves when told to, so every
queue hop and conversion between the endpointer and the synthesizer would read as
exactly 0 ms; the point of the plumbing check is the real CPU and event-loop cost
of the glue. The fakes' own configured delays (ASR and LLM ``asyncio.sleep``)
are measured out and subtracted.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np

from notamusicbot.config import ListenSettings
from notamusicbot.domain.listen import (
    ConsentState,
    SpeechStarted,
    UtteranceEvent,
)
from notamusicbot.domain.voice_profiles import VoiceProfile
from notamusicbot.listen.conversation import Conversation
from notamusicbot.listen.endpointer import Endpointer, EndpointerThread, ListenEvent
from notamusicbot.listen.ingress import ListenIngress
from notamusicbot.listen.metrics import parse_turn_line
from notamusicbot.listen.reply import ReplyAudio, ReplySynthesizer
from notamusicbot.listen.turn_taking import MonotonicClock, TurnManager
from notamusicbot.speech.pipeline import SpeechPipeline
from notamusicbot.speech.vad.energy import EnergyVad

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_unit_fakes():
    """Import tests/unit/conftest.py by path under a private name.

    ``from conftest import ...`` would resolve to whichever ``conftest`` module
    pytest imported first (there are several), so load the file explicitly.
    """
    path = REPO_ROOT / "tests" / "unit" / "conftest.py"
    spec = importlib.util.spec_from_file_location("_listen_unit_fakes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_fakes = _load_unit_fakes()
FakeConverter = _fakes.FakeConverter
FakeSynthesizer = _fakes.FakeSynthesizer
FakeContentProvider = _fakes.FakeContentProvider
FakeTranscriber = _fakes.FakeTranscriber
FakePacketSource = _fakes.FakePacketSource

BOT_ID = 999
OPTED_IN = 1
STRANGER = 2
SAMPLE_RATE_48K = 48000
PACKET_FRAMES = 960  # 20 ms at 48 kHz
ASR_DELAY_S = 0.03
LLM_DELAY_S = 0.03
PLUMBING_BUDGET_S = 0.050
QUESTION = "hey goku what's your favorite food"
REPLY = "I love a big bowl of rice!"


def speech_like_48k_stereo_packets() -> list[bytes]:
    """0.3 s silence, 1.0 s voiced (120 Hz harmonics, ~-12 dBFS), 1.2 s silence.

    The trailing 1.2 s exceeds the 600 ms end-of-speech silence, so the
    endpointer closes the utterance from the stream alone.
    """
    t = np.arange(int(1.0 * SAMPLE_RATE_48K)) / SAMPLE_RATE_48K
    voiced = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 8))
    voiced = 0.25 * voiced / np.max(np.abs(voiced))
    mono = np.concatenate(
        [
            np.zeros(int(0.3 * SAMPLE_RATE_48K)),
            voiced,
            np.zeros(int(1.2 * SAMPLE_RATE_48K)),
        ]
    )
    pcm = (np.clip(mono, -1, 1) * 32767).astype("<i2")
    stereo = np.repeat(pcm[:, None], 2, axis=1)
    packets = [
        stereo[i : i + PACKET_FRAMES].tobytes() for i in range(0, len(stereo), PACKET_FRAMES)
    ]
    assert all(len(p) == PACKET_FRAMES * 4 for p in packets[:-1])
    return packets


class PassThroughDecoder:
    """The packet payload already is 48 kHz stereo s16 PCM: return it unchanged."""

    def __init__(self, log: list[bytes]) -> None:
        self._log = log

    def decode(self, data: bytes) -> bytes:
        self._log.append(data)
        return data


class DelayedTranscriber(FakeTranscriber):
    """Scripted text after ``delay_s`` of real asyncio sleep (a fake model delay)."""

    def __init__(self, texts: list[str], delay_s: float) -> None:
        super().__init__(texts)
        self.delay_s = delay_s

    async def transcribe(self, audio):  # type: ignore[no-untyped-def]
        import asyncio

        await asyncio.sleep(self.delay_s)
        return await super().transcribe(audio)


class TimedSynthesizer(ReplySynthesizer):
    """Real ReplySynthesizer that records when each sentence synthesis begins."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.started_at: list[float] = []

    async def synthesize_sentence(self, text: str) -> ReplyAudio:
        self.started_at.append(time.monotonic())
        return await super().synthesize_sentence(text)


class RecordingPlayer:
    """ReplyPlayerPort fake: records PCM and when each play began; returns at once."""

    def __init__(self) -> None:
        self.plays: list[bytes] = []
        self.started_at: list[float] = []

    async def play(self, pcm: bytes) -> None:
        self.started_at.append(time.monotonic())
        self.plays.append(pcm)

    def stop(self) -> None:
        pass

    def is_playing(self) -> bool:
        return False


def _profile() -> VoiceProfile:
    return VoiceProfile.model_validate(
        {
            "id": "goku",
            "persona": {"display_name": "Goku"},
            "tts": {"speaker": "Ryan"},
            "voice_conversion": {"backend": "passthrough"},
            "audio": {},
        }
    )


class Harness:
    """The real chain with fakes at the edges; events are collected, not threaded."""

    def __init__(self, tmp_path: Path, *, opted_in: tuple[int, ...] = (OPTED_IN,)) -> None:
        self.settings = ListenSettings()
        self.consent = ConsentState(bot_user_id=BOT_ID)
        for user in opted_in:
            self.consent.opt_in(user)
        self.decoded: list[bytes] = []
        self.ingress = ListenIngress(
            self.consent,
            BOT_ID,
            lambda: PassThroughDecoder(self.decoded),
            self.settings.vad.ring_buffer_s,
        )
        # Real monotonic clock for the endpointer and the manager (see module docstring).
        self.clock = MonotonicClock()
        self.events: list[ListenEvent] = []
        self.emit_times: list[float] = []

        def emit(event: ListenEvent) -> None:
            self.emit_times.append(time.monotonic())
            self.events.append(event)

        self.endpointer = Endpointer(EnergyVad(), self.settings.vad, self.clock, emit)
        self.poller = EndpointerThread(self.ingress, self.endpointer, self.consent.opted_in)

        out = tmp_path / "speech"
        pipeline = SpeechPipeline(FakeSynthesizer(seconds=0.4), FakeConverter(), output_dir=out)
        self.synth = TimedSynthesizer(pipeline, _profile(), artifact_root=out)
        self.transcriber = DelayedTranscriber([QUESTION], ASR_DELAY_S)
        self.provider = FakeContentProvider([REPLY], delay_s=LLM_DELAY_S)
        self.player = RecordingPlayer()
        self.metrics: list[str] = []
        self.manager = TurnManager(
            self.settings,
            self.clock,
            self.consent,
            self.transcriber,
            Conversation(self.settings.reply.history_turns),
            self.provider,
            self.synth,
            None,  # no backchannel: this test times the reply path
            self.player,
            self.metrics.append,
            BOT_ID,
            guild_id=7,
        )

    def feed_wav(self, user_id: int, packets: list[bytes]) -> int:
        """FakePacketSource -> ingress, then one endpointer poll (synchronous)."""
        source = FakePacketSource([(user_id, p) for p in packets])
        delivered = source.deliver(self.ingress.accept_packet)
        self.poller.poll_once()
        return delivered

    async def dispatch(self) -> None:
        """Hand the collected endpointer events to the manager, in order."""
        for event in list(self.events):
            if isinstance(event, SpeechStarted):
                await self.manager.on_speech_started(event)
            elif isinstance(event, UtteranceEvent):
                await self.manager.on_utterance(event)
        await self.manager.wait_idle()


async def test_opted_in_user_gets_exactly_one_reply_with_low_plumbing_overhead(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    packets = speech_like_48k_stereo_packets()

    assert h.feed_wav(OPTED_IN, packets) == len(packets)
    utterances = [e for e in h.events if isinstance(e, UtteranceEvent)]
    assert len([e for e in h.events if isinstance(e, SpeechStarted)]) == 1
    assert len(utterances) == 1
    # About one second of voice plus 300 ms pre-roll, 16 kHz mono float32.
    assert 0.9 < utterances[0].audio.duration_s < 1.6
    assert utterances[0].audio.sample_rate_hz == 16000
    assert len(h.decoded) == len(packets)  # opted in: every packet was decoded

    await h.dispatch()

    # Exactly one reply PCM (the scripted reply is one sentence), produced by the
    # real ReplySynthesizer as 48 kHz stereo s16.
    assert len(h.transcriber.calls) == 1
    assert len(h.provider.calls) == 1
    assert len(h.player.plays) == 1  # no backchannel configured
    assert len(h.synth.started_at) == 1
    assert len(h.player.plays[0]) == int(0.4 * SAMPLE_RATE_48K) * 4  # 0.4 s, stereo, 2 bytes
    assert len(h.metrics) == 1

    # Metrics line parses and carries the manager's own stage fields.
    parsed = parse_turn_line(h.metrics[0])
    assert parsed is not None
    assert parsed["guild"] == "7"
    assert parsed["user"] == str(OPTED_IN)
    for key in ("eos->asr", "asr->llm", "llm->tts1", "tts1->audio"):
        assert key in parsed
    assert parsed["total"] is not None and parsed["total"] > 0.0

    # Plumbing overhead: endpointer emit -> first sentence starts synthesizing,
    # minus the fakes' own ASR and LLM delays.
    elapsed = h.synth.started_at[0] - h.emit_times[-1]
    overhead = elapsed - (ASR_DELAY_S + LLM_DELAY_S)
    assert overhead < PLUMBING_BUDGET_S, f"plumbing overhead {overhead * 1000:.1f} ms"
    # The delays really happened (a negative overhead would mean the subtraction lied).
    assert elapsed >= ASR_DELAY_S + LLM_DELAY_S
    print(f"plumbing overhead {overhead * 1000:.1f} ms (elapsed {elapsed * 1000:.1f} ms)")

    # No audio files survive: the synthesizer cleans its job artifacts.
    assert list((tmp_path / "speech").rglob("*.wav")) == []


async def test_user_who_is_not_opted_in_is_never_decoded_or_transcribed(tmp_path: Path) -> None:
    h = Harness(tmp_path, opted_in=(OPTED_IN,))
    packets = speech_like_48k_stereo_packets()

    h.feed_wav(STRANGER, packets)
    await h.dispatch()

    assert h.decoded == []  # dropped before decode
    assert h.ingress.pending_samples(STRANGER) == 0
    assert h.events == []  # no VAD, no utterance
    assert len(h.transcriber.calls) == 0  # ZERO transcribe calls
    assert h.provider.calls == []
    assert h.player.plays == []
    assert h.metrics == []


async def test_stranger_utterance_handed_straight_to_manager_is_not_transcribed(
    tmp_path: Path,
) -> None:
    """Belt and braces: even an event that reaches the manager is checked before ASR."""
    h = Harness(tmp_path, opted_in=(OPTED_IN,))
    h.feed_wav(OPTED_IN, speech_like_48k_stereo_packets())
    event = next(e for e in h.events if isinstance(e, UtteranceEvent))
    forged = UtteranceEvent(
        user_id=STRANGER, audio=event.audio, t_start=event.t_start, t_eos=event.t_eos
    )

    await h.manager.on_utterance(forged)
    await h.manager.wait_idle()

    assert len(h.transcriber.calls) == 0
    assert h.player.plays == []
