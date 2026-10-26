"""ASR stage timing diagnostic line (no behaviour change, no transcript text)."""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest
from _listen_turn_fakes import make_rig, spin, utterance

from conftest import FakeClock, FakeVad
from notamusicbot.config import ListenVadSettings
from notamusicbot.domain.listen import AudioChunk, Transcript, UtteranceEvent
from notamusicbot.listen.endpointer import Endpointer

LINE = re.compile(
    r"listen asr timing user=(\d+) eos->emit=([\d.]+|n/a) emit->asr_start=([\d.]+|n/a) "
    r"asr_wait=([\d.]+) asr_compute=([\d.]+) asr_done->accepted=([\d.]+) audio_s=([\d.]+)"
)


class SlowTranscriber:
    def __init__(self, clock: FakeClock, compute_s: float, inner_s: float) -> None:
        self.clock, self.compute_s, self.inner_s = clock, compute_s, inner_s

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        self.clock.advance(self.compute_s)
        return Transcript(text="tell me something fun", language="en", elapsed_s=self.inner_s)

    async def shutdown(self) -> None:
        return None


def _lines(caplog: pytest.LogCaptureFixture) -> list[re.Match[str]]:
    return [m for r in caplog.records if (m := LINE.search(r.getMessage()))]


async def test_line_emitted_once_with_deltas(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock(start=100.0)
    rig = make_rig(clock=clock, transcriber=SlowTranscriber(clock, 0.06, 0.05))  # type: ignore[arg-type]
    event = utterance(1, t_start=98.0, t_eos=99.0, audio_s=2.0)
    event = UtteranceEvent(event.user_id, event.audio, event.t_start, event.t_eos, t_emit=99.5)
    with caplog.at_level(logging.INFO, logger="notamusicbot.listen.turn_taking"):
        await rig.manager.on_utterance(event)
        await spin()
    found = _lines(caplog)
    assert len(found) == 1
    user, eos_emit, emit_start, wait, compute, accepted, audio_s = found[0].groups()
    assert user == "1"
    assert float(eos_emit) == pytest.approx(0.5)
    assert float(emit_start) == pytest.approx(0.5)  # clock was 100.0 at asr start
    assert float(compute) == pytest.approx(0.06)
    assert float(wait) == pytest.approx(0.01)
    assert float(accepted) == pytest.approx(0.0)
    assert float(audio_s) == pytest.approx(2.0)
    assert "tell me something fun" not in caplog.text.split("listen asr timing")[1].split("\n")[0]


async def test_not_emitted_for_non_consenting_user(caplog: pytest.LogCaptureFixture) -> None:
    rig = make_rig(users=(1,))
    with caplog.at_level(logging.INFO, logger="notamusicbot.listen.turn_taking"):
        await rig.manager.on_utterance(utterance(2))
        await spin()
    assert _lines(caplog) == []
    assert rig.transcriber.calls == []


async def test_legacy_event_without_t_emit_logs_na(caplog: pytest.LogCaptureFixture) -> None:
    rig = make_rig()
    with caplog.at_level(logging.INFO, logger="notamusicbot.listen.turn_taking"):
        await rig.manager.on_utterance(utterance(1))
        await spin()
    found = _lines(caplog)
    assert len(found) == 1
    assert found[0].group(2) == "n/a"


def test_endpointer_stamps_t_emit_with_its_clock() -> None:
    clock = FakeClock(start=10.0)
    events: list[object] = []
    vad = FakeVad(probabilities=[1.0] * 20, default=0.0)
    settings = ListenVadSettings(min_speech_ms=64, end_silence_ms=64, pre_roll_ms=0)
    ep = Endpointer(vad, settings, clock, events.append)  # type: ignore[arg-type]
    ep.feed(1, np.zeros(512 * 24, dtype=np.float32))
    utt = [e for e in events if isinstance(e, UtteranceEvent)]
    assert len(utt) == 1
    assert utt[0].t_emit == 10.0
    assert utt[0].t_emit >= utt[0].t_eos
