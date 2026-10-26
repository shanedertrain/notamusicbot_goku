"""Review additions for the listen speed build: streaming consent, stream limiter."""

from __future__ import annotations

import numpy as np
from _listen_turn_fakes import StreamPlayer, StreamSynth, TimelineClock, make_rig, spin, utterance

from conftest import FakeContentProvider
from notamusicbot.listen.reply import STREAM_GAIN_DB, _soft_limit


async def test_consent_revoked_before_turn_never_streams_synthesis() -> None:
    clock = TimelineClock(start=100.0)
    log: list[tuple[str, float, str]] = []
    synth = StreamSynth(clock, log)
    player = StreamPlayer(clock, log)
    rig = make_rig(
        clock=clock,
        synth=synth,  # type: ignore[arg-type]
        player=player,  # type: ignore[arg-type]
        provider=FakeContentProvider(["Hello there friend."]),
        users=(1,),
    )
    # The turn task is created but has not run when consent is withdrawn.
    await rig.manager.on_utterance(utterance(1))
    rig.consent.opt_out(1)
    await spin()
    await clock.run()
    await rig.manager.wait_idle()
    assert synth.calls == []
    assert rig.provider.calls == []


def test_soft_limit_is_linear_below_the_knee_and_stays_within_full_scale() -> None:
    low = np.array([-0.5, -0.1, 0.0, 0.3, 0.79], dtype=np.float32)
    assert np.allclose(_soft_limit(low), low)
    loud = np.array([-4.0, -1.0, 1.0, 1.5, 4.0], dtype=np.float32)
    out = _soft_limit(loud)
    assert np.all(np.abs(out) <= 1.0)
    assert np.all(np.sign(out) == np.sign(loud))
    assert np.all(np.diff(_soft_limit(np.linspace(0.0, 3.0, 300, dtype=np.float32))) >= 0)


def test_stream_gain_matches_the_live_measurement() -> None:
    # 8 real fast-TTS sentences vs the postprocessed WAV path (2026-10-25):
    # median RMS difference -3.7 dB at +3 dB gain, -0.6 dB at +5 dB.
    assert 3.5 <= STREAM_GAIN_DB <= 5.0
