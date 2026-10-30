"""TurnManager behavior with fakes and FakeClock (plan sections 5 and 7)."""

from __future__ import annotations

import asyncio
import logging
import re

import pytest
from _listen_turn_fakes import (
    BACKCHANNEL_PCM,
    BOT_ID,
    FakeBackchannel,
    FakeContentProvider,
    FakePlayer,
    FakeReplySynth,
    FakeTranscriber,
    StreamingProvider,
    TimedPlayer,
    TimedSynth,
    TimelineClock,
    UserTranscriber,
    make_rig,
    speech,
    spin,
    utterance,
)

from notamusicbot.config import ListenPrivacySettings
from notamusicbot.listen.metrics import parse_turn_line
from notamusicbot.listen.reply import ReplySynthesisError
from notamusicbot.listen.turn_taking import TurnState

REPLY = b"Hey, that sounds awesome!"


async def test_one_utterance_backchannel_before_reply() -> None:
    rig = make_rig(backchannel=FakeBackchannel())
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert len(rig.transcriber.calls) == 1
    assert len(rig.provider.calls) == 1
    assert rig.player.plays == [BACKCHANNEL_PCM, REPLY]
    assert rig.manager.state is TurnState.IDLE
    rendered = [t.render() for t in rig.conversation.history()]
    assert rendered == ["User 1: tell me something fun", "Goku: Hey, that sounds awesome!"]
    request = rig.provider.calls[0]
    assert request.requester_name == "User 1"
    assert request.transcript_text == "tell me something fun"


async def test_metrics_line_emitted() -> None:
    rig = make_rig()
    await rig.manager.on_utterance(utterance(1, t_start=99.0, t_eos=99.5))
    await rig.manager.wait_idle()
    assert len(rig.metrics) == 1
    parsed = parse_turn_line(rig.metrics[0])
    assert parsed is not None
    assert parsed["guild"] == "7"
    assert parsed["user"] == "1"
    # FakeClock never advances: eos 99.5 -> now 100.0 is the only non-zero stage.
    assert parsed["eos->asr"] == 0.5
    assert parsed["total"] == 0.5
    assert re.fullmatch(r"listen turn guild=7 user=1 .*", rig.metrics[0])


async def test_next_sentence_synthesized_while_first_plays() -> None:
    player = FakePlayer(hold=True)
    provider = FakeContentProvider(["First one here. Second one there. Third one gone."])
    rig = make_rig(player=player, provider=provider)
    await rig.manager.on_utterance(utterance(1))
    await spin()
    # Sentence 1 is playing (held); sentence 2 is already synthesized; 3 is trimmed away.
    assert player.plays == [b"First one here."]
    assert rig.synth.calls == ["First one here.", "Second one there."]
    player.release()
    await spin()
    assert player.plays == [b"First one here.", b"Second one there."]
    player.release()
    await rig.manager.wait_idle()
    assert rig.conversation.history()[-1].text == "First one here. Second one there."


async def test_barge_in_half_second_into_reply() -> None:
    player = FakePlayer(hold=True)
    provider = FakeContentProvider(["First one here. Second one there."])
    rig = make_rig(player=player, provider=provider)
    await rig.manager.on_utterance(utterance(1))
    await spin()
    assert rig.manager.state is TurnState.SPEAKING
    task = rig.manager.reply_task
    assert task is not None
    rig.clock.advance(0.5)
    await rig.manager.on_speech_started(speech(2, t_start=rig.clock() - 0.5))
    await spin()
    assert player.stop_calls == 1
    assert task.cancelled()
    assert rig.manager.state is TurnState.LISTENING
    assert rig.manager.active_user_id == 2
    texts = [t.render() for t in rig.conversation.history()]
    assert texts[-1] == "Goku: First one here. Second one there. [interrupted after sentence 1]"
    assert player.plays == [b"First one here."]  # sentence 2 never started


async def test_barge_in_below_400ms_does_not_stop() -> None:
    player = FakePlayer(hold=True)
    rig = make_rig(player=player)
    await rig.manager.on_utterance(utterance(1))
    await spin()
    rig.clock.advance(0.2)
    await rig.manager.on_speech_started(speech(2, t_start=rig.clock() - 0.2))
    assert player.stop_calls == 0
    # The speech ends after 0.2 s (a one-word noise utterance) before 400 ms.
    rig.manager._transcriber = FakeTranscriber(["uh"])  # type: ignore[attr-defined]
    await rig.manager.on_utterance(utterance(2, t_start=rig.clock() - 0.2, t_eos=rig.clock()))
    await spin()
    assert player.stop_calls == 0
    assert rig.manager.state is TurnState.SPEAKING
    player.release()
    await rig.manager.wait_idle()


async def test_sustained_speech_barges_in_once_400ms_pass() -> None:
    player = FakePlayer(hold=True)
    rig = make_rig(player=player)
    await rig.manager.on_utterance(utterance(1))
    await spin()
    rig.clock.advance(0.2)
    await rig.manager.on_speech_started(speech(2, t_start=rig.clock() - 0.2))
    assert player.stop_calls == 0
    await spin()  # the timer sleeps the remaining 0.2 s on the FakeClock
    assert player.stop_calls == 1
    assert rig.clock.sleeps == [pytest.approx(0.2)]


async def test_no_barge_in_in_first_300ms_of_backchannel() -> None:
    player = FakePlayer(hold=True)
    rig = make_rig(player=player, backchannel=FakeBackchannel())
    await rig.manager.on_utterance(utterance(1))
    await spin()
    assert player.plays == [BACKCHANNEL_PCM]
    rig.clock.advance(0.1)
    # Speech began 0.4 s ago, but the clip is only 0.1 s old: the guard delays it.
    await rig.manager.on_speech_started(speech(2, t_start=rig.clock() - 0.4))
    assert player.stop_calls == 0
    await spin()
    assert player.stop_calls == 1
    # 0.3 s guard from the clip start + 0.4 s sustained - 0.1 s elapsed.
    assert rig.clock.sleeps == [pytest.approx(0.6)]


async def test_bot_speech_never_barges_in() -> None:
    player = FakePlayer(hold=True)
    rig = make_rig(player=player)
    await rig.manager.on_utterance(utterance(1))
    await spin()
    rig.clock.advance(5.0)
    await rig.manager.on_speech_started(speech(BOT_ID, t_start=rig.clock() - 5.0))
    await spin()
    assert player.stop_calls == 0
    assert rig.manager.state is TurnState.SPEAKING
    player.release()
    await rig.manager.wait_idle()


async def test_two_utterances_while_thinking_merge_into_one_generate() -> None:
    tr = UserTranscriber({1: "hello there goku", 2: "what about me"})
    tr.gates = {1: asyncio.Event(), 2: asyncio.Event()}
    rig = make_rig(transcriber=tr, names={1: "Alice", 2: "Bob"})
    a = asyncio.create_task(rig.manager.on_utterance(utterance(1)))
    b = asyncio.create_task(rig.manager.on_utterance(utterance(2)))
    await spin()
    assert rig.manager.state is TurnState.THINKING
    tr.gates[1].set()
    await a  # Alice's turn is queued but Bob's ASR is still in flight
    await spin()
    assert rig.provider.calls == []
    tr.gates[2].set()
    await b
    await rig.manager.wait_idle()
    assert len(rig.provider.calls) == 1
    request = rig.provider.calls[0]
    assert request.transcript_text == "Alice: hello there goku\nBob: what about me"
    assert request.requester_name == "Alice and Bob"
    assert rig.player.plays == [REPLY]


async def test_same_user_replaces_pending_slot() -> None:
    tr = UserTranscriber({1: "first thing said"})
    tr.gates = {1: asyncio.Event()}
    rig = make_rig(transcriber=tr)
    a = asyncio.create_task(rig.manager.on_utterance(utterance(1)))
    await spin()
    tr.texts[1] = "second thing said"
    b = asyncio.create_task(rig.manager.on_utterance(utterance(1)))
    await spin()
    tr.gates[1].set()
    await asyncio.gather(a, b)
    await rig.manager.wait_idle()
    # Both ASR calls returned the (mutated) shared text; one slot, one generate.
    assert len(rig.provider.calls) == 1
    assert rig.provider.calls[0].transcript_text == "second thing said"


async def test_thirteen_utterances_in_a_minute_give_twelve_replies() -> None:
    users = (1, 2, 3)
    texts = [f"tell me story number {i}" for i in range(13)]
    rig = make_rig(transcriber=FakeTranscriber(texts))
    for i in range(13):
        await rig.manager.on_utterance(utterance(users[i % 3]))
        await rig.manager.wait_idle()
        rig.clock.advance(4.0)  # 13 utterances span 48 s of simulated time
    assert len(rig.transcriber.calls) == 13
    assert len(rig.provider.calls) == 12
    assert rig.player.plays == [REPLY] * 12
    # Once the oldest reply leaves the 60 s window a new one is allowed again.
    rig.clock.advance(60.0)
    rig.transcriber.texts.append("one more story please")
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert len(rig.player.plays) == 13


async def test_per_user_cooldown() -> None:
    rig = make_rig(reply={"per_user_cooldown_s": 2.0})
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    rig.clock.advance(1.0)
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert len(rig.provider.calls) == 1
    rig.clock.advance(1.0)
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert len(rig.provider.calls) == 2


async def test_llm_timeout_plays_nothing_and_returns_to_idle() -> None:
    provider = FakeContentProvider(["too late"], delay_s=1.0)
    rig = make_rig(provider=provider, reply={"llm_timeout_s": 0.05})
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert len(provider.calls) == 1
    assert rig.player.plays == []
    assert rig.manager.state is TurnState.IDLE
    assert [t.is_bot for t in rig.conversation.history()] == [False]


async def test_llm_error_plays_nothing() -> None:
    rig = make_rig(provider=FakeContentProvider([RuntimeError("down")]))
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert rig.player.plays == []
    assert rig.manager.state is TurnState.IDLE


async def test_synthesis_error_drops_reply(caplog: pytest.LogCaptureFixture) -> None:
    rig = make_rig(synth=FakeReplySynth(fail=True))
    with caplog.at_level(logging.DEBUG, logger="notamusicbot.listen.turn_taking"):
        await rig.manager.on_utterance(utterance(1))
        await rig.manager.wait_idle()
    assert rig.player.plays == []
    assert rig.manager.state is TurnState.IDLE
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert [t.is_bot for t in rig.conversation.history()] == [False]
    assert rig.metrics == []


async def test_every_human_is_transcribed_and_answered() -> None:
    rig = make_rig()
    await rig.manager.on_utterance(utterance(77))
    await rig.manager.wait_idle()
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert len(rig.transcriber.calls) == 2  # no per-user permission gate
    assert len(rig.provider.calls) == 2
    assert rig.manager.state is TurnState.IDLE


async def test_echo_transcript_is_dropped() -> None:
    tr = FakeTranscriber(["tell me something fun", "Hey, that sounds awesome!"])
    rig = make_rig(transcriber=tr)
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    rig.clock.advance(5.0)
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert len(tr.calls) == 2
    assert len(rig.provider.calls) == 1


@pytest.mark.parametrize("noise", ["Thank you.", "ok", "."])
async def test_noise_transcript_is_dropped(noise: str) -> None:
    rig = make_rig(transcriber=FakeTranscriber([noise]))
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert len(rig.transcriber.calls) == 1
    assert rig.provider.calls == []
    assert rig.manager.state is TurnState.IDLE


async def test_wake_word_and_follow_up_window() -> None:
    tr = FakeTranscriber(
        [
            "hello there friend",
            "Hey GOKU what is up",
            "and another thing then",
            "tell me about dragons",
        ]
    )
    rig = make_rig(respond_to="wake_word", transcriber=tr)
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert rig.provider.calls == []  # no wake word, bot has not spoken
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert len(rig.provider.calls) == 1  # case-insensitive wake word
    rig.clock.advance(5.0)  # inside the 8 s follow-up window
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert len(rig.provider.calls) == 2
    rig.clock.advance(20.0)  # window over
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert len(rig.provider.calls) == 2


async def test_backchannel_failure_is_ignored() -> None:
    player = FakePlayer()
    player.fail_on = {BACKCHANNEL_PCM}
    rig = make_rig(player=player, backchannel=FakeBackchannel())
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert player.plays == [BACKCHANNEL_PCM, REPLY]


async def test_backchannel_skipped_when_disabled_or_unavailable() -> None:
    off = make_rig(backchannel=FakeBackchannel(), reply={"backchannel_enabled": False})
    await off.manager.on_utterance(utterance(1))
    await off.manager.wait_idle()
    assert off.player.plays == [REPLY]
    empty = make_rig(backchannel=FakeBackchannel(empty=True))
    await empty.manager.on_utterance(utterance(1))
    await empty.manager.wait_idle()
    assert empty.player.plays == [REPLY]


async def test_utterance_during_reply_becomes_next_turn() -> None:
    player = FakePlayer(hold=True)
    tr = FakeTranscriber(["tell me something fun", "now tell me more"])
    rig = make_rig(player=player, transcriber=tr)
    await rig.manager.on_utterance(utterance(1))
    await spin()
    await rig.manager.on_utterance(utterance(1))  # queued, no barge-in
    assert len(rig.provider.calls) == 1
    player.release()
    await spin()
    player.release()
    await rig.manager.wait_idle()
    assert len(rig.provider.calls) == 2
    assert rig.provider.calls[1].transcript_text == "now tell me more"


async def test_shutdown_stops_everything_and_clears_history() -> None:
    player = FakePlayer(hold=True)
    rig = make_rig(player=player)
    await rig.manager.on_utterance(utterance(1))
    await spin()
    task = rig.manager.reply_task
    assert task is not None
    await rig.manager.shutdown()
    assert player.stop_calls == 1
    assert task.done()
    assert rig.conversation.history() == []
    assert rig.manager.state is TurnState.IDLE
    await rig.manager.on_utterance(utterance(1))  # ignored after shutdown
    assert len(rig.transcriber.calls) == 1


# ---------------------------------------------------- round-3 orchestrator tripwires


async def test_transcript_text_is_logged_at_debug_only_when_enabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig = make_rig(transcriber=FakeTranscriber(["secret words here"]))
    m = rig.manager
    with caplog.at_level(logging.DEBUG, logger="notamusicbot.listen.turn_taking"):
        await rig.manager.on_utterance(utterance(1))
        await rig.manager.wait_idle()
    assert not [r for r in caplog.records if "secret words here" in r.getMessage()]

    m._settings = m._settings.model_copy(  # type: ignore[attr-defined]
        update={"privacy": ListenPrivacySettings(log_transcripts=True)}
    )
    with caplog.at_level(logging.DEBUG, logger="notamusicbot.listen.turn_taking"):
        await rig.manager.on_utterance(utterance(1))
        await rig.manager.wait_idle()
    assert [r for r in caplog.records if "secret words here" in r.getMessage()]


async def test_bot_id_is_never_transcribed_or_answered() -> None:
    rig = make_rig()
    await rig.manager.on_utterance(utterance(BOT_ID))
    await rig.manager.wait_idle()
    assert rig.transcriber.calls == []
    assert rig.provider.calls == []
    assert rig.player.plays == []
    assert rig.manager.state is TurnState.IDLE


async def test_synthesis_error_on_second_sentence_keeps_first_only() -> None:
    class SecondFails(FakeReplySynth):
        async def synthesize_sentence(self, text: str):  # type: ignore[no-untyped-def]
            if self.calls:
                self.calls.append(text)
                raise ReplySynthesisError("rvc gone")
            return await super().synthesize_sentence(text)

    synth = SecondFails()
    provider = FakeContentProvider(["First one here. Second one there."])
    rig = make_rig(synth=synth, provider=provider)
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert synth.calls == ["First one here.", "Second one there."]
    assert rig.player.plays == [b"First one here."]
    assert rig.manager.state is TurnState.IDLE
    assert len(rig.metrics) == 1


THREE_CHUNKS = (
    "I really think that the best food in the whole world is ramen — "
    "especially the spicy kind from the little shop downtown."
)


def _timed_rig(
    *, fail_on_call: int | None = None, text: str = THREE_CHUNKS, max_sentences: int = 3
):  # type: ignore[no-untyped-def]
    clock = TimelineClock(start=100.0)
    log: list[tuple[str, float, str]] = []
    synth = TimedSynth(clock, log, synth_s=1.0, play_s=3.0, fail_on_call=fail_on_call)
    player = TimedPlayer(clock, log, play_s=3.0)
    rig = make_rig(
        clock=clock,
        synth=synth,  # type: ignore[arg-type]
        player=player,  # type: ignore[arg-type]
        provider=FakeContentProvider([text]),
        # Pin the pre-streaming caps (10 / 18 words) so the chunk text below is unchanged.
        reply={
            "max_sentences": max_sentences,
            "first_chunk_max_words": 10,
            "later_chunk_max_words": 18,
        },
    )
    return rig, clock, log, synth, player


def _starts(log: list[tuple[str, float, str]], kind: str, t0: float) -> list[float]:
    return [round(t - t0, 6) for k, t, _ in log if k == kind]


async def test_pipelined_playback_has_no_gap_between_chunks() -> None:
    # Three sentences, one chunk each (all short).
    rig, clock, log, synth, _player = _timed_rig(
        text="First one here. Second one there. Third one gone."
    )
    t0 = clock()
    await rig.manager.on_utterance(utterance(1))
    await clock.run()
    await rig.manager.wait_idle()
    assert synth.calls == ["First one here.", "Second one there.", "Third one gone."]
    assert _starts(log, "play_start", t0) == [1.0, 4.0, 7.0]
    assert _starts(log, "play_end", t0) == [4.0, 7.0, 10.0]
    assert _starts(log, "synth_start", t0) == [0.0, 1.0, 2.0]
    # Chunk 2 starts the instant chunk 1 ends (was +1.0 s before pipelining).
    assert abs(_starts(log, "play_start", t0)[1] - _starts(log, "play_end", t0)[0]) < 0.01
    # Call order: synth2 starts before play1 ends.
    names = [(k, x) for k, _, x in log]
    assert names.index(("synth_start", "Second one there.")) < names.index(
        ("play_end", "First one here.")
    )
    assert rig.manager.state is TurnState.IDLE


async def test_long_sentence_is_spoken_as_chunks_with_shared_sentence_index() -> None:
    rig, clock, log, synth, _player = _timed_rig(text=THREE_CHUNKS)
    t0 = clock()
    await rig.manager.on_utterance(utterance(1))
    await clock.run()
    await rig.manager.wait_idle()
    assert synth.calls == [
        "I really think that the best food in the whole",
        "world is ramen — especially the spicy kind from the little shop downtown.",
    ]
    assert _starts(log, "play_start", t0) == [1.0, 4.0]


async def test_barge_in_during_chunk_one_cancels_synthesis_and_marks_sentence_one() -> None:
    rig, clock, log, synth, player = _timed_rig(
        text="First one here. Second one there. Third one gone."
    )
    await rig.manager.on_utterance(utterance(1))
    await clock.run(until=clock() + 1.5)  # chunk 1 plays from 1.0
    assert rig.manager.state is TurnState.SPEAKING
    await rig.manager.on_speech_started(speech(2, t_start=clock() - 0.5))
    await clock.run(until=clock() + 0.1)
    assert player.stop_calls == 1
    calls_at_cancel = list(synth.calls)
    await clock.run()
    assert synth.calls == calls_at_cancel
    assert player.plays == [b"First one here."]
    assert rig.manager.state is TurnState.LISTENING
    assert rig.conversation.history()[-1].render() == (
        "Goku: First one here. Second one there. Third one gone. [interrupted after sentence 1]"
    )


async def test_barge_in_in_second_chunk_of_sentence_marks_that_sentence() -> None:
    rig, clock, log, synth, player = _timed_rig(text=THREE_CHUNKS)
    t0 = clock()
    await rig.manager.on_utterance(utterance(1))
    await clock.run(until=t0 + 4.5)  # chunk 2 (still sentence 1) plays from 4.0
    assert player.plays[-1].startswith(b"world is ramen")
    await rig.manager.on_speech_started(speech(2, t_start=clock() - 0.5))
    await clock.run(until=clock() + 0.1)
    assert player.stop_calls == 1
    assert rig.conversation.history()[-1].render().endswith("[interrupted after sentence 1]")


async def test_chunk_two_synthesis_error_plays_only_chunk_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig, clock, log, synth, player = _timed_rig(
        fail_on_call=2, text="First one here. Second one there. Third one gone."
    )
    with caplog.at_level(logging.DEBUG, logger="notamusicbot.listen.turn_taking"):
        await rig.manager.on_utterance(utterance(1))
        await clock.run()
        await rig.manager.wait_idle()
    assert player.plays == [b"First one here."]
    assert rig.manager.state is TurnState.IDLE
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert synth.calls == ["First one here.", "Second one there."]  # nothing after the failure


# ------------------------------------------------------------ streaming replies

THREE_SENTENCE_DELTAS = [
    "First ", "one ", "here. ", "Second ", "one ", "there. ", "Third ", "one ", "gone.",
]  # fmt: skip


def _streaming_rig(
    deltas: list[str],
    *,
    gap_s: float = 0.5,
    fail_after: int | None = None,
    max_sentences: int = 3,
    synth_s: float = 1.0,
    play_s: float = 3.0,
):  # type: ignore[no-untyped-def]
    clock = TimelineClock(start=100.0)
    log: list[tuple[str, float, str]] = []
    provider = StreamingProvider(clock, log, deltas, gap_s=gap_s, fail_after=fail_after)
    synth = TimedSynth(clock, log, synth_s=synth_s, play_s=play_s)
    player = TimedPlayer(clock, log, play_s=play_s)
    rig = make_rig(
        clock=clock,
        synth=synth,  # type: ignore[arg-type]
        player=player,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
        reply={"max_sentences": max_sentences},
    )
    return rig, clock, log, provider, synth, player


async def test_first_chunk_is_synthesized_while_the_llm_is_still_streaming() -> None:
    rig, clock, log, provider, synth, player = _streaming_rig(THREE_SENTENCE_DELTAS)
    t0 = clock()
    await rig.manager.on_utterance(utterance(1, t_eos=t0))
    await clock.run()
    await rig.manager.wait_idle()
    assert provider.generate_calls == 0  # stream() is preferred over generate()
    assert provider.closed is True
    assert synth.calls == ["First one here.", "Second one there.", "Third one gone."]
    # Deltas arrive at 0.5 s steps; "Second" (t=2.0) is the first character that
    # makes the "here." boundary final, so chunk 1 starts synthesis there.
    assert _starts(log, "delta", t0)[:4] == [0.5, 1.0, 1.5, 2.0]
    assert _starts(log, "synth_start", t0) == [2.0, 3.5, 4.5]
    stream_end = _starts(log, "stream_end", t0)
    assert stream_end == [4.5]
    assert _starts(log, "synth_start", t0)[0] < stream_end[0]
    # Chunk 1 is synthesized by 3.0 and plays at once; the rest follow back to back.
    assert _starts(log, "play_start", t0) == [3.0, 6.0, 9.0]
    assert [x for k, _, x in log if k == "play_start"] == [
        "First one here.",
        "Second one there.",
        "Third one gone.",
    ]
    assert (
        rig.conversation.history()[-1].text == "First one here. Second one there. Third one gone."
    )
    assert rig.manager.state is TurnState.IDLE


async def test_streamed_turn_metrics_end_asr_to_llm_at_the_first_chunk() -> None:
    rig, clock, log, _provider, _synth, _player = _streaming_rig(THREE_SENTENCE_DELTAS)
    await rig.manager.on_utterance(utterance(1, t_eos=100.0))
    await clock.run()
    await rig.manager.wait_idle()
    parsed = parse_turn_line(rig.metrics[0])
    assert parsed is not None
    # ASR is instant on this clock: first chunk ready at +2.0 s, synthesized at +3.0 s.
    assert parsed["eos->asr"] is None
    assert parsed["asr->llm"] == 2.0
    assert parsed["llm->tts1"] == 1.0
    assert parsed["tts1->audio"] is None
    assert parsed["total"] == 3.0


async def test_barge_in_while_the_llm_is_still_streaming_stops_the_stream() -> None:
    rig, clock, log, provider, synth, player = _streaming_rig(THREE_SENTENCE_DELTAS)
    t0 = clock()
    await rig.manager.on_utterance(utterance(1, t_eos=t0))
    await clock.run(until=t0 + 3.5)  # chunk 1 plays from 3.0; "Third" just arrived
    assert rig.manager.state is TurnState.SPEAKING
    await rig.manager.on_speech_started(speech(1, t_start=clock() - 0.5))
    await clock.run(until=clock() + 0.1)
    assert player.stop_calls == 1
    await clock.run()
    assert provider.closed is True
    assert _starts(log, "stream_end", t0) == []  # the stream was cut, not drained
    assert player.plays == [b"First one here."]
    assert rig.conversation.history()[-1].render() == (
        "Goku: First one here. Second one there. [interrupted after sentence 1]"
    )


async def test_stream_error_after_a_chunk_keeps_what_was_queued(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Four deltas arrive, then the provider fails: "First one here." is final,
    # the open "Second" is dropped.
    rig, clock, log, provider, synth, player = _streaming_rig(THREE_SENTENCE_DELTAS, fail_after=4)
    with caplog.at_level(logging.DEBUG, logger="notamusicbot.listen.turn_taking"):
        await rig.manager.on_utterance(utterance(1))
        await clock.run()
        await rig.manager.wait_idle()
    assert synth.calls == ["First one here."]
    assert player.plays == [b"First one here."]
    assert [t.render() for t in rig.conversation.history()] == [
        "User 1: tell me something fun",
        "Goku: First one here.",
    ]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [w.getMessage() for w in warnings] == ["LLM failed; rest dropped"]
    assert rig.manager.state is TurnState.IDLE
    assert len(rig.metrics) == 1


async def test_stream_error_before_any_chunk_drops_the_reply(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig, clock, log, provider, synth, player = _streaming_rig(THREE_SENTENCE_DELTAS, fail_after=2)
    with caplog.at_level(logging.DEBUG, logger="notamusicbot.listen.turn_taking"):
        await rig.manager.on_utterance(utterance(1))
        await clock.run()
        await rig.manager.wait_idle()
    assert synth.calls == []
    assert player.plays == []
    assert rig.metrics == []
    assert [t.is_bot for t in rig.conversation.history()] == [False]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [w.getMessage() for w in warnings] == ["LLM failed; turn dropped"]
    assert rig.manager.state is TurnState.IDLE


async def test_silent_stream_times_out_between_deltas() -> None:
    clock = TimelineClock(start=100.0)
    log: list[tuple[str, float, str]] = []

    class Silent(StreamingProvider):
        async def stream(self, request):  # type: ignore[no-untyped-def]
            await asyncio.sleep(10)
            yield "never"

    rig = make_rig(
        clock=clock,
        provider=Silent(clock, log, []),  # type: ignore[arg-type]
        reply={"llm_timeout_s": 0.05},
    )
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert rig.player.plays == []
    assert rig.manager.state is TurnState.IDLE


async def test_first_chunk_caps_come_from_settings() -> None:
    twelve = "one two three four five six seven eight nine ten eleven twelve."
    rig = make_rig(
        provider=FakeContentProvider([twelve]),
        reply={"first_chunk_max_words": 5, "later_chunk_max_words": 4},
    )
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert rig.synth.calls == [
        "one two three four five",
        "six seven eight nine",
        "ten eleven twelve.",
    ]


# ------------------------------------------------------------ early backchannel


def _bc_rig(**kwargs):  # type: ignore[no-untyped-def]
    clock = TimelineClock(start=100.0)
    log: list[tuple[str, float, str]] = []
    player = TimedPlayer(clock, log, play_s=1.0)
    rig = make_rig(
        clock=clock,
        player=player,  # type: ignore[arg-type]
        backchannel=FakeBackchannel(),
        **kwargs,
    )
    return rig, clock, log, player


async def test_early_backchannel_starts_at_end_of_speech_before_asr() -> None:
    gate = asyncio.Event()
    tr = UserTranscriber({1: "tell me something fun"})
    tr.gates = {1: gate}
    rig, clock, log, player = _bc_rig(transcriber=tr)
    t0 = clock()
    task = asyncio.create_task(rig.manager.on_utterance(utterance(1, audio_s=2.0)))
    await spin()
    # ASR is still blocked, and the clip already started (0.0 s after the utterance).
    assert tr.calls == [1]
    assert not task.done()
    assert player.plays == [BACKCHANNEL_PCM]
    assert _starts(log, "play_start", t0) == [0.0]
    gate.set()
    await task
    await clock.run()
    await rig.manager.wait_idle()
    assert player.plays == [BACKCHANNEL_PCM, REPLY]
    # The reply waits for the 1.0 s clip.
    assert _starts(log, "play_start", t0) == [0.0, 1.0]


async def test_no_early_backchannel_for_a_short_utterance() -> None:
    rig, clock, log, player = _bc_rig()
    await rig.manager.on_utterance(utterance(1, audio_s=0.8))
    await clock.run()
    await rig.manager.wait_idle()
    assert player.plays == [REPLY]
    # Exactly the threshold counts: 1.2 s starts the clip.
    edge, eclock, _elog, eplayer = _bc_rig()
    await edge.manager.on_utterance(utterance(1, audio_s=1.2))
    await eclock.run()
    await edge.manager.wait_idle()
    assert eplayer.plays == [BACKCHANNEL_PCM, REPLY]


async def test_no_early_backchannel_for_the_bot() -> None:
    rig, clock, log, player = _bc_rig()
    await rig.manager.on_utterance(utterance(BOT_ID, audio_s=2.0))
    await clock.run()
    await rig.manager.wait_idle()
    assert player.plays == []
    assert rig.transcriber.calls == []
    assert rig.provider.calls == []


async def test_dropped_utterance_keeps_its_clip_and_adds_no_second_one() -> None:
    rig, clock, log, player = _bc_rig(transcriber=FakeTranscriber(["Thank you."]))
    await rig.manager.on_utterance(utterance(1, audio_s=2.0))
    await clock.run()
    await rig.manager.wait_idle()
    assert player.plays == [BACKCHANNEL_PCM]  # accepted: played, then nothing
    assert rig.provider.calls == []
    assert rig.manager.state is TurnState.IDLE


async def test_backchannel_plays_once_per_turn_for_merged_utterances() -> None:
    tr = UserTranscriber({1: "hello there goku", 2: "what about me"})
    tr.gates = {1: asyncio.Event(), 2: asyncio.Event()}
    rig, clock, log, player = _bc_rig(transcriber=tr)
    a = asyncio.create_task(rig.manager.on_utterance(utterance(1, audio_s=2.0)))
    b = asyncio.create_task(rig.manager.on_utterance(utterance(2, audio_s=2.0)))
    await spin()
    tr.gates[1].set()
    tr.gates[2].set()
    await asyncio.gather(a, b)
    await clock.run()
    await rig.manager.wait_idle()
    assert len(rig.provider.calls) == 1
    assert player.plays == [BACKCHANNEL_PCM, REPLY]
    assert player.plays.count(BACKCHANNEL_PCM) == 1


async def test_no_early_backchannel_while_speaking() -> None:
    rig, clock, log, player = _bc_rig(
        transcriber=FakeTranscriber(["tell me something fun", "now tell me more"])
    )
    t0 = clock()
    await rig.manager.on_utterance(utterance(1, audio_s=2.0))
    await clock.run(until=t0 + 1.5)  # clip done at 1.0, reply playing
    assert rig.manager.state is TurnState.SPEAKING
    await rig.manager.on_utterance(utterance(1, audio_s=2.0))  # queued as the next turn
    await clock.run()
    await rig.manager.wait_idle()
    assert player.plays == [BACKCHANNEL_PCM, REPLY, REPLY]


async def test_no_early_backchannel_inside_cooldown_or_over_the_rate_limit() -> None:
    cool, cclock, _clog, cplayer = _bc_rig(reply={"per_user_cooldown_s": 5.0})
    await cool.manager.on_utterance(utterance(1, audio_s=2.0))
    await cclock.run()
    await cool.manager.wait_idle()
    assert cplayer.plays == [BACKCHANNEL_PCM, REPLY]
    cclock.now += 1.0  # inside the 5 s cooldown: the utterance will be dropped
    await cool.manager.on_utterance(utterance(1, audio_s=2.0))
    await cclock.run()
    await cool.manager.wait_idle()
    assert cplayer.plays == [BACKCHANNEL_PCM, REPLY]
    cclock.now += 10.0
    limited, lclock, _llog, lplayer = _bc_rig(reply={"max_replies_per_minute": 1})
    await limited.manager.on_utterance(utterance(1, audio_s=2.0))
    await lclock.run()
    await limited.manager.wait_idle()
    lclock.now += 5.0
    await limited.manager.on_utterance(utterance(1, audio_s=2.0))
    await lclock.run()
    await limited.manager.wait_idle()
    assert lplayer.plays == [BACKCHANNEL_PCM, REPLY]


async def test_early_backchannel_disabled_plays_only_the_reply() -> None:
    off, oclock, _olog, oplayer = _bc_rig(reply={"backchannel_enabled": False})
    await off.manager.on_utterance(utterance(1, audio_s=2.0))
    await oclock.run()
    await off.manager.wait_idle()
    assert oplayer.plays == [REPLY]


async def test_wake_word_mode_skips_the_early_backchannel_until_addressed() -> None:
    rig, clock, log, player = _bc_rig(
        respond_to="wake_word", transcriber=FakeTranscriber(["hey goku what is up"])
    )
    await rig.manager.on_utterance(utterance(1, audio_s=2.0))
    await clock.run()
    await rig.manager.wait_idle()
    assert player.plays == [REPLY]  # the wake word is only known after ASR
    clock.now += 2.0  # inside the follow-up window now
    rig.manager._transcriber = FakeTranscriber(["and another thing then"])  # type: ignore[attr-defined]
    await rig.manager.on_utterance(utterance(1, audio_s=2.0))
    await clock.run()
    await rig.manager.wait_idle()
    assert player.plays == [REPLY, BACKCHANNEL_PCM, REPLY]
