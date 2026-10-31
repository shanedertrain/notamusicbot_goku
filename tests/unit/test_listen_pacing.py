"""Adaptive chunk budget: how many words the next chunk may hold."""

from __future__ import annotations

import pytest

from notamusicbot.listen.pacing import (
    INITIAL_RTF,
    MAX_CHUNK_WORDS,
    SAFETY_FACTOR,
    SAFETY_LEAD_S,
    ChunkPacer,
)


def test_first_chunk_is_never_padded() -> None:
    assert ChunkPacer().budget_words(0.0) == 0


def test_budget_grows_with_the_audio_queued_ahead() -> None:
    pacer = ChunkPacer()
    pacer.note_delivered(2.0)
    small = pacer.budget_words(10.0)
    pacer.note_delivered(4.0)
    assert pacer.budget_words(10.0) > small > 0


def test_budget_matches_the_rtf_formula() -> None:
    pacer = ChunkPacer()
    pacer.note_delivered(4.0)
    usable = 4.0 - SAFETY_LEAD_S
    expected = int(usable / (INITIAL_RTF * SAFETY_FACTOR) * pacer.words_per_s)
    assert pacer.budget_words(0.0) == expected


def test_playback_progress_eats_the_lead() -> None:
    pacer = ChunkPacer()
    pacer.note_delivered(4.0)
    pacer.note_playback_started(100.0)
    assert pacer.lead_s(100.0) == pytest.approx(4.0)
    assert pacer.lead_s(102.5) == pytest.approx(1.5)
    assert pacer.lead_s(120.0) == 0.0
    assert pacer.budget_words(120.0) == 0


def test_slower_synthesis_shrinks_the_budget() -> None:
    fast, slow = ChunkPacer(), ChunkPacer()
    for pacer, took in ((fast, 1.0), (slow, 4.0)):
        pacer.note_delivered(4.0)
        for _ in range(10):
            pacer.note_synthesis(words=10, audio_s=4.0, took_s=took)
    assert slow.rtf > fast.rtf
    assert slow.budget_words(0.0) < fast.budget_words(0.0)


def test_budget_is_capped() -> None:
    pacer = ChunkPacer()
    pacer.note_delivered(600.0)
    assert pacer.budget_words(0.0) == MAX_CHUNK_WORDS


def test_begin_turn_resets_the_queue_but_keeps_the_speed() -> None:
    pacer = ChunkPacer()
    for _ in range(10):
        pacer.note_synthesis(words=10, audio_s=4.0, took_s=2.0)
    learned = pacer.rtf
    pacer.note_delivered(5.0)
    pacer.note_playback_started(1.0)
    pacer.begin_turn()
    assert pacer.lead_s(1.0) == 0.0
    assert pacer.rtf == learned


def test_degenerate_measurements_are_ignored() -> None:
    pacer = ChunkPacer()
    before = (pacer.rtf, pacer.words_per_s)
    pacer.note_synthesis(words=0, audio_s=3.0, took_s=1.0)
    pacer.note_synthesis(words=5, audio_s=0.0, took_s=1.0)
    assert (pacer.rtf, pacer.words_per_s) == before


def test_opener_target_applies_only_before_any_audio() -> None:
    pacer = ChunkPacer()
    assert pacer.opener_target_words() > 1
    pacer.note_delivered(2.0)
    assert pacer.opener_target_words() == 0
