"""Unit tests for the naive commentary sentence-cap helper."""

from __future__ import annotations

from notamusicbot.content.trim import split_sentences, trim_sentences


def test_trim_keeps_short_text_unchanged() -> None:
    text = "Two sentences here. And one more!"
    assert trim_sentences(text, 3) == text


def test_trim_cuts_after_third_sentence_keeping_terminators() -> None:
    text = "One. Two? Three! Four. Five."
    assert trim_sentences(text, 3) == "One. Two? Three!"


def test_trim_handles_quotes_and_exclamations() -> None:
    text = 'He said "wait." Then he left! Really? Yes.'
    assert trim_sentences(text, 2) == 'He said "wait." Then he left!'


def test_trim_none_or_invalid_limit_returns_stripped_text() -> None:
    text = "  A. B. C.  "
    assert trim_sentences(text, None) == "A. B. C."
    assert trim_sentences(text, 0) == "A. B. C."
    assert trim_sentences(text, -1) == "A. B. C."


def test_trim_empty_text() -> None:
    assert trim_sentences("", 3) == ""
    assert trim_sentences("   ", 3) == ""


def test_split_sentences_drops_trailing_whitespace_fragments() -> None:
    assert split_sentences("A. B!\n\nC?") == ["A.", "B!", "C?"]
    assert split_sentences("no terminator") == ["no terminator"]
