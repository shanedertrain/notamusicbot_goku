"""Listen-mode conversation, prompt and text-guard tests (CPU only, no models)."""

from __future__ import annotations

import random
from difflib import SequenceMatcher

import pytest

from notamusicbot.content.base import ContentKind
from notamusicbot.content.prompts import build_chat_prompt, build_prompt
from notamusicbot.listen.conversation import (
    Conversation,
    StreamChunk,
    StreamChunker,
    build_chat_request,
    clean_for_speech,
    echo_guard,
    is_noise_transcript,
    split_spoken_chunks,
    trim_reply,
)

BOT_SENTENCE = "The quick brown fox jumps over the lazy dog."
NEAR_70 = "the quick brown fox sat on the old mat today"
NEAR_30 = "hello there my friend how are you"
PERSONA = "Energetic, friendly Saiyan radio host."


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def test_echo_guard_identical_text() -> None:
    assert echo_guard(BOT_SENTENCE, BOT_SENTENCE) is True


def test_echo_guard_true_at_ratio_near_070() -> None:
    bot = "the quick brown fox jumps over the lazy dog"
    ratio = _ratio(NEAR_70, bot)
    assert 0.6 < ratio < 0.75  # 0.6667 for this pair
    assert echo_guard(NEAR_70, BOT_SENTENCE) is True


def test_echo_guard_false_at_ratio_near_030() -> None:
    bot = "the quick brown fox jumps over the lazy dog"
    ratio = _ratio(NEAR_30, bot)
    assert 0.25 < ratio < 0.35  # 0.3158 for this pair
    assert echo_guard(NEAR_30, BOT_SENTENCE) is False


def test_echo_guard_substring_ignores_case_and_punctuation() -> None:
    long_bot = "Well, the quick brown fox jumps over the lazy dog, my friend!"
    assert echo_guard("Quick BROWN fox!", long_bot) is True
    assert (
        _ratio("quick brown fox", "well the quick brown fox jumps over the lazy dog my friend")
        < 0.6
    )


def test_echo_guard_empty_is_false() -> None:
    assert echo_guard("", BOT_SENTENCE) is False
    assert echo_guard("hello", "") is False


def test_prompt_includes_only_last_two_of_five_turns() -> None:
    conv = Conversation(history_turns=2)
    conv.add_user("Alice", "old-one")
    conv.add_bot("old-two")
    conv.add_user("Alice", "old-three")
    conv.add_bot("recent-four")
    conv.add_user("Bob", "recent-five")
    assert [t.text for t in conv.history()] == ["recent-four", "recent-five"]

    request = build_chat_request(PERSONA, "Bob", "what is up", conv.history())
    assert request.kind is ContentKind.CHAT
    prompt = build_chat_prompt(request)
    assert build_prompt(request) == prompt
    assert "Bob" in prompt
    assert "recent-four" in prompt
    assert "recent-five" in prompt
    for old in ("old-one", "old-two", "old-three"):
        assert old not in prompt
    assert "Saiyan" in prompt
    assert "one conversational sentence of at most 18 words" in prompt
    assert "no lists, no markdown, no emojis, no stage directions" in prompt
    assert "Bob just said: what is up" in prompt


def test_prompt_without_history_has_no_history_header() -> None:
    request = build_chat_request(PERSONA, "Bob", "hello there", [])
    assert "Recent conversation" not in build_chat_prompt(request)


def test_interrupted_marker_format() -> None:
    conv = Conversation(history_turns=8)
    conv.add_user("Alice", "tell me a story")
    conv.add_bot("Once upon a time. There was a dragon.")
    conv.mark_interrupted(1)
    rendered = [t.render() for t in conv.history()]
    assert rendered == [
        "Alice: tell me a story",
        "Goku: Once upon a time. There was a dragon. [interrupted after sentence 1]",
    ]
    request = build_chat_request(PERSONA, "Alice", "stop", conv.history())
    assert "[interrupted after sentence 1]" in build_chat_prompt(request)


def test_mark_interrupted_without_bot_turn_is_noop() -> None:
    conv = Conversation(history_turns=3)
    conv.add_user("Alice", "hi there")
    conv.mark_interrupted(2)
    assert conv.history()[0].interrupted_after is None


def test_clear_and_zero_history() -> None:
    conv = Conversation(history_turns=3)
    conv.add_user("Alice", "hi there")
    conv.clear()
    assert conv.history() == []
    zero = Conversation(history_turns=0)
    zero.add_user("Alice", "hi there")
    assert zero.history() == []


def test_trim_reply() -> None:
    assert trim_reply("A. B. C.", 2) == "A. B."
    assert trim_reply("A. B.", 5) == "A. B."
    assert trim_reply("  A. B. C. ", 0) == "A. B. C."


def test_is_noise_transcript() -> None:
    for noisy in ("", ".", "Thank you.", "ok", "  Thanks for watching.  "):
        assert is_noise_transcript(noisy) is True, noisy
    assert is_noise_transcript("what is up") is False


STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "and",
    "or",
    "but",
    "that",
    "is",
    "with",
    "for",
    "at",
    "by",
}
LONG_22 = (
    "I really think that the best food in the whole world is ramen \u2014 "
    "especially the spicy kind from the little shop downtown."
)


def test_clean_for_speech_values() -> None:
    assert clean_for_speech("Wow \U0001f680 that's *great*!") == "Wow that's great!"
    assert clean_for_speech("Sure *laughs* thing") == "Sure thing"
    assert clean_for_speech("Sure *clears throat*, thing (grins) ok") == "Sure, thing ok"
    assert clean_for_speech("# Title\n- `code` item\n* other") == "Title code item other"
    assert clean_for_speech("") == ""


def test_trim_reply_interjection_not_counted() -> None:
    assert trim_reply("Hey! I love ramen. It is good. Yes.", 2) == "Hey! I love ramen. It is good."


def test_chunks_interjection_merges() -> None:
    assert split_spoken_chunks("Hey! I love ramen.", 2) == ["Hey! I love ramen."]


def test_chunks_sentence_cap() -> None:
    assert " ".join(split_spoken_chunks("A. B. C.", 2)) == "A. B."
    assert split_spoken_chunks("", 2) == []


def test_chunks_long_sentence_caps_and_stopwords() -> None:
    chunks = split_spoken_chunks(LONG_22, 2)
    assert chunks == [
        "I really think that the best food in the whole",
        "world is ramen \u2014 especially the spicy kind from the little shop downtown.",
    ]
    assert " ".join(chunks) == LONG_22
    assert all(c.split()[-1].strip(".,").lower() not in STOP for c in chunks)


def test_chunks_prefer_comma_and_avoid_stop_word() -> None:
    text = "Well, I think we should go to the park and then eat some good food."
    chunks = split_spoken_chunks(text, 2)
    assert chunks == ["Well, I think we should go to the park", "and then eat some good food."]


def test_chunks_no_punctuation_25_words() -> None:
    text = " ".join(f"w{i}" for i in range(25))
    chunks = split_spoken_chunks(text, 2)
    assert [len(c.split()) for c in chunks] == [10, 15]
    assert " ".join(chunks) == text


def test_chunks_avoid_stranded_tail() -> None:
    text = " ".join(f"w{i}" for i in range(11))
    assert [len(c.split()) for c in split_spoken_chunks(text, 2)] == [8, 3]


# --------------------------------------------------------------- StreamChunker


def _random_splits(text: str, rng: random.Random, count: int) -> list[list[str]]:
    """``count`` ways to cut ``text`` into 0-25 deltas (cut points are random)."""
    splits = []
    for _ in range(count):
        cuts = sorted(rng.sample(range(len(text) + 1), min(len(text) + 1, rng.randint(0, 25))))
        parts, prev = [], 0
        for cut in cuts:
            parts.append(text[prev:cut])
            prev = cut
        parts.append(text[prev:])
        splits.append(parts)
    return splits


def _stream(deltas: list[str], max_sentences: int, first: int, later: int) -> list[str]:
    chunker = StreamChunker(max_sentences, first, later)
    out: list[str] = []
    for delta in deltas:
        out.extend(chunker.feed(delta))
    out.extend(chunker.finish())
    return out


def _words(chunk: str) -> int:
    return len([w for w in chunk.split() if w != "\u2014"])  # a lone dash is not a word


def _norm(chunks: list[str]) -> str:
    return " ".join(" ".join(chunks).split())


@pytest.mark.parametrize(("first", "later"), [(10, 18), (8, 12), (3, 3)])
def test_stream_chunker_matches_batch_for_20_random_splits_of_the_ramen_text(
    first: int, later: int
) -> None:
    rng = random.Random(7)
    reference = split_spoken_chunks(LONG_22, 2, first, later)
    assert len([w for w in LONG_22.split() if w != "\u2014"]) == 22
    for deltas in _random_splits(LONG_22, rng, 20):
        chunks = _stream(deltas, 2, first, later)
        assert _norm(chunks) == _norm(reference) == " ".join(LONG_22.split())
        assert chunks == reference  # stronger than the invariant: same cuts too
        assert _words(chunks[0]) <= first
        assert all(_words(c) <= later for c in chunks[1:])


def test_stream_chunker_ramen_reference_values() -> None:
    assert split_spoken_chunks(LONG_22, 2, 8, 12) == [
        "I really think that the best food",
        "in the whole world is ramen \u2014",
        "especially the spicy kind from the little shop downtown.",
    ]
    assert _stream([w + " " for w in LONG_22.split(" ")], 2, 10, 18) == [
        "I really think that the best food in the whole",
        "world is ramen \u2014 especially the spicy kind from the little shop downtown.",
    ]


TRICKY = [
    "Hey! I love ramen. It is good, really good. Yes sir that is true; I would eat it all day "
    "long, every day of the week! Wow.",
    "Well, I think we should go to the park and then eat some good food, and also play a game "
    "or two before the sun goes down. Sure *laughs* thing (grins) ok. Fine.",
    " ".join(f"w{i}" for i in range(25)) + ". Next one here! Hey?",
    'Wow!! Really... that is wild? "Yes." He said. 3.5 is bigger than 3.',
    "A. B. C. D.",
    "Hey! Wow!",
    "# Title\n- item one is here\n- item two is here and it is rather long indeed yes it is\nDone.",
]


@pytest.mark.parametrize("max_sentences", [0, 1, 2, 3])
@pytest.mark.parametrize("text", TRICKY)
def test_stream_chunker_matches_batch_on_tricky_texts(text: str, max_sentences: int) -> None:
    rng = random.Random(7)
    reference = split_spoken_chunks(text, max_sentences, 8, 12)
    for deltas in _random_splits(text, rng, 20):
        assert _stream(deltas, max_sentences, 8, 12) == reference


def test_stream_chunker_single_delta_and_empty() -> None:
    assert _stream([LONG_22], 2, 10, 18) == split_spoken_chunks(LONG_22, 2, 10, 18)
    assert _stream([], 2, 10, 18) == []
    assert _stream(["   "], 2, 10, 18) == []
    chunker = StreamChunker(2)
    assert chunker.feed("") == []
    assert chunker.finish() == []
    assert chunker.feed("more") == []  # nothing after finish


def test_stream_chunker_emits_long_sentence_chunk_before_the_text_ends() -> None:
    chunker = StreamChunker(2, 10, 18)
    words = LONG_22.split(" ")
    emitted_at: list[tuple[int, list[str]]] = []
    for i, word in enumerate(words):
        out = chunker.feed(word + " ")
        if out:
            emitted_at.append((i, out))
    # The plain cut after "whole" needs 13 stable words (cap 10 + 3): word 13 is the dash.
    assert emitted_at == [(13, ["I really think that the best food in the whole"])]
    assert chunker.finish() == [
        "world is ramen \u2014 especially the spicy kind from the little shop downtown."
    ]


def test_stream_chunker_sentence_is_final_once_the_next_sentence_starts() -> None:
    chunker = StreamChunker(3, 8, 12)
    steps = ["Hey!", " I", " love", " ramen.", " It", " is", " good.", " Yes", " sir."]
    got = [(d, chunker.feed_chunks(d)) for d in steps]
    assert [(d, c) for d, c in got if c] == [
        (" It", [StreamChunk("Hey! I love ramen.", 1)]),  # "Hey!" merged, held until here
        (" Yes", [StreamChunk("It is good.", 2)]),
    ]
    assert chunker.finish_chunks() == [StreamChunk("Yes sir.", 3)]
    assert chunker.sentence_text(1) == "Hey! I love ramen."
    assert chunker.sentence_text(3) == "Yes sir."
    assert chunker.done is True


def test_stream_chunker_stops_after_max_sentences_and_ignores_the_rest() -> None:
    chunker = StreamChunker(2, 8, 12)
    assert chunker.feed("One is here. Two is here. Three") == [
        "One is here.",
        "Two is here.",
    ]
    assert chunker.done is True
    assert chunker.feed(" is here. Four is here.") == []
    assert chunker.finish() == []
    # A trailing interjection is a unit of its own only while a slot is free.
    assert _stream(["One is here. Hey!"], 2, 8, 12) == ["One is here.", "Hey!"]
    assert _stream(["One is here. Two is here. Hey!"], 2, 8, 12) == ["One is here.", "Two is here."]


def test_stream_chunker_holds_text_inside_an_open_paren_or_star() -> None:
    chunker = StreamChunker(3, 8, 12)
    assert chunker.feed("Sure thing. (grins and nods. Then ") == []  # "(" still open
    assert chunker.feed("laughs) Anyway it works. And") == [
        "Sure thing.",
        "Anyway it works.",
    ]
    star = StreamChunker(3, 8, 12)
    assert star.feed("Okay then. *clears throat. Hmm* Yes it is. Next") == [
        "Okay then.",
        "Yes it is.",
    ]


def test_stream_chunker_never_loses_or_reorders_text_without_stop_or_cap() -> None:
    text = " ".join(f"word{i}" for i in range(60))
    chunks = _stream([w + " " for w in text.split(" ")], 0, 8, 12)
    assert " ".join(chunks) == text
    assert len(chunks[0].split()) <= 8
    assert all(len(c.split()) <= 12 for c in chunks[1:])


def test_extend_bot_appends_only_to_a_trailing_bot_turn() -> None:
    conv = Conversation(history_turns=4)
    conv.extend_bot("nobody")  # no turns: no-op
    conv.add_user("Alice", "hi there")
    conv.extend_bot("not a bot turn")
    assert [t.text for t in conv.history()] == ["hi there"]
    conv.add_bot("First one here.")
    conv.extend_bot("Second one there.")
    conv.extend_bot("  ")
    assert [t.render() for t in conv.history()] == [
        "Alice: hi there",
        "Goku: First one here. Second one there.",
    ]
