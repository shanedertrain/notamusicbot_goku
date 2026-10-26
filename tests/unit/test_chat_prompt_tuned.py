"""Tuned listen-mode reply prompt (plan section 17): one short sentence, brief thinking."""

from __future__ import annotations

from notamusicbot.content import prompts
from notamusicbot.content.base import ContentKind, ContentRequest

_THINK = (
    " Think briefly: reason in at most a couple of short lines, then answer in "
    "ONE conversational sentence of at most 18 words."
)


def _chat() -> ContentRequest:
    return ContentRequest(
        kind=ContentKind.CHAT,
        persona_description="Goku",
        requester_name="Bob",
        transcript_text="hi",
    )


def test_system_prompt_ends_with_the_tuned_suffix() -> None:
    assert prompts.CHAT_SYSTEM_PROMPT.endswith(_THINK)


def test_system_prompt_keeps_persona_and_rules() -> None:
    p = prompts.CHAT_SYSTEM_PROMPT
    assert p.startswith("You are Goku, a friendly participant in a live voice chat. ")
    assert "No lists, no markdown, no emojis, no stage directions" in p
    assert "Never mention being an AI model unless you are asked." in p


def test_chat_prompt_asks_for_one_short_sentence() -> None:
    text = prompts.build_chat_prompt(_chat())
    assert "one conversational sentence of at most 18 words" in text
    assert "one or two short spoken" not in text
    assert "no lists, no markdown, no emojis, no stage directions" in text


def test_non_listen_prompts_unchanged() -> None:
    song = ContentRequest(
        kind=ContentKind.SONG,
        persona_description="a Saiyan",
        song_name="S",
        artist_name="A",
        requester_name="Bob",
        requester_background="a farmer",
    )
    assert prompts.build_song_comment_prompt(song).split("\n")[:3] == [
        "You are a Saiyan and remember to speak in first person.",
        "Your response must be 2 - 3 sentences.",
        "You are announcing the song S by A as the host of the 'Galactic Beats' radio show.",
    ]
    news = ContentRequest(kind=ContentKind.NEWS, persona_description="a Saiyan", article_text="T")
    assert "18 words" not in prompts.build_news_comment_prompt(news)
    reddit = ContentRequest(kind=ContentKind.REDDIT, post_title="t", post_text="x")
    assert "18 words" not in prompts.build_reddit_comment_prompt(reddit)


def test_chat_system_prompt_defaults_to_goku_and_honours_persona_override() -> None:
    from notamusicbot.content.base import ContentKind, ContentRequest

    plain = ContentRequest(kind=ContentKind.CHAT)
    assert prompts.chat_system_prompt(plain) == prompts.CHAT_SYSTEM_PROMPT
    custom = ContentRequest(kind=ContentKind.CHAT, system_prompt="  You are Barack Obama.  ")
    assert prompts.chat_system_prompt(custom) == "You are Barack Obama."


def test_obama_profile_carries_its_own_chat_prompt() -> None:
    from pathlib import Path

    import yaml

    from notamusicbot.domain.voice_profiles import VoiceProfile

    path = Path(__file__).resolve().parents[2] / "config" / "voices" / "obama.yaml"
    profile = VoiceProfile(**yaml.safe_load(path.read_text()))
    assert "Barack Obama" in profile.persona.chat_prompt
    assert "Goku" not in profile.persona.chat_prompt
    assert profile.voice_conversion.model == "Obama"
