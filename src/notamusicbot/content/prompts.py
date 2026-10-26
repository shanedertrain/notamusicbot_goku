"""Prompt construction for content generation.

The radio-host persona and style text are ported verbatim (in spirit) from the
legacy ``genai/genai_base.py`` prompts. This module is pure string work: no
model, no network, no I/O — safe to unit-test without a GPU.
"""

from __future__ import annotations

from notamusicbot.content.base import ContentKind, ContentRequest

# Ported from genai/genai_base.py: the bot hosts the 'Galactic Beats' radio
# show, speaks in first person, and stays short ("less is more", podcast.py).
RADIO_SYSTEM_PROMPT = (
    "You are the host of the 'Galactic Beats' radio show. "
    "Remember to speak in first person and stay in character. "
    "Your response is spoken aloud on the radio, so keep it short: "
    "2-3 sentences, plain spoken prose, no lists, no markdown, no emojis."
)

# Listen-mode voice chat (plan section 5): Goku is a conversation participant,
# not a radio host. Local-only; used as the system message for CHAT requests.
CHAT_SYSTEM_PROMPT = (
    "You are Goku, a friendly participant in a live voice chat. "
    "Your reply is spoken aloud, so keep it short: one or two short spoken "
    "sentences of plain prose. No lists, no markdown, no emojis, no stage "
    "directions, and no show-host patter. Speak in first person and answer "
    "what was just said. Never mention being an AI model unless you are asked."
    " Think briefly: reason in at most a couple of short lines, then answer in "
    "ONE conversational sentence of at most 18 words."
)


def chat_system_prompt(request: ContentRequest) -> str:
    """System message for a listen-mode reply: the persona's own prompt, else Goku's."""
    return request.system_prompt.strip() or CHAT_SYSTEM_PROMPT


def _persona_clause(persona_description: str) -> str:
    persona = persona_description.strip() or "a friendly radio host"
    return f"You are {persona} and remember to speak in first person."


def build_song_comment_prompt(request: ContentRequest) -> str:
    """Port of ``GenAIChat.generate_song_comment``."""
    requester = request.requester_name.strip() or "a listener"
    background = request.requester_background.strip()
    lines = [
        _persona_clause(request.persona_description),
        "Your response must be 2 - 3 sentences.",
        (
            f"You are announcing the song {request.song_name} "
            f"by {request.artist_name} as the host of the 'Galactic Beats' radio show."
        ),
        f"The requester is {requester}, who's known for being {background}.",
        (
            "Make a light-hearted joke about the requestor's background "
            "that ties into the song or its topic."
        ),
        "Do not repeat jokes about the requestor's background across prompts.",
    ]
    return "\n".join(lines)


def build_news_comment_prompt(request: ContentRequest) -> str:
    """Port of ``GenAIChat.generate_news_comment``."""
    return "\n".join(
        [
            _persona_clause(request.persona_description),
            "Your response must be 2 - 3 sentences.",
            "You are reading a news story as the host of the Galactic Beats radio show.",
            "What you will say is a summary of the following article text:",
            request.article_text,
        ]
    )


def build_reddit_comment_prompt(request: ContentRequest) -> str:
    """Port of ``GenAIChat.generate_reddit_post_comment``."""
    return "\n".join(
        [
            _persona_clause(request.persona_description),
            "Limit your response to maximum 3 paragraphs.",
            "You are summarizing a reddit post as the host of the Galactic Beats radio show.",
            f"Announce the title of the post: {request.post_title}.",
            f"The post content to be summarized is: {request.post_text}.",
        ]
    )


def build_chat_prompt(request: ContentRequest) -> str:
    """Prompt for a spoken reply in listen mode (plan section 5).

    ``request.chat_history`` already holds only the turns to include; the
    speaker's name comes from ``requester_name``.
    """
    speaker = request.requester_name.strip() or "a listener"
    lines = [
        _persona_clause(request.persona_description),
        (
            "You are in a live voice chat. Answer in one conversational sentence "
            "of at most 18 words, no lists, no markdown, no emojis, no stage directions."
        ),
    ]
    if request.chat_history:
        lines.append("Recent conversation:")
        lines.extend(request.chat_history)
    lines.append(f"{speaker} just said: {request.transcript_text.strip()}")
    lines.append("Your reply:")
    return "\n".join(lines)


def build_prompt(request: ContentRequest) -> str:
    """Dispatch to the prompt builder matching ``request.kind``."""
    builders = {
        ContentKind.SONG: build_song_comment_prompt,
        ContentKind.NEWS: build_news_comment_prompt,
        ContentKind.REDDIT: build_reddit_comment_prompt,
        ContentKind.CHAT: build_chat_prompt,
    }
    return builders[request.kind](request)


def static_song_comment(song_name: str, requester_name: str) -> str:
    """Static template commentary (previous pre-Phase-F behavior)."""
    requester = requester_name.strip() or "everyone"
    return f"Next up: {song_name}, requested by {requester}."


def static_article_comment(title: str) -> str:
    """Static template commentary for news/reddit items."""
    title = title.strip() or "this story"
    return f"Here is a quick look at: {title}."
