"""Unit tests for content-provider prompt construction and provider flow."""

from __future__ import annotations

import pytest

from notamusicbot.config import AppSettings, ContentSettings
from notamusicbot.content import prompts
from notamusicbot.content.base import ContentKind, ContentRequest, StaticTemplateProvider
from notamusicbot.discord_bot.commentary import resolve_commentary_text

SONG_REQUEST = ContentRequest(
    kind=ContentKind.SONG,
    persona_description="Energetic, friendly Saiyan radio host",
    song_name="Space Oddity",
    artist_name="David Bowie",
    requester_name="Andrew",
    requester_background="a right-wing, conspiracy theorist who loves guns and UFOs",
)


class FakeProvider:
    """Records the request and returns canned text."""

    def __init__(self, text: str = "fake commentary", exc: Exception | None = None) -> None:
        self.text = text
        self.exc = exc
        self.requests: list[ContentRequest] = []

    async def generate(self, request: ContentRequest) -> str:
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc
        return self.text


def test_radio_system_prompt_is_ported_from_legacy() -> None:
    assert "Galactic Beats" in prompts.RADIO_SYSTEM_PROMPT
    assert "first person" in prompts.RADIO_SYSTEM_PROMPT


def test_song_prompt_includes_persona_song_and_background() -> None:
    prompt = prompts.build_song_comment_prompt(SONG_REQUEST)
    assert "Energetic, friendly Saiyan radio host" in prompt
    assert "Space Oddity" in prompt
    assert "David Bowie" in prompt
    assert "Andrew" in prompt
    assert "conspiracy theorist" in prompt
    assert "Galactic Beats" in prompt
    assert "2 - 3 sentences" in prompt


def test_song_prompt_falls_back_to_generic_persona() -> None:
    prompt = prompts.build_song_comment_prompt(
        ContentRequest(kind=ContentKind.SONG, song_name="X", artist_name="Y")
    )
    assert "a friendly radio host" in prompt


def test_news_prompt_includes_article_text() -> None:
    prompt = prompts.build_news_comment_prompt(
        ContentRequest(kind=ContentKind.NEWS, article_text="Asteroid passes Earth.")
    )
    assert "Asteroid passes Earth." in prompt
    assert "Galactic Beats" in prompt


def test_reddit_prompt_includes_title_and_text() -> None:
    prompt = prompts.build_reddit_comment_prompt(
        ContentRequest(kind=ContentKind.REDDIT, post_title="Big news", post_text="Body text")
    )
    assert "Big news" in prompt
    assert "Body text" in prompt
    assert "3 paragraphs" in prompt


def test_build_prompt_dispatches_by_kind() -> None:
    assert "Space Oddity" in prompts.build_prompt(SONG_REQUEST)
    assert "Body text" in prompts.build_prompt(
        ContentRequest(kind=ContentKind.REDDIT, post_title="T", post_text="Body text")
    )


async def test_static_provider_song_comment() -> None:
    text = await StaticTemplateProvider().generate(SONG_REQUEST)
    assert text == "Next up: Space Oddity, requested by Andrew."


async def test_resolve_uses_provider_result() -> None:
    provider = FakeProvider("LLM says hi")
    assert await resolve_commentary_text(provider, SONG_REQUEST) == "LLM says hi"
    assert provider.requests[0].persona_description == SONG_REQUEST.persona_description


async def test_resolve_without_provider_falls_back_to_static() -> None:
    text = await resolve_commentary_text(None, SONG_REQUEST)
    assert text == "Next up: Space Oddity, requested by Andrew."


async def test_resolve_provider_failure_falls_back_to_static() -> None:
    provider = FakeProvider(exc=RuntimeError("GPU OOM"))
    text = await resolve_commentary_text(provider, SONG_REQUEST)
    assert text == "Next up: Space Oddity, requested by Andrew."


async def test_resolve_article_failure_falls_back_to_title() -> None:
    provider = FakeProvider(exc=RuntimeError("boom"))
    text = await resolve_commentary_text(
        provider, ContentRequest(kind=ContentKind.NEWS, post_title="Headline", article_text="body")
    )
    assert text == "Here is a quick look at: Headline."


def test_content_settings_defaults_match_local_qwen() -> None:
    settings = AppSettings()
    assert settings.content.provider == "local_qwen"
    assert settings.content.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert settings.content.max_concurrent_generations == 1
    assert settings.content.enabled is True


def test_disabled_content_yields_no_provider() -> None:
    app_settings = AppSettings(content=ContentSettings(enabled=False))
    assert _build_provider(app_settings) is None


def test_enabled_static_provider_is_constructed() -> None:
    app_settings = AppSettings(content=ContentSettings(enabled=True, provider="static"))
    provider = _build_provider(app_settings)
    assert isinstance(provider, StaticTemplateProvider)


def _build_provider(app_settings: AppSettings):
    from notamusicbot.app import Application

    return Application(app_settings).build_content_provider()


def test_runtime_checkable_protocol() -> None:
    from notamusicbot.content.base import ContentProvider as Protocol

    assert isinstance(FakeProvider(), Protocol)
    assert isinstance(StaticTemplateProvider(), Protocol)


@pytest.mark.parametrize("kind", [ContentKind.SONG, ContentKind.NEWS, ContentKind.REDDIT])
async def test_static_provider_never_raises(kind: ContentKind) -> None:
    text = await StaticTemplateProvider().generate(ContentRequest(kind=kind))
    assert isinstance(text, str) and text
