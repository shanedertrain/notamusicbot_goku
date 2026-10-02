"""Typed configuration layer.

Secrets come from environment variables (loaded from ``.env``); non-secret
application configuration comes from YAML.  No settings object is constructed
at import time.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecretsSettings(BaseSettings):
    """Secret values read from the environment / ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    discord_token: str | None = Field(default=None, alias="DISCORD_TOKEN")
    discord_guild: str | None = Field(default=None, alias="DISCORD_GUILD")
    # No LLM API key: content generation runs locally (Qwen3-4B, handoff 14
    # override). Legacy GEMINI/GEMENI_API_KEY values are simply ignored.
    news_api_key: str | None = Field(default=None, alias="NEWS_API_KEY")
    reddit_client_id: str | None = Field(default=None, alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str | None = Field(default=None, alias="REDDIT_CLIENT_SECRET")
    spotify_client_id: str | None = Field(default=None, alias="SPOTIFY_CLIENT_ID")
    spotify_client_secret: str | None = Field(default=None, alias="SPOTIFY_CLIENT_SECRET")


class SpeechSettings(BaseModel):
    """Speech pipeline configuration (non-secret)."""

    voices_dir: Path = Path("config/voices")
    users_config: Path = Path("config/users.yaml")
    output_dir: Path = Path("var/speech")
    tts_device: str = "cuda"
    tts_dtype: str = "bfloat16"
    tts_attn_implementation: str = "sdpa"
    tts_model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    max_concurrent_inference: int = Field(default=1, ge=1)
    queue_size: int = Field(default=8, ge=1)
    warmup_on_start: bool = False
    sample_rate_hz: int = 48000
    channels: int = 2
    default_speed: float = Field(default=1.0, ge=0.25, le=4.0)
    default_target_peak_dbfs: float = Field(default=-1.0, le=0.0)
    commentary_enabled: bool = False
    # Voice-conversion backends the router may serve (see
    # speech/voice_conversion/router.py). Profiles requesting another backend
    # fail with a clear error instead of silently falling back.
    voice_conversion_backends: list[str] = Field(default_factory=lambda: ["passthrough", "rvc"])
    # What to do when a profile requests the 'rvc' backend but the RVC worker
    # is unavailable (no GPU, missing worker runtime):
    #   error       -> raise; commentary generation is skipped (default: never
    #                  emit un-converted source voice)
    #   passthrough -> degrade to passthrough with a warning
    rvc_unavailable_policy: str = "error"


class DiscordSettings(BaseModel):
    """Discord-facing settings (non-secret; token lives in secrets)."""

    command_prefix: str = "!g"
    # Optional sound played when the bot first joins a voice channel.
    start_sound: Path | None = None
    # Delay between deletions in the deleteallmessages cleanup command.
    message_delete_delay_s: float = Field(default=1.0, ge=0.0)


class MediaSettings(BaseModel):
    """yt-dlp / media acquisition settings (non-secret)."""

    yt_clients: list[str] = Field(default_factory=lambda: ["default", "web", "tv_embedded"])
    cookies_file: Path = Path("input/youtube_cookies.txt")
    cache_dir: Path = Path("var/cache/youtube")
    js_runtimes: dict[str, Path] = Field(default_factory=lambda: {"deno": Path("~/.deno/bin/deno")})
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )


class ContentSettings(BaseModel):
    """Local content-generation settings (handoff section 14 override: no cloud LLM).

    ``provider`` selects the implementation: ``local_qwen`` (LLM on the local
    GPU) or ``static`` (template only). Model loading is always deferred to
    startup — never at import time.
    """

    enabled: bool = False
    provider: str = "local_qwen"
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    device: str = "cuda"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    max_new_tokens: int = Field(default=120, ge=1)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_concurrent_generations: int = Field(default=1, ge=1)
    system_prompt: str = ""  # empty -> prompts.RADIO_SYSTEM_PROMPT


class AppSettings(BaseModel):
    """Non-secret application settings, normally loaded from ``config/app.yaml``."""

    speech: SpeechSettings = SpeechSettings()
    discord: DiscordSettings = DiscordSettings()
    media: MediaSettings = MediaSettings()
    content: ContentSettings = ContentSettings()


DEFAULT_APP_CONFIG_PATH = Path("config/app.yaml")


def resolve_app_config_path(path: Path | None = None) -> Path | None:
    """Resolve the app-config file via the documented chain.

    Resolution order: explicit argument → ``NOTAMUSICBOT_CONFIG`` env override
    → ``config/app.yaml`` → ``None`` (built-in defaults). A missing file is not
    an error; callers fall back to defaults.
    """
    if path is not None:
        return path
    import os

    env_path = os.environ.get("NOTAMUSICBOT_CONFIG")
    if env_path:
        return Path(env_path)
    return DEFAULT_APP_CONFIG_PATH


def load_app_settings(path: Path | None = None) -> AppSettings:
    """Load app settings from YAML (or defaults when the file is absent).

    ``path=None`` resolves the default ``config/app.yaml`` (see
    :func:`resolve_app_config_path`); when that file does not exist, built-in
    defaults apply. When it exists, its values win over the defaults.
    """
    resolved = resolve_app_config_path(path)
    if resolved is None or not resolved.is_file():
        return AppSettings()
    import yaml

    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return AppSettings.model_validate(data)
