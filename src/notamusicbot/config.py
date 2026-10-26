"""Typed configuration layer.

Secrets come from environment variables (loaded from ``.env``); non-secret
application configuration comes from YAML.  No settings object is constructed
at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def assert_loopback_url(url: str, *, what: str = "listen.llm.base_url") -> str:
    """Return ``url`` unchanged if its host is loopback, else raise ``ValueError``.

    Plan section 17: the owner's Spark is reached only through the SSH tunnel
    on loopback. Used by the config validator AND again by the Spark chat
    provider at call time, before any socket is opened.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.scheme not in {"http", "https"} or host not in LOOPBACK_HOSTS:
        allowed = ", ".join(sorted(LOOPBACK_HOSTS))
        raise ValueError(f"{what} host {host!r} is not loopback (allowed: {allowed})")
    return url


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


class VoiceConversionSettings(BaseModel):
    """Voice-conversion tuning that overrides the profile YAMLs.

    ``f0_method_override`` replaces the RVC f0 extraction method carried by
    the model YAMLs (``rvc_cli/models/*/``), which ship ``harvest`` — the
    slow CPU pitch extractor. ``pm`` (parselmouth) is the fast option
    (typically 3-10x faster); ``crepe``/``crepe-tiny`` run on the GPU.
    ``None`` (default) keeps each profile yaml's own value — no silent
    quality change for existing installs.
    """

    f0_method_override: str | None = Field(
        default=None,
        pattern="^(pm|harvest|dio|crepe|crepe-tiny|rmvpe)$",
    )
    # RVC output-quality knobs (docs/RVC_QUALITY_PLAN.md 3.3). Every default
    # reproduces the previous behaviour; ``use_index``, ``auto_pitch`` and the
    # ``rmvpe`` f0 methods need the quality work and the RMVPE weights.
    use_index: bool = False
    index_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    index_nprobe: int | None = Field(default=None, ge=1, le=256)
    protect: float | None = Field(default=None, ge=0.0, le=0.5)
    rms_mix_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    pitch_shift_offset: int = Field(default=0, ge=-24, le=24)
    auto_pitch: bool = False
    auto_pitch_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    auto_pitch_max: int = Field(default=12, ge=0, le=24)
    # RVC model name -> target median f0 (Hz); lookup is case-insensitive.
    target_f0_median_hz: dict[str, float] = Field(default_factory=dict)
    f0_post_median: int = Field(default=0, ge=0, le=101)
    f0_post_octave_repair: bool = False
    f0_post_clamp: tuple[float, float] | None = None
    stream_f0_algo: str = Field(default="pm", pattern="^(pm|rmvpe)$")
    # Opt-in streaming RVC (docs/RVC_STREAMING_DESIGN.md): the reply path
    # converts the TTS stream block by block instead of whole sentences.
    # Streams force the ``pm`` f0 method. ``False`` keeps today's behavior.
    streaming: bool = False
    stream_first_block_ms: int = Field(default=100, ge=20, le=1000)
    stream_block_ms: int = Field(default=320, ge=100, le=2000)
    stream_context_ms: int = Field(default=1000, ge=200, le=4000)
    stream_lookahead_ms: int = Field(default=100, ge=40, le=500)
    stream_crossfade_ms: int = Field(default=40, ge=10, le=200)
    stream_search_ms: int = Field(default=20, ge=0, le=100)

    @field_validator("f0_post_median")
    @classmethod
    def _validate_median_width(cls, value: int) -> int:
        if value != 0 and (value < 3 or value % 2 == 0):
            msg = f"f0_post_median must be 0 or an odd number >= 3 (got {value})"
            raise ValueError(msg)
        return value

    @field_validator("f0_post_clamp")
    @classmethod
    def _validate_clamp(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and not 0 < value[0] < value[1]:
            msg = f"f0_post_clamp must satisfy 0 < min < max (got {value})"
            raise ValueError(msg)
        return value

    @field_validator("target_f0_median_hz")
    @classmethod
    def _validate_targets(cls, value: dict[str, float]) -> dict[str, float]:
        for name, hz in value.items():
            if not 40.0 <= hz <= 600.0:
                msg = f"target_f0_median_hz[{name!r}] must be within 40..600 Hz (got {hz})"
                raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _validate_stream_timing(self) -> VoiceConversionSettings:
        for name in (
            "stream_first_block_ms",
            "stream_block_ms",
            "stream_context_ms",
            "stream_lookahead_ms",
            "stream_crossfade_ms",
            "stream_search_ms",
        ):
            if getattr(self, name) % 20 != 0:
                msg = f"{name} must be a multiple of 20 ms (got {getattr(self, name)})"
                raise ValueError(msg)
        if self.stream_crossfade_ms > self.stream_block_ms:
            msg = "stream_crossfade_ms must not exceed stream_block_ms"
            raise ValueError(msg)
        return self


class SpeechSettings(BaseModel):
    """Speech pipeline configuration (non-secret)."""

    voices_dir: Path = Path("config/voices")
    users_config: Path = Path("config/users.yaml")
    output_dir: Path = Path("var/speech")
    tts_device: str = "cuda"
    tts_dtype: str = "bfloat16"
    tts_attn_implementation: str = "sdpa"
    tts_model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    # faster-qwen3-tts (CUDA graphs, streaming); falls back to stock on failure.
    tts_fast: bool = True
    tts_stream_chunk_size: int = Field(default=8, ge=1)
    max_concurrent_inference: int = Field(default=1, ge=1)
    queue_size: int = Field(default=8, ge=1)
    warmup_on_start: bool = False
    sample_rate_hz: int = 48000
    channels: int = 2
    default_speed: float = Field(default=1.0, ge=0.25, le=4.0)
    default_target_peak_dbfs: float = Field(default=-1.0, le=0.0)
    # Review Round-2 P1: the commentary flow is the primary feature of this
    # rework, so fresh installs (config/app.example.yaml) start with it on.
    commentary_enabled: bool = True
    # Debug/benchmark mode: retain tts_source.wav/converted.wav/final.wav in the
    # job directory instead of applying the standard artifact cleanup policy.
    keep_artifacts: bool = False
    # Failed jobs keep their intermediate artifacts for diagnosis (marked with a
    # ``.failed`` timestamp); directories older than this many hours are pruned
    # opportunistically when the next speech job starts.
    failed_job_retention_hours: float = Field(default=24.0, ge=0.0)
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
    # Overrides for values carried by the RVC model YAMLs (see
    # VoiceConversionSettings). Empty section = yaml values win (shipped
    # default: f0_method harvest).
    voice_conversion: VoiceConversionSettings = VoiceConversionSettings()


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

    # Review Round-2 P1: enabled by default — a fresh setup must exercise the
    # content -> Qwen TTS -> RVC path during ordinary music playback.
    enabled: bool = True
    provider: str = "local_qwen"
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    device: str = "cuda"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    max_new_tokens: int = Field(default=120, ge=1)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    # Cap on spoken commentary length: generated text is trimmed to at most
    # this many sentences BEFORE TTS (TTS/RVC cost scales with audio length).
    # null disables trimming. The prompt already asks for 2-3 sentences; this
    # enforces it (the LLM regularly overshoots).
    max_sentences: int | None = Field(default=3, ge=1)
    # Prefetch post-song commentary while a STREAM song is still playing (the
    # generation runs through the shared InferenceCoordinator, so it
    # serializes with every other GPU job). At NATURAL completion a ready
    # prefetched artifact is enqueued immediately (near-zero wait); on
    # skip/stop/failure — or when the prefetch did not finish in time — it is
    # discarded and the at-completion generation path runs as before.
    prefetch_enabled: bool = True
    # NOTE: no per-content concurrency setting — LLM generation admission is
    # governed by the shared InferenceCoordinator (review Round-4 P2).
    system_prompt: str = ""  # empty -> prompts.RADIO_SYSTEM_PROMPT


class RedditSettings(BaseModel):
    """Reddit fetching settings (credentials live in secrets)."""

    # Empty list -> legacy behavior: the reddit front page.
    subreddits: list[str] = Field(default_factory=list)
    post_limit: int = Field(default=40, ge=1)
    user_agent: str = "windows:reddit_access:v1.0 (by u/Proud-Election-4152)"
    request_timeout_s: float = Field(default=10.0, gt=0.0)


class InterstitialSettings(BaseModel):
    """Randomized news/reddit interstitials between songs (legacy behavior).

    Disabled by default so defaults stay safe; enable explicitly in
    ``config/app.yaml``. Fires after a finished song with ``probability`` and
    at most once per ``min_interval_s``.
    """

    enabled: bool = False
    probability: float = Field(default=0.25, ge=0.0, le=1.0)
    min_interval_s: float = Field(default=300.0, ge=0.0)
    voice_profile: str = "goku"
    news_country: str = "us"
    news_category: str | None = None


class PodcastSettings(BaseModel):
    """Two-voice podcast conversation mode (``!gpodcast``)."""

    enabled: bool = False
    host_profile: str = "obama"
    # Replies after the host opening (total spoken lines = reply_turns + 1).
    reply_turns: int = Field(default=2, ge=1, le=6)


class ListenVadSettings(BaseModel):
    """Voice-activity detection and endpointing (CPU only, never the GPU)."""

    backend: Literal["silero", "energy"] = "silero"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Speech starts after this much sustained voice; ends after this much silence.
    min_speech_ms: int = Field(default=250, ge=1)
    end_silence_ms: int = Field(default=600, ge=1)
    max_utterance_s: float = Field(default=15.0, gt=0.0)
    pre_roll_ms: int = Field(default=300, ge=0)
    # Per-user bounded ring; the oldest audio is dropped beyond this.
    ring_buffer_s: float = Field(default=30.0, gt=0.0)

    @model_validator(mode="after")
    def _end_silence_covers_min_speech(self) -> ListenVadSettings:
        if self.end_silence_ms < self.min_speech_ms:
            raise ValueError("listen.vad.end_silence_ms must be >= listen.vad.min_speech_ms")
        return self


class ListenAsrSettings(BaseModel):
    """Speech recognition (local only, no speech API).

    Owner decision: Parakeet TDT 0.6B v2 int8 via onnx-asr on the CPU (4
    intra-op threads), so it never competes with the GPU. ``whisper_hf`` is the
    GPU fallback (Whisper small.en); it loads ``whisper_model_id``, and
    ``model_id`` / ``quantization`` / ``threads`` apply to the Parakeet backend.
    ``device: cuda`` runs Parakeet fp32 on the GPU regardless of ``quantization``
    (the int8 graph gains nothing there, so int8 is overridden with one warning);
    load runs a silent self-check and falls back to CPU int8 if CUDA/cuDNN fails.
    """

    backend: Literal["parakeet_onnx", "whisper_hf"] = "parakeet_onnx"
    model_id: str = "istupakov/parakeet-tdt-0.6b-v2-onnx"
    whisper_model_id: str = "openai/whisper-small.en"
    device: Literal["cpu", "cuda"] = "cpu"
    threads: int = Field(default=4, ge=1)
    quantization: Literal["int8", "fp32"] = "int8"
    language: str = "en"
    timeout_s: float = Field(default=10.0, gt=0.0)


class ListenReplySettings(BaseModel):
    """Spoken-reply shaping, barge-in and rate limits."""

    max_sentences: int = Field(default=2, ge=1)
    max_new_tokens: int = Field(default=60, ge=1)
    backchannel_enabled: bool = True
    backchannel_texts: list[str] = Field(
        default_factory=lambda: ["Hm?", "Oh yeah?", "Let me think."]
    )
    barge_in_min_speech_ms: int = Field(default=400, ge=1)
    per_user_cooldown_s: float = Field(default=2.0, ge=0.0)
    max_replies_per_minute: int = Field(default=12, ge=1)
    # In-memory rolling history only; never written to disk.
    history_turns: int = Field(default=8, ge=0)
    llm_timeout_s: float = Field(default=20.0, gt=0.0)
    # Streamed reply chunking: the first chunk is short so TTS starts early.
    first_chunk_max_words: int = Field(default=8, ge=3)
    later_chunk_max_words: int = Field(default=12, ge=3)
    # TTS length cap: max(tts_cap_floor_seconds, words * tts_max_seconds_per_word).
    tts_max_seconds_per_word: float = Field(default=1.0, gt=0.0)
    tts_cap_floor_seconds: float = Field(default=2.0, ge=0.0)


class ListenLlmSettings(BaseModel):
    """Where listen-mode reply text comes from (plan section 17).

    ``spark`` is the owner's Spark system over the loopback SSH tunnel (never
    any other host); ``local`` is the in-process Qwen3-4B. With ``spark`` and
    ``fallback: local`` an unreachable Spark falls back to the in-process model.
    """

    provider: Literal["spark", "local"] = "spark"
    base_url: str = "http://127.0.0.1:8000/v1"
    # "auto" = first id from GET {base_url}/models; never a hardcoded id.
    model: str = "auto"
    timeout_s: float = Field(default=20.0, gt=0.0)
    first_token_timeout_s: float = Field(default=8.0, gt=0.0)
    fallback: Literal["local", "none"] = "local"
    health_interval_s: float = Field(default=30.0, gt=0.0)
    # "off" sends thinking:false (about 1 s); "low" sends reasoning_effort low
    # (about +1-2 s first token, better answers). Reasoning tokens count against
    # max_tokens, so "low" adds ``thinking_token_budget`` to the reply limit.
    thinking: Literal["off", "low"] = "off"
    thinking_token_budget: int = Field(default=300, ge=1)
    # Hedged request (thinking "low" only): if the first request has produced no
    # content delta (reasoning deltas do not count) after this many seconds, start
    # a second thinking-off request and stream whichever yields text first.
    # None or 0 = off.
    hedge_after_s: float | None = Field(default=None, ge=0.0)

    @field_validator("base_url")
    @classmethod
    def _base_url_is_loopback(cls, value: str) -> str:
        return assert_loopback_url(value)


class ListenPrivacySettings(BaseModel):
    """Transcript logging and opt-in persistence (both off by default)."""

    # DEBUG-level transcript logging only when true.
    log_transcripts: bool = False
    # Owner decision: opt-ins are per session unless this is turned on.
    persist_opt_ins: bool = False


class ListenSettings(BaseModel):
    """Voice listen mode (``!glisten``).

    Off by default, and an empty ``allowed_channel_ids`` makes the session
    refuse to start even when ``enabled`` is true (enforced at session start,
    not here, so such a config still loads).
    """

    enabled: bool = False
    allowed_channel_ids: list[int] = Field(default_factory=list)
    # Empty -> guild administrators only.
    operator_user_ids: list[int] = Field(default_factory=list)
    voice_profile: str = "goku"
    announce_on_join: bool = True
    announcement_text: str = (
        "Heads up, I'm listening now. Only people who type listen on get "
        "transcribed, and nothing is saved."
    )
    respond_to: Literal["all", "wake_word"] = "all"
    wake_word: str = "goku"
    follow_up_window_s: float = Field(default=8.0, ge=0.0)
    vad: ListenVadSettings = ListenVadSettings()
    asr: ListenAsrSettings = ListenAsrSettings()
    reply: ListenReplySettings = ListenReplySettings()
    llm: ListenLlmSettings = ListenLlmSettings()
    privacy: ListenPrivacySettings = ListenPrivacySettings()


class AppSettings(BaseModel):
    """Non-secret application settings, normally loaded from ``config/app.yaml``."""

    speech: SpeechSettings = SpeechSettings()
    discord: DiscordSettings = DiscordSettings()
    media: MediaSettings = MediaSettings()
    content: ContentSettings = ContentSettings()
    reddit: RedditSettings = RedditSettings()
    interstitial: InterstitialSettings = InterstitialSettings()
    podcast: PodcastSettings = PodcastSettings()
    listen: ListenSettings = ListenSettings()


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
