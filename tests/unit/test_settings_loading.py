"""Tests for load_app_settings() default-path resolution (P0-3).

Chain: explicit arg → NOTAMUSICBOT_CONFIG env → config/app.yaml → defaults.
"""

from __future__ import annotations

from pathlib import Path

from notamusicbot.config import AppSettings, load_app_settings, resolve_app_config_path


def test_explicit_path_wins_over_everything(tmp_path: Path, monkeypatch) -> None:
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("speech:\n  queue_size: 3\n")
    monkeypatch.setenv("NOTAMUSICBOT_CONFIG", str(tmp_path / "env.yaml"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "app.yaml").write_text("speech:\n  queue_size: 7\n")
    settings = load_app_settings(explicit)
    assert settings.speech.queue_size == 3


def test_env_var_overrides_default_path(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / "custom.yaml"
    env_file.write_text("discord:\n  command_prefix: '!x'\n")
    monkeypatch.setenv("NOTAMUSICBOT_CONFIG", str(env_file))
    assert resolve_app_config_path() == env_file
    settings = load_app_settings()
    assert settings.discord.command_prefix == "!x"


def test_default_path_reads_config_app_yaml(tmp_path: Path, monkeypatch) -> None:
    """A distinctive value in config/app.yaml must win over built-in defaults."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "app.yaml").write_text(
        "speech:\n  queue_size: 42\n  commentary_enabled: true\ncontent:\n  enabled: true\n"
    )
    settings = load_app_settings()
    assert resolve_app_config_path() == Path("config/app.yaml")
    assert settings.speech.queue_size == 42
    assert settings.speech.commentary_enabled is True
    assert settings.content.enabled is True


def test_missing_default_file_still_gives_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # no config/app.yaml here
    monkeypatch.delenv("NOTAMUSICBOT_CONFIG", raising=False)
    settings = load_app_settings()
    assert settings == AppSettings()


def test_default_path_dunder_repo_uses_repo_config(monkeypatch) -> None:
    """In the repo root, load_app_settings() must resolve config/app.yaml."""
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    assert resolve_app_config_path() == Path("config/app.yaml")
    # config/app.yaml is not committed, so defaults apply unless a developer
    # scaffolded it; both outcomes must load cleanly.
    settings = load_app_settings()
    assert isinstance(settings, AppSettings)
