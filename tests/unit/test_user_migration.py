"""Migration tests using the real legacy input shape (29 OBAMA / 1 OBAMA_TRANS)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from notamusicbot.domain.voice_profiles import load_voice_profiles
from notamusicbot.tools.migrate_users import (
    MigrationError,
    migrate_users_to_profiles,
    resolve_legacy_profile,
)

REPO = Path(__file__).resolve().parents[2]


def _legacy_shape_entry(name: str, background: str = "No Background") -> dict:
    """One user entry in the exact legacy input/users.json shape."""
    return {
        "screen_name": name,
        "real_name": name,
        "background": background,
        "model_name": "OBAMA",
    }


@pytest.fixture()
def legacy_users_file(tmp_path: Path) -> Path:
    """29 users -> OBAMA, 1 user -> OBAMA_TRANS (real legacy dict shape)."""
    users = {f"id{i}": _legacy_shape_entry(f"user{i}") for i in range(29)}
    users["id-trans"] = _legacy_shape_entry("trans-user")
    users["id-trans"]["model_name"] = "OBAMA_TRANS"
    path = tmp_path / "users.json"
    path.write_text(json.dumps(users), encoding="utf-8")
    return path


@pytest.fixture()
def voices_dir(tmp_path: Path) -> Path:
    """Minimal profile set mirroring config/voices (ids obama/obama_trans)."""
    voices = tmp_path / "voices"
    voices.mkdir()
    for pid, model in (("obama", "Obama"), ("obama_trans", "Obama"), ("goku", "Goku")):
        (voices / f"{pid}.yaml").write_text(
            yaml.safe_dump(
                {
                    "id": pid,
                    "persona": {"display_name": pid, "description": "test"},
                    "tts": {"speaker": "Ryan", "language": "English"},
                    "voice_conversion": {"backend": "rvc", "model": model},
                    "audio": {"speed": 1.0, "target_peak_dbfs": -1.0},
                }
            ),
            encoding="utf-8",
        )
    return voices


def test_real_legacy_file_migrates_identity() -> None:
    """Against the real input/users.json: identity survives, no goku fallback."""
    source = REPO / "input" / "users.json"
    if not source.is_file():  # legacy file not shipped in every checkout
        pytest.skip("input/users.json not present")
    dest = REPO / "config" / "users.yaml"
    entries = migrate_users_to_profiles(source, dest, REPO / "config" / "voices")
    counts: dict[str, int] = {}
    for entry in entries.values():
        counts[entry["voice_profile"]] = counts.get(entry["voice_profile"], 0) + 1
    assert counts == {"obama": 29, "obama_trans": 1}
    assert not any(e["voice_profile"] == "goku" for e in entries.values())
    # OBAMA_TRANS resolves through an alias (weights shared with Obama) and
    # must be recorded as such, not silently conflated.
    trans = [e for e in entries.values() if e["voice_profile"] == "obama_trans"]
    assert len(trans) == 1
    assert trans[0]["legacy_model_name"] == "OBAMA_TRANS"
    assert trans[0]["voice_profile_source"] == "legacy_alias"


def test_migration_maps_obama_and_obama_trans(
    legacy_users_file: Path, tmp_path: Path, voices_dir: Path
) -> None:
    dest = tmp_path / "out" / "users.yaml"
    entries = migrate_users_to_profiles(legacy_users_file, dest, voices_dir)
    assert len(entries) == 30
    assert sum(e["voice_profile"] == "obama" for e in entries.values()) == 29
    assert sum(e["voice_profile"] == "obama_trans" for e in entries.values()) == 1
    written = yaml.safe_load(dest.read_text(encoding="utf-8"))["users"]
    assert written["id28"]["voice_profile"] == "obama"
    assert written["id-trans"]["voice_profile"] == "obama_trans"


def test_migration_never_defaults_to_goku(
    legacy_users_file: Path, tmp_path: Path, voices_dir: Path
) -> None:
    dest = tmp_path / "users.yaml"
    entries = migrate_users_to_profiles(legacy_users_file, dest, voices_dir)
    assert all(e["voice_profile"] != "goku" for e in entries.values())


def test_unknown_legacy_model_fails_loudly(tmp_path: Path, voices_dir: Path) -> None:
    source = tmp_path / "users.json"
    source.write_text(
        json.dumps({"u1": {"screen_name": "x", "model_name": "BARBIE"}}),
        encoding="utf-8",
    )
    dest = tmp_path / "users.yaml"
    with pytest.raises(MigrationError, match="BARBIE"):
        migrate_users_to_profiles(source, dest, voices_dir)
    assert not dest.exists()  # nothing written on failure


def test_resolve_is_case_insensitive_and_uses_model_dir(
    voices_dir: Path,
) -> None:
    profiles = load_voice_profiles(voices_dir)
    assert resolve_legacy_profile("OBAMA", profiles) == "obama"
    assert resolve_legacy_profile("obama_trans", profiles) == "obama_trans"
    # GOKU hits the second branch: the RVC model directory of the goku profile.
    assert resolve_legacy_profile("Goku", profiles) == "goku"
    with pytest.raises(MigrationError):
        resolve_legacy_profile("NOPE", profiles)


def test_cli_exits_nonzero_on_unknown_model(
    legacy_users_file: Path, tmp_path: Path, voices_dir: Path, monkeypatch
) -> None:
    import subprocess
    import sys

    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"u1": {"model_name": "BARBIE"}}), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "notamusicbot.tools.migrate_users",
            "--source",
            str(source),
            "--dest",
            str(tmp_path / "users.yaml"),
            "--voices-dir",
            str(voices_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode != 0
    assert "BARBIE" in (result.stderr + result.stdout)
