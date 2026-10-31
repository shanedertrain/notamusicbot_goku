"""Migration tests using the real legacy input shape (29 OBAMA / 1 OBAMA_TRANS)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
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
    """Minimal clone profile set (ids obama/goku; OBAMA_TRANS must alias obama)."""
    voices = tmp_path / "voices"
    voices.mkdir()
    reference = tmp_path / "ref.wav"
    sf.write(reference, np.zeros(2400, dtype=np.float32), 24000, subtype="PCM_16")
    for pid in ("obama", "goku"):
        (voices / f"{pid}.yaml").write_text(
            yaml.safe_dump(
                {
                    "id": pid,
                    "persona": {"display_name": pid, "description": "test"},
                    "tts": {
                        "language": "English",
                        "reference_audio": str(reference),
                        "reference_text": "Hi.",
                    },
                    "audio": {"speed": 1.0, "target_peak_dbfs": -1.0},
                }
            ),
            encoding="utf-8",
        )
    return voices


# The live input/users.json proof moved to tests/evidence/
# test_users_migration_evidence.py (review Round-4 section 7).


def test_migration_maps_obama_and_obama_trans_to_obama(
    legacy_users_file: Path, tmp_path: Path, voices_dir: Path
) -> None:
    dest = tmp_path / "out" / "users.yaml"
    entries = migrate_users_to_profiles(legacy_users_file, dest, voices_dir)
    assert len(entries) == 30
    # OBAMA_TRANS aliases obama (longest profile-id prefix); no separate profile.
    assert sum(e["voice_profile"] == "obama" for e in entries.values()) == 30
    written = yaml.safe_load(dest.read_text(encoding="utf-8"))["users"]
    assert written["id28"]["voice_profile"] == "obama"
    assert written["id-trans"]["voice_profile"] == "obama"
    # The migrated file carries canonical keys only.
    assert set(written["id-trans"]) == {
        "screen_name",
        "real_name",
        "background",
        "voice_profile",
    }


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


def test_resolve_is_case_insensitive_and_prefix_aliases(voices_dir: Path) -> None:
    profiles = load_voice_profiles(voices_dir)
    assert resolve_legacy_profile("OBAMA", profiles) == "obama"
    assert resolve_legacy_profile("obama_trans", profiles) == "obama"
    assert resolve_legacy_profile("Goku", profiles) == "goku"
    assert resolve_legacy_profile("GOKU_RAW", profiles) == "goku"
    with pytest.raises(MigrationError):
        resolve_legacy_profile("NOPE", profiles)


def test_empty_source_raises(tmp_path: Path, voices_dir: Path) -> None:
    source = tmp_path / "users.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="No valid user entries"):
        migrate_users_to_profiles(source, tmp_path / "users.yaml", voices_dir)


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
