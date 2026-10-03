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
    """Minimal profile set mirroring config/voices (ids obama/obama_trans).

    The obama_trans profile carries ``variant: obama_trans``: it shares the
    Obama weights but must select the ``obama_trans.yaml`` config.
    """
    voices = tmp_path / "voices"
    voices.mkdir()
    for pid, model, variant in (
        ("obama", "Obama", "obama"),
        ("obama_trans", "Obama", "obama_trans"),
        ("goku", "Goku", None),
    ):
        vc: dict = {"backend": "rvc", "model": model}
        if variant:
            vc["variant"] = variant
        (voices / f"{pid}.yaml").write_text(
            yaml.safe_dump(
                {
                    "id": pid,
                    "persona": {"display_name": pid, "description": "test"},
                    "tts": {"speaker": "Ryan", "language": "English"},
                    "voice_conversion": vc,
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


# ----------------------- round-4 P0: migration preserves CONFIG identity ------


def test_migration_writes_variant_for_trans_alias(
    legacy_users_file: Path, tmp_path: Path, voices_dir: Path
) -> None:
    dest = tmp_path / "out" / "users.yaml"
    entries = migrate_users_to_profiles(legacy_users_file, dest, voices_dir)
    trans = [e for e in entries.values() if e["voice_profile"] == "obama_trans"]
    obama = [e for e in entries.values() if e["voice_profile"] == "obama"]
    assert trans[0]["variant"] == "obama_trans"
    assert all(e["variant"] == "obama" for e in obama)
    written = yaml.safe_load(dest.read_text(encoding="utf-8"))["users"]
    assert written["id-trans"]["variant"] == "obama_trans"


def test_migrated_obama_trans_resolves_trans_config_values(
    legacy_users_file: Path, tmp_path: Path, voices_dir: Path, monkeypatch
) -> None:
    """Config-identity acceptance: the migrated obama_trans user's profile,
    driven through the fake worker speaking the real protocol, must select
    the trans yaml's actual values — not whichever yaml the worker finds first."""
    import asyncio
    import sys

    from notamusicbot.domain.speech import AudioArtifact
    from notamusicbot.domain.voice_profiles import to_conversion_profile
    from notamusicbot.speech.voice_conversion.rvc import RvcVoiceConverter

    test_rvc_variant = pytest.importorskip("test_rvc_variant")

    dest = tmp_path / "out" / "users.yaml"
    migrate_users_to_profiles(legacy_users_file, dest, voices_dir)
    migrated = yaml.safe_load(dest.read_text(encoding="utf-8"))["users"]
    assert migrated["id-trans"]["variant"] == "obama_trans"

    # Fixture model directory: two yamls with observably different values.
    models_dir = tmp_path / "models" / "Obama"
    models_dir.mkdir(parents=True)
    (models_dir / "obama.yaml").write_text(
        yaml.safe_dump(test_rvc_variant.OBAMA_YAML), encoding="utf-8"
    )
    (models_dir / "obama_trans.yaml").write_text(
        yaml.safe_dump(test_rvc_variant.OBAMA_TRANS_YAML), encoding="utf-8"
    )
    (models_dir / "obama.pth").write_bytes(b"weights")
    (models_dir / "obama.index").write_bytes(b"index")

    script = tmp_path / "fake_worker.py"
    script.write_text(test_rvc_variant.FAKE_VARIANT_WORKER, encoding="utf-8")
    log_path = tmp_path / "requests.jsonl"

    def spawn(self, *args, **kwargs):  # noqa: ANN001
        import subprocess

        process = subprocess.Popen(
            [
                sys.executable,
                str(script),
                "variants",
                str(log_path),
                str(tmp_path / "models"),
            ],
            cwd=str(tmp_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._process = process
        from notamusicbot.speech.voice_conversion.rvc import _StderrDrainer

        self._stderr_drainer = _StderrDrainer(process.stderr, "fake-worker")
        self._stderr_drainer.start()
        if not self._wait_for_ready(process, self._start_timeout):
            self._terminate(process)
            self._process = None
            raise RuntimeError("fake worker not ready")

    monkeypatch.setattr(RvcVoiceConverter, "_spawn_and_wait_ready", spawn)
    monkeypatch.setattr(RvcVoiceConverter, "_resolve_model_name", staticmethod(lambda model: model))

    profile = load_voice_profiles(voices_dir)["obama_trans"]
    conversion = to_conversion_profile(profile)
    assert conversion.variant == "obama_trans"

    src = tmp_path / "in.wav"
    src.write_bytes(b"")
    converter = RvcVoiceConverter(
        worker_python=sys.executable,
        worker_script=script,
        start_timeout_s=10.0,
        load_timeout_s=10.0,
        convert_timeout_s=10.0,
    )
    converter.log_path = log_path  # type: ignore[attr-defined]
    out = tmp_path / "out.wav"
    try:
        asyncio.run(
            converter.convert(
                AudioArtifact(path=src, sample_rate_hz=16000, duration_s=1.0),
                conversion,
                out,
            )
        )
        loaded = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and "loaded-config" in line
        ]
    finally:
        asyncio.run(converter.shutdown())
    assert loaded[0]["request_variant"] == "obama_trans"
    assert loaded[0]["values"] == test_rvc_variant.OBAMA_TRANS_YAML
    assert loaded[0]["values"]["tts_type"] != test_rvc_variant.OBAMA_YAML["tts_type"]
