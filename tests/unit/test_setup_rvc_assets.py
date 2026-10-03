"""Review Round-3 P2: setup.sh RVC asset validation.

Runs the ``validate_rvc_assets`` function from ``scripts/setup.sh`` directly
(extracted into a subshell) against a temp-dir fixture — no real models.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SETUP_SH = Path(__file__).resolve().parents[2] / "scripts" / "setup.sh"

PROFILE = """\
id: testvoice
persona:
  display_name: Test
tts:
  backend: qwen_custom_voice
  speaker: Aiden
voice_conversion:
  backend: rvc
  model: TestVoice
  pitch_shift: 0
audio:
  speed: 1.0
"""

PASSTHROUGH_PROFILE = PROFILE.replace("backend: rvc", "backend: passthrough")


def run_validation(voices: Path, models: Path) -> tuple[int, str]:
    """Extract validate_rvc_assets() from setup.sh and run it in a subshell."""
    cmd = (
        f'source /dev/stdin <<<"$(sed -n \'/^validate_rvc_assets()/,/^}}/p\' "{SETUP_SH}")"\n'
        f'validate_rvc_assets "{voices}" "{models}"'
    )
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def _write_profile(voices: Path, text: str = PROFILE, name: str = "testvoice.yaml") -> None:
    voices.mkdir(parents=True, exist_ok=True)
    (voices / name).write_text(text, encoding="utf-8")


def _make_complete(voices: Path, models: Path) -> None:
    _write_profile(voices)
    mdir = models / "TestVoice"
    mdir.mkdir(parents=True)
    (mdir / "testvoice.yaml").write_text("variant: base\n", encoding="utf-8")
    (mdir / "model.pth").write_bytes(b"x")
    (mdir / "model.index").write_bytes(b"x")


def test_complete_assets_pass(tmp_path: Path) -> None:
    voices, models = tmp_path / "voices", tmp_path / "models"
    _make_complete(voices, models)
    code, out = run_validation(voices, models)
    assert code == 0, out


def test_missing_model_dir_fails(tmp_path: Path) -> None:
    voices, models = tmp_path / "voices", tmp_path / "models"
    _write_profile(voices)
    code, out = run_validation(voices, models)
    assert code != 0
    assert "model directory missing" in out


def test_missing_variant_yaml_fails(tmp_path: Path) -> None:
    voices, models = tmp_path / "voices", tmp_path / "models"
    _write_profile(voices)
    mdir = models / "TestVoice"
    mdir.mkdir(parents=True)
    (mdir / "model.pth").write_bytes(b"x")
    (mdir / "model.index").write_bytes(b"x")
    code, out = run_validation(voices, models)
    assert code != 0
    assert "variant yaml missing" in out


def test_missing_pth_and_index_fail(tmp_path: Path) -> None:
    voices, models = tmp_path / "voices", tmp_path / "models"
    _write_profile(voices)
    mdir = models / "TestVoice"
    mdir.mkdir(parents=True)
    (mdir / "testvoice.yaml").write_text("variant: base\n", encoding="utf-8")
    code, out = run_validation(voices, models)
    assert code != 0
    assert ".pth" in out and ".index" in out


def test_passthrough_profiles_are_not_validated(tmp_path: Path) -> None:
    voices, models = tmp_path / "voices", tmp_path / "models"
    _write_profile(voices, PASSTHROUGH_PROFILE, "plain.yaml")
    code, out = run_validation(voices, models)
    assert code == 0, out


# ----------------------- review Round-4 P2: explicit voice_conversion.variant --


ALT_PROFILE = PROFILE.replace(
    "voice_conversion:",
    "voice_conversion:\n  variant: obama_trans",
)


def test_explicit_variant_yaml_is_checked_not_profile_stem(tmp_path: Path) -> None:
    """Profile file president_alt.yaml with variant obama_trans must make
    setup validate obama_trans.yaml — NOT president_alt.yaml."""
    voices, models = tmp_path / "voices", tmp_path / "models"
    _write_profile(voices, ALT_PROFILE, "president_alt.yaml")
    mdir = models / "TestVoice"
    mdir.mkdir(parents=True)
    (mdir / "model.pth").write_bytes(b"x")
    (mdir / "model.index").write_bytes(b"x")
    # Only the variant yaml exists; president_alt.yaml deliberately absent.
    (mdir / "obama_trans.yaml").write_text("tts_type: trans\n", encoding="utf-8")
    code, out = run_validation(voices, models)
    assert code == 0, out


def test_explicit_variant_missing_fails_even_if_stem_yaml_exists(tmp_path: Path) -> None:
    """The inverse: president_alt.yaml exists in the model dir but the
    configured variant obama_trans.yaml does not — setup must still fail."""
    voices, models = tmp_path / "voices", tmp_path / "models"
    _write_profile(voices, ALT_PROFILE, "president_alt.yaml")
    mdir = models / "TestVoice"
    mdir.mkdir(parents=True)
    (mdir / "model.pth").write_bytes(b"x")
    (mdir / "model.index").write_bytes(b"x")
    (mdir / "president_alt.yaml").write_text("tts_type: base\n", encoding="utf-8")
    code, out = run_validation(voices, models)
    assert code != 0
    assert "variant yaml missing" in out
    assert "obama_trans.yaml" in out


def test_variant_absent_falls_back_to_directory_default_yaml(tmp_path: Path) -> None:
    """No voice_conversion.variant -> validate the directory-default yaml:
    stem matching the model dir name, else the single yaml present."""
    voices, models = tmp_path / "voices", tmp_path / "models"
    _write_profile(voices)
    mdir = models / "TestVoice"
    mdir.mkdir(parents=True)
    (mdir / "model.pth").write_bytes(b"x")
    (mdir / "model.index").write_bytes(b"x")
    # Exactly one yaml, stem differing from the dir name: still the default.
    (mdir / "other.yaml").write_text("tts_type: base\n", encoding="utf-8")
    code, out = run_validation(voices, models)
    assert code == 0, out


def test_variant_absent_multiple_yamls_no_default_fails(tmp_path: Path) -> None:
    """No explicit variant, several yamls, none matching the dir name:
    there is no unambiguous directory default -> setup must fail."""
    voices, models = tmp_path / "voices", tmp_path / "models"
    _write_profile(voices)
    mdir = models / "TestVoice"
    mdir.mkdir(parents=True)
    (mdir / "model.pth").write_bytes(b"x")
    (mdir / "model.index").write_bytes(b"x")
    (mdir / "alpha.yaml").write_text("tts_type: a\n", encoding="utf-8")
    (mdir / "beta.yaml").write_text("tts_type: b\n", encoding="utf-8")
    code, out = run_validation(voices, models)
    assert code != 0
    assert "variant yaml missing" in out
