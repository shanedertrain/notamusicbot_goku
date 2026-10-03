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
