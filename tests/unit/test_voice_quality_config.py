"""RVC quality settings: config validation, pipeline merge, wire fields."""

from __future__ import annotations

import asyncio
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from notamusicbot.config import VoiceConversionSettings
from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile
from notamusicbot.speech.pipeline import SpeechPipeline
from notamusicbot.speech.voice_conversion.rvc import RvcVoiceConverter

# ------------------------------------------------------------------ config


def test_defaults_are_old_behaviour() -> None:
    s = VoiceConversionSettings()
    assert (s.use_index, s.index_rate, s.index_nprobe, s.protect, s.rms_mix_rate) == (
        False,
        None,
        None,
        None,
        None,
    )
    assert (s.pitch_shift_offset, s.auto_pitch, s.auto_pitch_strength, s.auto_pitch_max) == (
        0,
        False,
        1.0,
        12,
    )
    assert s.target_f0_median_hz == {}
    assert (s.f0_post_median, s.f0_post_octave_repair, s.f0_post_clamp) == (0, False, None)
    assert s.stream_f0_algo == "pm"
    assert s.f0_method_override is None


def test_rmvpe_accepted_in_override() -> None:
    assert VoiceConversionSettings(f0_method_override="rmvpe").f0_method_override == "rmvpe"
    with pytest.raises(ValidationError):
        VoiceConversionSettings(f0_method_override="nope")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"f0_post_median": 4},
        {"f0_post_median": 1},
        {"f0_post_median": -3},
        {"protect": 0.9},
        {"index_rate": 1.5},
        {"index_nprobe": 0},
        {"rms_mix_rate": 2.0},
        {"pitch_shift_offset": 30},
        {"auto_pitch_strength": 1.2},
        {"auto_pitch_max": 25},
        {"f0_post_clamp": (500.0, 50.0)},
        {"f0_post_clamp": (100.0, 100.0)},
        {"f0_post_clamp": (0.0, 100.0)},
        {"target_f0_median_hz": {"Obama": 20.0}},
        {"target_f0_median_hz": {"Obama": 900.0}},
        {"stream_f0_algo": "harvest"},
    ],
)
def test_validators_reject(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        VoiceConversionSettings(**kwargs)


def test_valid_values_accepted() -> None:
    s = VoiceConversionSettings(
        f0_post_median=5,
        protect=0.33,
        f0_post_clamp=(50, 500),
        target_f0_median_hz={"Obama": 108},
        stream_f0_algo="rmvpe",
    )
    assert s.f0_post_clamp == (50.0, 500.0)
    assert s.f0_post_median == 5


# ---------------------------------------------------------------- pipeline


def make_pipeline(**kwargs: Any) -> SpeechPipeline:
    quality = kwargs.pop("quality", None)
    if isinstance(quality, dict):
        quality = VoiceConversionSettings(**quality)
    return SpeechPipeline(None, None, quality=quality, **kwargs)  # type: ignore[arg-type]


PROFILE = VoiceConversionProfile(backend="rvc", model="Obama", variant="obama", pitch_shift=2)


def test_all_defaults_profile_unchanged() -> None:
    for pipe in (make_pipeline(), make_pipeline(quality=VoiceConversionSettings())):
        assert pipe._apply_f0_override(PROFILE) == PROFILE  # noqa: SLF001
    pipe = make_pipeline(f0_method_override="pm", quality=VoiceConversionSettings())
    assert pipe._apply_f0_override(PROFILE) == replace(PROFILE, f0_method="pm")  # noqa: SLF001


def test_non_rvc_backend_untouched() -> None:
    passthrough = VoiceConversionProfile(backend="passthrough")
    pipe = make_pipeline(quality={"pitch_shift_offset": 3, "use_index": True})
    assert pipe._apply_f0_override(passthrough) == passthrough  # noqa: SLF001


def test_override_merge_arithmetic_and_copy() -> None:
    pipe = make_pipeline(
        f0_method_override="rmvpe",
        quality={
            "pitch_shift_offset": -5,
            "use_index": True,
            "index_rate": 0.5,
            "index_nprobe": 16,
            "protect": 0.33,
            "rms_mix_rate": 0.25,
            "f0_post_median": 5,
            "f0_post_octave_repair": True,
            "f0_post_clamp": (50, 500),
            "stream_f0_algo": "rmvpe",
        },
    )
    out = pipe._apply_f0_override(PROFILE)  # noqa: SLF001
    assert out.pitch_shift == -3
    assert out.f0_method == "rmvpe"
    assert (out.use_index, out.index_rate, out.index_nprobe) == (True, 0.5, 16)
    assert (out.protect, out.rms_mix_rate) == (0.33, 0.25)
    assert dict(out.f0_post or ()) == {
        "median_width": 5,
        "octave_repair": True,
        "clamp_min_hz": 50.0,
        "clamp_max_hz": 500.0,
    }
    assert out.stream_f0_algo == "rmvpe"
    assert out.auto_pitch_target_hz is None
    hash(out)  # stays hashable


def test_auto_pitch_target_case_insensitive_and_gated() -> None:
    quality = {
        "auto_pitch": True,
        "auto_pitch_strength": 0.5,
        "auto_pitch_max": 7,
        "target_f0_median_hz": {"obama": 108.0},
    }
    out = make_pipeline(quality=quality)._apply_f0_override(PROFILE)  # noqa: SLF001
    assert (out.auto_pitch_target_hz, out.auto_pitch_strength, out.auto_pitch_max) == (
        108.0,
        0.5,
        7,
    )
    other = replace(PROFILE, model="Goku")
    assert make_pipeline(quality=quality)._apply_f0_override(other) == other  # noqa: SLF001
    off = dict(quality, auto_pitch=False)
    assert make_pipeline(quality=off)._apply_f0_override(PROFILE) == PROFILE  # noqa: SLF001


def test_profile_defaults() -> None:
    p = VoiceConversionProfile(backend="rvc")
    defaults = {
        f.name: f.default
        for f in fields(VoiceConversionProfile)
        if f.name
        in {
            "use_index",
            "index_rate",
            "index_nprobe",
            "protect",
            "rms_mix_rate",
            "auto_pitch_target_hz",
            "auto_pitch_strength",
            "auto_pitch_max",
            "f0_post",
            "stream_f0_algo",
        }
    }
    assert defaults == {
        "use_index": False,
        "index_rate": None,
        "index_nprobe": None,
        "protect": None,
        "rms_mix_rate": None,
        "auto_pitch_target_hz": None,
        "auto_pitch_strength": 1.0,
        "auto_pitch_max": 12,
        "f0_post": None,
        "stream_f0_algo": "pm",
    }
    assert hash(p) == hash(VoiceConversionProfile(backend="rvc"))


# -------------------------------------------------------------- wire (convert)


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []


def make_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[RvcVoiceConverter, Recorder]:
    conv = RvcVoiceConverter(worker_python="python", worker_script=tmp_path / "w.py")
    rec = Recorder()
    out = tmp_path / "out.wav"
    out.write_bytes(b"x")

    async def start() -> None:
        return None

    async def ensure(*_a: Any, **_k: Any) -> None:
        return None

    async def request(action: str, timeout: float | None = None, **fields: Any) -> dict[str, Any]:
        rec.calls.append((action, fields))
        return {"output_wav_path": str(out)}

    monkeypatch.setattr(RvcVoiceConverter, "is_running", property(lambda _s: True))
    monkeypatch.setattr(conv, "start", start)
    monkeypatch.setattr(conv, "_ensure_model", ensure)
    monkeypatch.setattr(conv, "_request", request)
    monkeypatch.setattr(conv, "_probe_output", lambda p: AudioArtifact(p, 16000, 1.0))
    return conv, rec


def run_convert(conv: RvcVoiceConverter, profile: VoiceConversionProfile, tmp_path: Path) -> None:
    src = AudioArtifact(path=tmp_path / "in.wav", sample_rate_hz=24000, duration_s=1.0)
    asyncio.run(conv.convert(src, profile, tmp_path / "out.wav"))


def test_convert_default_request_unchanged(monkeypatch, tmp_path) -> None:
    conv, rec = make_adapter(monkeypatch, tmp_path)
    run_convert(conv, PROFILE, tmp_path)
    ((action, fields_),) = rec.calls
    assert action == "convert"
    assert set(fields_) == {"input_wav_path", "output_wav_path", "pitch_shift"}
    assert fields_["pitch_shift"] == 2
    # explicit f0 method is the only pre-existing optional field
    run_convert(conv, replace(PROFILE, f0_method="pm"), tmp_path)
    assert set(rec.calls[1][1]) == {"input_wav_path", "output_wav_path", "pitch_shift", "f0_method"}


def test_convert_forwards_new_fields(monkeypatch, tmp_path) -> None:
    conv, rec = make_adapter(monkeypatch, tmp_path)
    profile = replace(
        PROFILE,
        f0_method="rmvpe",
        use_index=True,
        index_rate=0.5,
        index_nprobe=16,
        protect=0.33,
        rms_mix_rate=0.25,
        auto_pitch_target_hz=108.0,
        auto_pitch_strength=0.5,
        auto_pitch_max=9,
        f0_post=tuple(
            sorted(
                {
                    "median_width": 5,
                    "octave_repair": True,
                    "clamp_min_hz": 50.0,
                    "clamp_max_hz": 500.0,
                }.items()
            )
        ),
    )
    run_convert(conv, profile, tmp_path)
    f = rec.calls[0][1]
    assert f["f0_method"] == "rmvpe"
    assert f["use_index"] is True and f["index_rate"] == 0.5 and f["index_nprobe"] == 16
    assert f["protect"] == 0.33 and f["rms_mix_rate"] == 0.25
    assert f["auto_pitch"] == {"target_median_hz": 108.0, "max_abs": 9, "strength": 0.5}
    assert f["f0_post"] == {
        "median_width": 5,
        "octave_repair": True,
        "clamp_min_hz": 50.0,
        "clamp_max_hz": 500.0,
    }
    json.dumps(f)  # wire-serialisable


def test_index_rate_ignored_without_use_index(monkeypatch, tmp_path) -> None:
    conv, rec = make_adapter(monkeypatch, tmp_path)
    run_convert(conv, replace(PROFILE, index_rate=0.5, index_nprobe=8), tmp_path)
    assert "use_index" not in rec.calls[0][1]
    assert "index_rate" not in rec.calls[0][1]


# --------------------------------------------------------------- wire (stream)


def _stream_open_fields(monkeypatch, tmp_path, profile) -> dict[str, Any]:
    conv, rec = make_adapter(monkeypatch, tmp_path)
    from notamusicbot.speech.voice_conversion import rvc as rvc_mod

    conv._stream_params = rvc_mod.VoiceStreamParams(enabled=True)  # noqa: SLF001

    async def chunks():
        if False:
            yield None

    async def go() -> None:
        try:
            async for _ in conv.convert_stream(chunks(), profile):
                pass
        except Exception:  # noqa: BLE001 - worker replies are stubbed
            pass

    asyncio.run(go())
    return next(f for a, f in rec.calls if a == "stream_open")


def test_stream_open_default_unchanged(monkeypatch, tmp_path) -> None:
    f = _stream_open_fields(monkeypatch, tmp_path, PROFILE)
    assert set(f) == {
        "stream_id",
        "pitch_shift",
        "f0_method",
        "first_block_ms",
        "block_ms",
        "context_ms",
        "lookahead_ms",
        "crossfade_ms",
        "search_ms",
    }
    assert f["f0_method"] == "pm" and f["pitch_shift"] == 2


def test_stream_open_forwards_new_fields(monkeypatch, tmp_path) -> None:
    profile = replace(
        PROFILE,
        f0_method="harvest",
        use_index=True,
        index_rate=0.4,
        index_nprobe=8,
        protect=0.4,
        rms_mix_rate=0.3,  # not forwarded to streams
        auto_pitch_target_hz=108.0,  # not forwarded to streams
        f0_post=(("median_width", 5), ("octave_repair", False)),
        stream_f0_algo="rmvpe",
    )
    f = _stream_open_fields(monkeypatch, tmp_path, profile)
    assert f["f0_method"] == "pm" and f["f0_algo"] == "rmvpe"
    assert f["use_index"] is True and f["index_rate"] == 0.4 and f["index_nprobe"] == 8
    assert f["protect"] == 0.4
    assert f["f0_post"] == {"median_width": 5, "octave_repair": False}
    assert "rms_mix_rate" not in f and "auto_pitch" not in f
