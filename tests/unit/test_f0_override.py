"""Unit tests for the config-selectable RVC f0_method override plumbing.

Covers: settings loading/validation, the pipeline-level override application,
and the adapter→worker IPC field via the fake worker.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from conftest import FakeConverter, FakeSynthesizer
from notamusicbot.config import AppSettings
from notamusicbot.domain.speech import (
    AudioProcessingProfile,
    TtsRequest,
    VoiceConversionProfile,
)
from notamusicbot.speech.pipeline import SpeechPipeline

# ------------------------------------------------------------------ settings


def test_settings_default_keeps_profile_yaml_f0_method() -> None:
    settings = AppSettings()
    assert settings.speech.voice_conversion.f0_method_override is None


def test_settings_accepts_pm_override() -> None:
    settings = AppSettings.model_validate(
        {"speech": {"voice_conversion": {"f0_method_override": "pm"}}}
    )
    assert settings.speech.voice_conversion.f0_method_override == "pm"


def test_settings_rejects_unknown_f0_method() -> None:
    with pytest.raises(ValueError):
        AppSettings.model_validate(
            {"speech": {"voice_conversion": {"f0_method_override": "swift"}}}
        )


# ------------------------------------------------------------------ pipeline


def _run(pipeline: SpeechPipeline, conversion: VoiceConversionProfile, tmp_path: Path):
    return asyncio.run(
        pipeline.run(
            TtsRequest(text="hi", language="English", speaker="Aiden"),
            conversion,
            AudioProcessingProfile(),
        )
    )


def test_pipeline_applies_f0_override_to_rvc_profile(tmp_path: Path) -> None:
    pipeline = SpeechPipeline(
        FakeSynthesizer(),
        FakeConverter(),
        output_dir=tmp_path / "speech",
        f0_method_override="pm",
    )
    conversion = VoiceConversionProfile(backend="rvc", model="Goku")
    _run(pipeline, conversion, tmp_path)
    converter = pipeline._converter
    assert isinstance(converter, FakeConverter)
    assert converter.calls[0][1].f0_method == "pm"


def test_pipeline_override_leaves_passthrough_untouched(tmp_path: Path) -> None:
    pipeline = SpeechPipeline(
        FakeSynthesizer(),
        FakeConverter(),
        output_dir=tmp_path / "speech",
        f0_method_override="pm",
    )
    conversion = VoiceConversionProfile(backend="passthrough")
    _run(pipeline, conversion, tmp_path)
    converter = pipeline._converter
    assert isinstance(converter, FakeConverter)
    assert converter.calls[0][1].f0_method is None


def test_pipeline_without_override_keeps_profile_value(tmp_path: Path) -> None:
    pipeline = SpeechPipeline(
        FakeSynthesizer(),
        FakeConverter(),
        output_dir=tmp_path / "speech",
    )
    conversion = VoiceConversionProfile(backend="rvc", model="Goku", f0_method="harvest")
    _run(pipeline, conversion, tmp_path)
    converter = pipeline._converter
    assert isinstance(converter, FakeConverter)
    assert converter.calls[0][1].f0_method == "harvest"


# ------------------------------------------------------- adapter → worker IPC


def _load_fake_worker_module():
    """Load tests/unit/test_rvc_adapter.py (no package __init__) by path."""
    import importlib.util

    path = Path(__file__).parent / "test_rvc_adapter.py"
    spec = importlib.util.spec_from_file_location("test_rvc_adapter_helper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def rvc_helper():
    return _load_fake_worker_module()


def _run_fake_convert(tmp_path: Path, helper: Any, f0_method: str | None) -> list[dict[str, Any]]:
    """Drive one convert through the fake worker; return its logged requests."""
    import subprocess
    import sys

    from notamusicbot.speech.voice_conversion.rvc import RvcVoiceConverter, _StderrDrainer

    script = tmp_path / "fake_worker.py"
    script.write_text(helper.FAKE_WORKER, encoding="utf-8")
    log_path = tmp_path / "requests.jsonl"

    converter = RvcVoiceConverter(
        worker_python=sys.executable,
        worker_script=script,
        start_timeout_s=10.0,
        load_timeout_s=10.0,
        convert_timeout_s=10.0,
    )
    converter._argv_extra = ("ok", str(log_path), "")  # type: ignore[attr-defined]

    original = RvcVoiceConverter._spawn_and_wait_ready

    def spawn(self: RvcVoiceConverter) -> None:
        argv = [str(self._worker_python), str(self._worker_script), "ok", str(log_path), ""]
        process = subprocess.Popen(
            argv,
            cwd=str(script.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._process = process
        self._stderr_drainer = _StderrDrainer(process.stderr, "fake-worker")
        self._stderr_drainer.start()
        assert self._wait_for_ready(process, self._start_timeout)

    RvcVoiceConverter._spawn_and_wait_ready = spawn  # type: ignore[method-assign]
    try:
        asyncio.run(converter.start())
        source_wav = tmp_path / "src.wav"
        source_wav.write_bytes(b"fake")
        profile = VoiceConversionProfile(backend="rvc", model="Fake", f0_method=f0_method)
        out = tmp_path / "out.wav"
        asyncio.run(converter.convert(helper.make_source(source_wav), profile, out))
        assert out.is_file()
    finally:
        RvcVoiceConverter._spawn_and_wait_ready = original  # type: ignore[method-assign]
        try:
            asyncio.run(converter.shutdown())
        except Exception:
            pass

    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("action") == "convert"
    ]


def test_adapter_sends_f0_method_override_in_convert_request(tmp_path, rvc_helper) -> None:
    """The fake worker records requests; the convert request carries f0_method."""
    requests = _run_fake_convert(tmp_path, rvc_helper, f0_method="pm")
    assert requests, "fake worker recorded no convert request"
    assert requests[0]["f0_method"] == "pm"


def test_profile_without_f0_method_omits_ipc_field(tmp_path, rvc_helper) -> None:
    """No override ⇒ the request carries no f0_method (yaml default wins)."""
    requests = _run_fake_convert(tmp_path, rvc_helper, f0_method=None)
    assert requests and "f0_method" not in requests[0]
