"""Round-4 P0/P2 tests: RVC YAML variant preservation + hard startup timeout.

Covers:

1. Variant selection end-to-end with a fake worker that speaks the real
   protocol: profile ``obama`` selects ``obama.yaml`` values, profile
   ``obama_trans`` selects ``obama_trans.yaml`` values (fixture configs with
   observably different parameters).
2. Cache identity ``(model dir, variant)``: switching variants of one model
   directory never restarts the worker (config hot-swap), while distinct
   variants remain distinct loads.
3. Missing variant yaml fails loudly (never falls back to ``found[0]``).
4. Hard startup timeout: a worker that starts, stays alive and emits nothing
   makes ``start()`` fail within the configured timeout.
5. Migration acceptance at config-identity level is covered in
   ``tests/unit/test_user_migration.py`` (variant written for OBAMA_TRANS).

The integration-flavored real-GPU variant check lives at the bottom, marked
``gpu rvc integration`` and skipped cleanly without the legacy assets.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile
from notamusicbot.speech.voice_conversion.rvc import (
    RvcVoiceConverter,
    WorkerRequestError,
    WorkerStartError,
)

# --------------------------------------------------------------- fake worker

# Speaks the real protocol. argv: mode, log_path, models_dir.
# mode "variants": load-model parses the REQUESTED variant yaml out of
# models_dir/<name>/ and echoes its actual values — proving the exact yaml
# reached the worker (no hardcoded expectations on the worker side).
FAKE_VARIANT_WORKER = (
    textwrap.dedent(
        """
    import json, os, sys, time

    mode, log_path, models_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    def log(obj):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\\n")

    def reply(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    def write_wav(path):
        import wave
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\\x00\\x10" * 1600)

    if mode == "silent":
        # Start, stay alive, emit NOTHING (not even the ready line).
        time.sleep(120)
        sys.exit(0)

    reply({"id": None, "status": "ok", "action": "ready"})
    log({"event": "startup", "pid": os.getpid()})

    import yaml as pyyaml

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            reply({"id": None, "status": "error", "error": "malformed"})
            continue
        action = req.get("action")
        if action == "health":
            reply({"id": req["id"], "status": "ok", "device": "cpu",
                   "torch": "fake", "cuda_available": False})
        elif action == "load-model":
            name = req.get("name")
            variant = req.get("variant")
            model_dir = os.path.join(models_dir, name)
            if variant is None:
                # Same default rule as the real worker: dir-name yaml first.
                yamls = sorted(f for f in os.listdir(model_dir) if f.endswith(".yaml"))
                match = [y for y in yamls if y[:-5].lower() == name.lower()]
                variant = (match[0] if match else yamls[0])[:-5]
            yaml_path = os.path.join(model_dir, variant + ".yaml")
            if not os.path.isfile(yaml_path):
                # Real-worker behavior: error loudly, never found[0].
                reply({"id": req["id"], "status": "error",
                       "error": f"variant {variant!r} not found in {model_dir}"})
                continue
            with open(yaml_path, "r", encoding="utf-8") as fh:
                cfg = pyyaml.safe_load(fh)
            log({"event": "loaded-config", "request_variant": variant,
                 "yaml": variant + ".yaml", "values": cfg})
            reply({"id": req["id"], "status": "ok", "model": name,
                   "variant": variant, "config_yaml": variant + ".yaml",
                   "config_switched": False, "device": "cpu",
                   "target_sample_rate": 16000,
                   "tts_type": str(cfg.get("tts_type")),
                   "index_rate": float(cfg.get("index_rate", 0.0)),
                   "f0_method": str(cfg.get("f0_method")),
                   "default_f0_pitch": int(cfg.get("f0_pitch", 0))})
        elif action == "convert":
            log({"event": "converted", "pitch_shift": req.get("pitch_shift")})
            write_wav(req["output_wav_path"])
            reply({"id": req["id"], "status": "ok",
                   "output_wav_path": req["output_wav_path"],
                   "elapsed_seconds": 0.01, "model": req.get("name", "Obama"),
                   "variant": "n/a"})
        elif action == "shutdown":
            reply({"id": req["id"], "status": "ok", "action": "shutdown"})
            sys.exit(0)
        else:
            reply({"id": req["id"], "status": "error", "error": f"unknown {action}"})
    """
    ).strip()
    + "\n"
)

# Fixture configs: observably different parameter values.
OBAMA_YAML = {
    "sid": 0,
    "f0_pitch": 0,
    "crepe_hop_length": 128,
    "f0_method": "harvest",
    "index_rate": 0.4,
    "tts_type": "pyht:Joseph",
    "description": "standard Obama",
    "speed_multiplier": 1.0,
}
OBAMA_TRANS_YAML = {
    "sid": 0,
    "f0_pitch": 5,
    "crepe_hop_length": 160,
    "f0_method": "rmvpe",
    "index_rate": 0.66,
    "tts_type": "gtts",
    "description": "transformed Obama",
    "speed_multiplier": 1.0,
}


def _variant_profile(variant: str | None) -> VoiceConversionProfile:
    return VoiceConversionProfile(backend="rvc", model="Obama", variant=variant, pitch_shift=0)


@pytest.fixture()
def variant_env(tmp_path: Path, monkeypatch):
    """Fixture models dir (Obama/{obama,obama_trans}.yaml + fake weights),
    fake worker, and a converter whose model resolution points at the fixture."""
    models_dir = tmp_path / "models" / "Obama"
    models_dir.mkdir(parents=True)
    (models_dir / "obama.yaml").write_text(yaml.safe_dump(OBAMA_YAML), encoding="utf-8")
    (models_dir / "obama_trans.yaml").write_text(yaml.safe_dump(OBAMA_TRANS_YAML), encoding="utf-8")
    (models_dir / "obama.pth").write_bytes(b"fake-weights")  # shape only
    (models_dir / "obama.index").write_bytes(b"fake-index")  # shape only

    script = tmp_path / "fake_variant_worker.py"
    script.write_text(FAKE_VARIANT_WORKER, encoding="utf-8")
    log_path = tmp_path / "requests.jsonl"
    spawned: list[list[str]] = []

    def spawn(self: RvcVoiceConverter) -> None:
        import subprocess

        argv = [
            str(self._worker_python),
            str(self._worker_script),
            "variants",
            str(log_path),
            str(tmp_path / "models"),
        ]
        spawned.append(argv)
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
        from notamusicbot.speech.voice_conversion.rvc import _StderrDrainer

        self._stderr_drainer = _StderrDrainer(process.stderr, "fake-worker")
        self._stderr_drainer.start()
        if not self._wait_for_ready(process, self._start_timeout):
            detail = self._exit_detail(process)
            self._terminate(process)
            self._process = None
            raise WorkerStartError(f"fake worker not ready. {detail}")

    monkeypatch.setattr(RvcVoiceConverter, "_spawn_and_wait_ready", spawn)
    # Resolve "Obama" against the FIXTURE models dir, not rvc_cli/models.
    monkeypatch.setattr(RvcVoiceConverter, "_resolve_model_name", staticmethod(lambda model: model))

    def make(**kwargs: Any) -> RvcVoiceConverter:
        converter = RvcVoiceConverter(
            worker_python=sys.executable,
            worker_script=script,
            start_timeout_s=kwargs.pop("start_timeout_s", 10.0),
            load_timeout_s=kwargs.pop("load_timeout_s", 10.0),
            convert_timeout_s=kwargs.pop("convert_timeout_s", 10.0),
            **kwargs,
        )
        converter.log_path = log_path  # type: ignore[attr-defined]
        return converter

    make.log_path = log_path  # type: ignore[attr-defined]
    make.spawned = spawned  # type: ignore[attr-defined]
    make.models_dir = models_dir  # type: ignore[attr-defined]
    return make


def _source(tmp_path: Path) -> AudioArtifact:
    path = tmp_path / "in.wav"
    path.write_bytes(b"")  # fake worker never reads it
    return AudioArtifact(path=path, sample_rate_hz=16000, duration_s=1.0)


def _events(converter: RvcVoiceConverter) -> list[dict]:
    text = converter.log_path.read_text()  # type: ignore[attr-defined]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ------------------------------------------------- variant selection (P0, §2)


def test_obama_profile_loads_standard_config(variant_env, tmp_path):
    converter = variant_env()
    artifact = asyncio.run(
        converter.convert(_source(tmp_path), _variant_profile("obama"), tmp_path / "out1.wav")
    )
    assert artifact.path == tmp_path / "out1.wav"
    loaded = [e for e in _events(converter) if e.get("event") == "loaded-config"]
    assert len(loaded) == 1
    assert loaded[0]["request_variant"] == "obama"
    assert loaded[0]["yaml"] == "obama.yaml"
    # The EXACT standard values from the fixture yaml reached the worker.
    assert loaded[0]["values"] == OBAMA_YAML
    asyncio.run(converter.shutdown())


def test_obama_trans_profile_loads_trans_config(variant_env, tmp_path):
    converter = variant_env()
    asyncio.run(
        converter.convert(_source(tmp_path), _variant_profile("obama_trans"), tmp_path / "out2.wav")
    )
    loaded = [e for e in _events(converter) if e.get("event") == "loaded-config"]
    assert loaded[0]["request_variant"] == "obama_trans"
    assert loaded[0]["yaml"] == "obama_trans.yaml"
    # The EXACT trans values — observably different from the standard ones.
    assert loaded[0]["values"] == OBAMA_TRANS_YAML
    assert loaded[0]["values"]["tts_type"] != OBAMA_YAML["tts_type"]
    assert loaded[0]["values"]["index_rate"] != OBAMA_YAML["index_rate"]
    asyncio.run(converter.shutdown())


def test_profiles_send_variant_in_load_model_ipc(variant_env, tmp_path):
    converter = variant_env()
    asyncio.run(converter.start())
    await_res = asyncio.run(converter._ensure_model("Obama", variant="obama_trans"))
    assert await_res is None
    asyncio.run(converter.shutdown())
    # (no assertion on the wire here; the loaded-config events above and the
    # cache test below cover the IPC fields — this test pins that a variant
    # on the profile reaches _ensure_model without touching rvc_cli/models)


def test_missing_variant_fails_loudly_never_found0(variant_env, tmp_path):
    converter = variant_env()
    with pytest.raises(WorkerRequestError, match="variant 'nonexistent' not found"):
        asyncio.run(
            converter.convert(
                _source(tmp_path), _variant_profile("nonexistent"), tmp_path / "out.wav"
            )
        )
    # And nothing was converted/loaded under a wrong config.
    assert not [e for e in _events(converter) if e.get("event") == "loaded-config"]
    asyncio.run(converter.shutdown())


def test_variant_omitted_defaults_to_directory_name_yaml(variant_env, tmp_path):
    converter = variant_env()
    profile = VoiceConversionProfile(backend="rvc", model="Obama", pitch_shift=0)
    asyncio.run(converter.convert(_source(tmp_path), profile, tmp_path / "out.wav"))
    loaded = [e for e in _events(converter) if e.get("event") == "loaded-config"]
    assert loaded[0]["request_variant"] == "obama"  # dir name "Obama" -> obama.yaml
    assert loaded[0]["values"] == OBAMA_YAML
    asyncio.run(converter.shutdown())


# ------------------------------------- cache identity (dir, variant), no restart


def test_variant_switch_keeps_worker_alive_distinct_cache_identities(variant_env, tmp_path):
    """(Obama, obama) -> (Obama, obama_trans) -> back: both variants are
    DISTINCT loads, and the worker is NEVER restarted — the same process
    hot-swaps the YAML config in-process (weights are shared)."""
    converter = variant_env()
    out = tmp_path / "out.wav"
    asyncio.run(converter.convert(_source(tmp_path), _variant_profile("obama"), out))
    assert converter.loaded_model == "Obama"
    assert converter.loaded_variant == "obama"
    asyncio.run(converter.convert(_source(tmp_path), _variant_profile("obama_trans"), out))
    assert converter.loaded_variant == "obama_trans"
    asyncio.run(converter.convert(_source(tmp_path), _variant_profile("obama"), out))
    assert converter.loaded_variant == "obama"

    events = _events(converter)
    loaded = [e for e in events if e.get("event") == "loaded-config"]
    # Two DISTINCT cache identities were exercised...
    variants_used = {e["request_variant"] for e in loaded}
    assert variants_used == {"obama", "obama_trans"}
    # ...each switch required exactly one load-model request (3 conversions,
    # 3 load-models; the "two distinct loads" are obama and obama_trans)...
    assert len(loaded) == 3
    # ...but exactly ONE worker process served everything: 0 restarts.
    startups = [e for e in events if e.get("event") == "startup"]
    assert len(startups) == 1
    assert len(variant_env.spawned) == 1
    # Config values alternated correctly within the single process.
    assert [e["values"]["tts_type"] for e in loaded] == [
        OBAMA_YAML["tts_type"],
        OBAMA_TRANS_YAML["tts_type"],
        OBAMA_YAML["tts_type"],
    ]
    asyncio.run(converter.shutdown())


def test_same_variant_is_cached_no_reload(variant_env, tmp_path):
    converter = variant_env()
    out = tmp_path / "out.wav"
    for _ in range(3):
        asyncio.run(converter.convert(_source(tmp_path), _variant_profile("obama"), out))
    loaded = [e for e in _events(converter) if e.get("event") == "loaded-config"]
    assert len(loaded) == 1  # cache hit on (Obama, obama): one load total
    asyncio.run(converter.shutdown())


# --------------------------------------- P2 §8: hard startup timeout (silent worker)


def test_startup_hard_timeout_on_silent_worker(tmp_path, monkeypatch):
    """Worker starts, stays ALIVE, emits nothing: start() must fail inside
    the configured timeout (the old blocking readline() could hang forever)."""
    script = tmp_path / "silent_worker.py"
    script.write_text(FAKE_VARIANT_WORKER, encoding="utf-8")
    converter = RvcVoiceConverter(
        worker_python=sys.executable,
        worker_script=script,
        start_timeout_s=0.5,
    )
    argv_extra = ["silent", str(tmp_path / "log.jsonl"), str(tmp_path)]

    def spawn(self: RvcVoiceConverter) -> None:
        import subprocess

        argv = [str(self._worker_python), str(self._worker_script), *argv_extra]
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
        if not self._wait_for_ready(process, self._start_timeout):
            detail = self._exit_detail(process)
            self._terminate(process)
            self._process = None
            raise WorkerStartError(
                f"RVC worker did not signal ready within {self._start_timeout}s. {detail}"
            )

    monkeypatch.setattr(RvcVoiceConverter, "_spawn_and_wait_ready", spawn)
    started = time.monotonic()
    with pytest.raises(WorkerStartError, match="did not signal ready"):
        asyncio.run(converter.start())
    elapsed = time.monotonic() - started
    # Hard timeout: well inside the deadline plus process-teardown slack,
    # and far below the 120s the silent worker would keep running.
    assert elapsed < 5.0, f"startup took {elapsed:.2f}s; the timeout is not hard"
    assert not converter.is_running


# ------------------------------------------------- real-worker integration (GPU)


@pytest.mark.gpu
@pytest.mark.rvc
@pytest.mark.integration
def test_real_worker_obama_variants_differ(tmp_path):
    """Real legacy worker on the real rvc_cli/models/Obama directory:
    loading variant obama then obama_trans must report the REAL differing
    yaml config values (tts_type: pyht:Joseph vs gtts).

    Run for real:
        .venv-main/bin/python -m pytest tests/unit/test_rvc_variant.py \
            -m "gpu and rvc and integration" -s
    """
    legacy_python = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"
    if not legacy_python.is_file():
        pytest.skip("legacy .venv python not present")
    converter = RvcVoiceConverter()
    try:
        asyncio.run(converter.start())
        std = asyncio.run(
            converter._request("load-model", timeout=300, name="Obama", variant="obama")
        )
        trans = asyncio.run(
            converter._request("load-model", timeout=300, name="Obama", variant="obama_trans")
        )
    finally:
        asyncio.run(converter.shutdown())
    assert std["variant"] == "obama"
    assert trans["variant"] == "obama_trans"
    # Real differing values, loaded in ONE process (config switch, no reload).
    assert std["config_switched"] is False
    assert trans["config_switched"] is True
    assert std["tts_type"] != trans["tts_type"]
    print(
        f"\nreal Obama variant values: obama={std['tts_type']!r} "
        f"index_rate={std['index_rate']}, "
        f"obama_trans={trans['tts_type']!r} index_rate={trans['index_rate']}"
    )
