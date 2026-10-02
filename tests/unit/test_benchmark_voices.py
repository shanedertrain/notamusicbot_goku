"""Unit tests for the voice benchmark CLI helpers (no GPU, no worker)."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import pytest
import soundfile as sf

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile
from notamusicbot.tools.benchmark_voices import (
    BenchmarkRow,
    load_corpus,
    peak_dbfs,
    run_benchmark,
    write_metadata,
    write_results_csv,
)


class ScriptedSynthesizer:
    """Writes a fixed sine WAV per call, recording speakers."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.calls: list[str] = []

    async def synthesize(self, request, output_path: Path) -> AudioArtifact:
        import numpy as np

        self.calls.append(request.speaker)
        n = self.sample_rate  # 1 second
        wave = 0.25 * np.sin(2 * np.pi * 330 * np.arange(n) / self.sample_rate)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, wave, self.sample_rate, subtype="PCM_16")
        return AudioArtifact(path=output_path, sample_rate_hz=self.sample_rate, duration_s=1.0)


class CopyConverter:
    """Copies the source artifact like the passthrough converter."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def convert(self, source, profile, output_path: Path) -> AudioArtifact:
        import shutil

        self.calls.append(profile.model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.path, output_path)
        return AudioArtifact(path=output_path, sample_rate_hz=source.sample_rate_hz, duration_s=1.0)


def test_load_corpus_strips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("line one\n\n  \nline two\n", encoding="utf-8")
    assert load_corpus(path) == ["line one", "line two"]


def test_peak_dbfs_known_sine(tmp_path: Path) -> None:
    import numpy as np

    path = tmp_path / "p.wav"
    sf.write(path, 0.5 * np.sin(np.linspace(0, 100, 1000)), 16000, subtype="PCM_16")
    assert peak_dbfs(path) == pytest.approx(-6.02, abs=0.1)


def test_run_benchmark_writes_auditions_and_rows(tmp_path: Path) -> None:
    synthesizer = ScriptedSynthesizer()
    converter = CopyConverter()
    out_dir = tmp_path / "bench"
    conversion = VoiceConversionProfile(backend="rvc", model="Fake", pitch_shift=0)

    rows = asyncio.run(
        run_benchmark(
            synthesizer,
            converter,
            ["Aiden", "Ryan"],
            ["hello world", "second line"],
            out_dir,
            conversion=conversion,
        )
    )

    assert len(rows) == 4
    assert all(row.error == "" for row in rows)
    assert synthesizer.calls == ["Aiden", "Aiden", "Ryan", "Ryan"]
    assert converter.calls == ["Fake", "Fake", "Fake", "Fake"]
    aiden_source = out_dir / "Aiden" / "source" / "line_01.wav"
    aiden_converted = out_dir / "Aiden" / "converted" / "line_01.wav"
    assert aiden_source.is_file() and aiden_converted.is_file()
    for row in rows:
        assert row.duration_s == pytest.approx(1.0)
        assert row.tts_time > 0 and row.vc_time > 0 and row.total_time > 0
        assert row.tts_rtf > 0 and row.vc_rtf > 0 and row.total_rtf > 0
        assert row.peak_dbfs == pytest.approx(-12.04, abs=0.2)


def test_run_benchmark_records_error_and_continues(tmp_path: Path) -> None:
    class FailingConverter(CopyConverter):
        async def convert(self, source, profile, output_path):  # noqa: ANN001
            if source.path.name.startswith("line_01"):
                msg = "worker exploded"
                raise RuntimeError(msg)
            return await super().convert(source, profile, output_path)

    out_dir = tmp_path / "bench"
    rows = asyncio.run(
        run_benchmark(
            ScriptedSynthesizer(),
            FailingConverter(),
            ["Aiden"],
            ["bad line", "good line"],
            out_dir,
            conversion=VoiceConversionProfile(backend="rvc", model="Fake"),
        )
    )

    assert rows[0].error.endswith("worker exploded")
    assert rows[0].duration_s == 0.0 and rows[0].vc_rtf == 0.0
    assert rows[1].error == ""
    assert (out_dir / "Aiden" / "converted" / "line_02.wav").is_file()


def test_write_results_csv_columns(tmp_path: Path) -> None:
    row = BenchmarkRow(
        speaker="Ryan",
        line=1,
        text="hi",
        tts_time=1.0,
        vc_time=0.5,
        total_time=1.5,
        duration_s=2.0,
        tts_rtf=0.5,
        vc_rtf=0.25,
        total_rtf=0.75,
        peak_dbfs=-3.0,
    )
    path = tmp_path / "results.csv"
    write_results_csv([row], path)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows_out = list(reader)
    assert reader.fieldnames == [
        "speaker",
        "line",
        "text",
        "tts_time",
        "vc_time",
        "total_time",
        "duration_s",
        "tts_rtf",
        "vc_rtf",
        "total_rtf",
        "peak_dbfs",
        "error",
    ]
    assert rows_out[0]["speaker"] == "Ryan"
    assert float(rows_out[0]["vc_rtf"]) == pytest.approx(0.25)
    assert rows_out[0]["error"] == ""


def test_write_metadata_is_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    metadata = {
        "tts_model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "vc_model": "Goku",
        "speakers": ["Aiden", "Ryan"],
        "failures": 0,
    }
    write_metadata(metadata, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["vc_model"] == "Goku"
    assert loaded["speakers"] == ["Aiden", "Ryan"]
