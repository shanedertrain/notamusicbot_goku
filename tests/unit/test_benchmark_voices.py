"""Unit tests for the voice benchmark CLI (fake synthesizer, no GPU)."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from notamusicbot.domain.speech import AudioArtifact, TtsRequest
from notamusicbot.tools.benchmark_voices import (
    BenchmarkRow,
    build_parser,
    load_corpus,
    peak_dbfs,
    run_benchmark,
    write_metadata,
    write_results_csv,
)

CORPUS = ["Next up: a song.", "That was something else."]


class FakeCloneTts:
    """Writes a deterministic WAV per request, recording the requests."""

    def __init__(self, fail_on_line: int | None = None) -> None:
        self.calls: list[TtsRequest] = []
        self.fail_on_line = fail_on_line

    async def synthesize(self, request: TtsRequest, output_path: Path) -> AudioArtifact:
        self.calls.append(request)
        if self.fail_on_line is not None and len(self.calls) == self.fail_on_line:
            raise RuntimeError("model exploded")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n = 2400
        sf.write(output_path, 0.5 * np.ones(n, dtype=np.float32), 24000, subtype="PCM_16")
        return AudioArtifact(path=output_path, sample_rate_hz=24000, duration_s=0.1)


def _targets(**modes: str) -> dict[str, TtsRequest]:
    return {
        profile: TtsRequest(
            text="placeholder",
            reference_audio=f"/refs/{profile}.wav",
            reference_text="ref",
            clone_mode=mode,
        )
        for profile, mode in modes.items()
    }


def test_run_benchmark_writes_audition_wavs_and_rows(tmp_path: Path) -> None:
    out = tmp_path / "bench"
    synth = FakeCloneTts()
    rows = asyncio.run(run_benchmark(synth, _targets(obama="xvec", goku="icl"), CORPUS, out))
    assert len(rows) == 4
    assert {row.profile for row in rows} == {"obama", "goku"}
    assert [row.text for row in rows if row.profile == "obama"] == CORPUS
    assert all(row.error == "" for row in rows)
    assert all(row.duration_s > 0 and row.total_time > 0 for row in rows)
    assert len(synth.calls) == 4
    # Each call carries the profile's reference clone wiring.
    modes = {(call.reference_audio, call.clone_mode) for call in synth.calls}
    assert modes == {("/refs/obama.wav", "xvec"), ("/refs/goku.wav", "icl")}
    for profile in ("obama", "goku"):
        assert sorted(p.name for p in (out / profile).glob("*.wav")) == [
            "line_01.wav",
            "line_02.wav",
        ]


def test_run_benchmark_records_failure_and_continues(tmp_path: Path) -> None:
    rows = asyncio.run(
        run_benchmark(FakeCloneTts(fail_on_line=2), _targets(obama="xvec"), CORPUS, tmp_path)
    )
    assert len(rows) == 2
    assert rows[0].error == ""
    assert "model exploded" in rows[1].error


def test_peak_dbfs_measures_full_scale(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/sine_2s.wav")
    assert peak_dbfs(fixture) <= 0.0  # a WAV cannot exceed full scale
    full_scale = tmp_path / "_full_scale.wav"
    sf.write(full_scale, np.ones(240, dtype=np.float32), 24000, subtype="PCM_16")
    # PCM_16 full scale is 32767/32768 (-0.0003 dBFS), not exactly 0.
    assert peak_dbfs(full_scale) == pytest.approx(0.0, abs=1e-3)


def test_csv_and_metadata_writers(tmp_path: Path) -> None:
    rows = [BenchmarkRow(profile="obama", line=1, text="hi", total_time=1.0, duration_s=2.0)]
    write_results_csv(rows, tmp_path / "out" / "results.csv")
    with (tmp_path / "out" / "results.csv").open(encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle))
    assert entries[0]["profile"] == "obama"
    assert entries[0]["text"] == "hi"

    write_metadata({"profiles": ["obama"]}, tmp_path / "out" / "metadata.json")
    assert json.loads((tmp_path / "out" / "metadata.json").read_text()) == {"profiles": ["obama"]}


def test_load_corpus_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("one\n\n  two  \n", encoding="utf-8")
    assert load_corpus(path) == ["one", "two"]


def test_parser_has_no_conversion_flags() -> None:
    args = build_parser().parse_args([])
    assert args.profiles == ["obama"]
    assert not hasattr(args, "no_vc")
