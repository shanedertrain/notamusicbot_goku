from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from notamusicbot.domain.listen import ListenTurnMetrics
from notamusicbot.listen.metrics import format_turn_metrics, parse_turn_line

_BENCH_PATH = Path(__file__).resolve().parents[2] / "tools" / "listen_bench.py"
_spec = importlib.util.spec_from_file_location("listen_bench_tool", _BENCH_PATH)
assert _spec and _spec.loader
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

LINE_RE = re.compile(
    r"^listen turn guild=\d+ user=\d+ eos->asr=\d+\.\d\d asr->llm=\d+\.\d\d "
    r"llm->tts1=\d+\.\d\d tts1->audio=\d+\.\d\d total=\d+\.\d\d$"
)


def _metrics(**kw: float) -> ListenTurnMetrics:
    base = {
        "eos_to_asr_s": 0.40,
        "asr_to_llm_s": 1.20,
        "llm_to_tts1_s": 2.70,
        "tts1_to_audio_s": 0.30,
    }
    base.update(kw)
    return ListenTurnMetrics(turn_id="t", user_id=7, **base)


def test_format_matches_regex_with_total_4_60() -> None:
    line = format_turn_metrics(_metrics(), 11, 7)
    assert LINE_RE.match(line)
    assert line == (
        "listen turn guild=11 user=7 eos->asr=0.40 asr->llm=1.20 "
        "llm->tts1=2.70 tts1->audio=0.30 total=4.60"
    )


def test_explicit_total_wins() -> None:
    line = format_turn_metrics(_metrics(total_s=5.0), 1, 2)
    assert line.endswith("total=5.00")


def test_missing_stage_is_na_and_total_sums_rest() -> None:
    line = format_turn_metrics(_metrics(llm_to_tts1_s=0.0, tts1_to_audio_s=0.0), 1, 2)
    assert "llm->tts1=n/a tts1->audio=n/a" in line
    assert line.endswith("total=1.60")
    parsed = parse_turn_line(line)
    assert parsed is not None
    assert parsed["llm->tts1"] is None and parsed["total"] == 1.6


def test_parse_round_trip_with_log_prefix() -> None:
    line = "2026-10-25 12:00:00 INFO x: " + format_turn_metrics(_metrics(), 11, 7)
    assert parse_turn_line(line) == {
        "guild": "11",
        "user": "7",
        "eos->asr": 0.40,
        "asr->llm": 1.20,
        "llm->tts1": 2.70,
        "tts1->audio": 0.30,
        "total": 4.60,
    }


def test_parse_rejects_other_lines() -> None:
    assert parse_turn_line("speech job metrics foo") is None
    assert parse_turn_line("") is None


def test_nearest_rank_percentiles() -> None:
    data = [float(i) for i in range(1, 11)]
    assert bench.percentile(data, 50) == 5
    assert bench.percentile(data, 90) == 9
    assert bench.percentile([3.0], 90) == 3.0
    assert bench.percentile(list(reversed(data)), 50) == 5


def _write_log(path: Path, totals: list[float]) -> Path:
    lines = ["noise line"]
    for t in totals:
        lines.append(
            format_turn_metrics(ListenTurnMetrics(turn_id="t", user_id=1, total_s=t), 1, 1)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_cli_under_budget_exits_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = _write_log(tmp_path / "ok.log", [4.0, 5.0, 6.0, 7.0, 8.0])
    assert bench.main([str(log)]) == 0
    out = capsys.readouterr().out
    assert "total" in out and "within budget" in out


def test_cli_over_p90_exits_1(tmp_path: Path) -> None:
    log = _write_log(tmp_path / "p90.log", [2.0, 2.0, 2.0, 2.0, 9.5])
    assert bench.main([str(log)]) == 1


def test_cli_over_p50_exits_1(tmp_path: Path) -> None:
    log = _write_log(tmp_path / "p50.log", [6.5, 6.5, 6.5])
    assert bench.main([str(log)]) == 1
    assert bench.main([str(log), "--budget-p50", "7"]) == 0


def test_cli_reads_stdin_and_empty_is_2() -> None:
    line = format_turn_metrics(_metrics(), 1, 2)
    assert bench.main(["-"], stdin=[line]) == 0
    assert bench.main(["-"], stdin=["nothing"]) == 2
