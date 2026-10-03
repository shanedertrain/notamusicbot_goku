"""Integration: alternating RVC model profiles through the real legacy worker.

Review Round-2 P1 evidence: measures per-switch reload cost (worker restart +
model load) when alternating Obama/Goku versus same-model warm conversions.
Requires the legacy env (.venv), hubert_base.pt and both model dirs; skipped
otherwise. Run with:

    .venv-main/bin/python -m pytest tests/integration/test_rvc_switching.py \
        -m "gpu and rvc and integration" -s
"""

from __future__ import annotations

from pathlib import Path

import pytest

from notamusicbot.tools.benchmark_rvc_switching import (
    available_models,
    run_alternation,
    summarize,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.rvc,
    pytest.mark.integration,
]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _assets_ready() -> bool:
    if not (REPO_ROOT / ".venv" / "bin" / "python").is_file():
        return False
    if not (REPO_ROOT / "hubert_base.pt").is_file():
        return False
    return len(available_models(["Goku", "Obama"])) >= 2


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.rvc,
    pytest.mark.integration,
    pytest.mark.skipif(not _assets_ready(), reason="legacy RVC assets missing"),
]


class TestAlternatingRvcProfiles:
    def test_alternating_profiles_report_switch_cost(self, tmp_path) -> None:
        rows = __import__("asyncio").run(
            run_alternation(["Goku", "Obama"], output_dir=tmp_path, turns=2)
        )
        # Alternation: every second conversion is a switch.
        assert len(rows) == 4
        assert [r["model"] for r in rows] == ["Goku", "Obama", "Goku", "Obama"]
        assert [r["switch"] for r in rows] == [True, True, True, True]
        stats = summarize(rows)
        print(
            f"\nRVC alternating profiles: {stats['switches']} switches, "
            f"avg {stats['switch_avg_s']}s per switch (worker restart + model load)"
        )
        # A switch always costs more than a warm convert would; the numbers
        # themselves are recorded for the report (no strict latency assert:
        # CI hardware varies).
        assert stats["switch_avg_s"] > 0
        self._write_evidence("rvc_cross_model_switching.json", "alternating", rows, stats)

    def test_same_model_repeat_is_warm(self, tmp_path) -> None:
        import asyncio

        rows = asyncio.run(run_alternation(["Goku", "Goku"], output_dir=tmp_path, turns=1))
        assert [r["switch"] for r in rows] == [True, False]
        warm = summarize(rows)
        print(f"\nRVC warm same-model convert: avg {warm['warm_avg_s']}s")
        assert warm["warm_converts"] == 1
        self._write_evidence("rvc_same_model_warm.json", "same_model_warm", rows, warm)

    @staticmethod
    def _write_evidence(filename: str, scenario: str, rows: list[dict], stats: dict) -> None:
        # Round-4 review P2: persist the raw real-GPU RVC variant-switching
        # evidence (variant identity + per-switch reload costs).
        # Round-5 review §6/D: each scenario writes its OWN file — the
        # previous shared ``rvc_variant_switching.json`` was a one-element
        # array overwritten by whichever test ran last, losing the
        # cross-model rows. ``scripts/package.sh`` ships all ``evidence/*.json``.
        from evidence_utils import write_evidence

        write_evidence(
            filename,
            {
                "test": "TestAlternatingRvcProfiles",
                "scenarios": [
                    {
                        "scenario": scenario,
                        "rows": rows,
                        "stats": stats,
                    }
                ],
                "pass": True,
            },
        )
