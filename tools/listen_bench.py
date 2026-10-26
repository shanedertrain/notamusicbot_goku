"""Listen-mode latency report: P50/P90 per stage and total from log lines.

Reads "listen turn ..." log lines (a file, or stdin when no path or "-") and
prints count, P50 and P90 for each stage and the total. Exits 1 if total P90
exceeds --budget-p90 (default 9.0 s) or total P50 exceeds --budget-p50
(default 6.0 s), else 0. Lines that are not turn lines are ignored; no turns
at all exits 2.

Percentile method: nearest rank. For n sorted values and percentile p,
the result is the value at 1-based rank ceil(p/100 * n). So for 1..10,
P50 = 5 and P90 = 9. Stages that were "n/a" are skipped for that stage.

    .venv-main/bin/python tools/listen_bench.py bot.log
    journalctl ... | .venv-main/bin/python tools/listen_bench.py

The WAV harness (fakes / --gpu) is a separate task and not part of this file.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from notamusicbot.listen.metrics import FIELD_KEYS, parse_turn_line  # noqa: E402


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile: value at 1-based rank ceil(pct/100 * n)."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def collect(lines: Iterable[str]) -> dict[str, list[float]]:
    """Parse turn lines into per-field value lists (n/a skipped)."""
    series: dict[str, list[float]] = {key: [] for key in FIELD_KEYS}
    for line in lines:
        parsed = parse_turn_line(line)
        if parsed is None:
            continue
        for key in FIELD_KEYS:
            if parsed[key] is not None:
                series[key].append(parsed[key])
    return series


def main(argv: Sequence[str] | None = None, stdin: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log", nargs="?", default="-", help="log file, or - for stdin")
    parser.add_argument("--budget-p90", type=float, default=9.0)
    parser.add_argument("--budget-p50", type=float, default=6.0)
    args = parser.parse_args(argv)

    if args.log == "-":
        series = collect(sys.stdin if stdin is None else stdin)
    else:
        with open(args.log, encoding="utf-8", errors="replace") as fh:
            series = collect(fh)

    totals = series["total"]
    if not totals:
        print("no listen turn lines found", file=sys.stderr)
        return 2

    print(f"{'stage':<12} {'count':>5} {'p50':>7} {'p90':>7}")
    for key in FIELD_KEYS:
        vals = series[key]
        if vals:
            p50, p90 = f"{percentile(vals, 50):.2f}", f"{percentile(vals, 90):.2f}"
        else:
            p50 = p90 = "n/a"
        print(f"{key:<12} {len(vals):>5} {p50:>7} {p90:>7}")

    total_p50, total_p90 = percentile(totals, 50), percentile(totals, 90)
    over = []
    if total_p90 > args.budget_p90:
        over.append(f"total P90 {total_p90:.2f}s > {args.budget_p90:.2f}s")
    if total_p50 > args.budget_p50:
        over.append(f"total P50 {total_p50:.2f}s > {args.budget_p50:.2f}s")
    if over:
        print("OVER BUDGET: " + "; ".join(over))
        return 1
    print(f"within budget (P50 <= {args.budget_p50:.2f}s, P90 <= {args.budget_p90:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
