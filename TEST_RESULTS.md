# Test Results — notamusicbot rework (review-response round)

Semantics: every result is PASS / FAIL / SKIPPED / NOT_RUN. Nothing
unavailable was converted into a pass. All counts below are
coordinator-verified (re-run 2026-10-02).

## Current round

| Check | Command / environment | Result | Count |
|---|---|---|---|
| Unit tests (main venv) | `.venv-main/bin/python -m pytest tests/unit` | **PASS** | 181 passed, 0 failed |
| Unit tests without discord (P1-14 proof) | `[media,dev]` venv with **discord.py absent**, `pytest tests/unit` | **PASS** | 180 passed, 1 skipped — the skip is `test_app_production_routing` via `importorskip`; proves the suite does not require the discord extra |
| RVC integration (Goku, RTX 4070 Ti SUPER) | `pytest tests/integration -m "rvc and integration"` | **PASS** | 2 passed |
| Clean-clone unit suite (Gate R7) | fresh `git clone` + submodule init + fresh uv py3.12 venv `.[media,dev]` | **PASS** | 180 passed, 1 skipped; import OK |
| Clean-clone RVC smoke (Gate R7) | same clone, after documented local assets supplied | **PASS** | 2 passed |
| Production routing proof (Gate R2) | `tests/unit/test_app_production_routing.py` — spy RVC adapter through the real composition root | **PASS** | — |
| Voice migration (Gate R3) | real legacy user file through migration | **PASS** | 29 users → obama, 1 → obama_trans (`legacy_alias` weights); 0 goku; unknown model exits non-zero |
| RVC persistence (Gate R4) | two sequential Goku conversions | **PASS** | 1 worker start, 1 load-model, 2 converts, 0 reloads |
| Media lifecycle (Gate R5) | fallback download → playback → file deleted; webpage-URL download with no stream URL | **PASS** | — |
| Content paths (Gate R6) | song / news / reddit / podcast through ContentProvider → SpeechPipeline → queue | **PASS** | interstitial + podcast default disabled |
| Benchmark (Gate R8) | re-run after P1-8 fix | **PASS** | 10/10 lines, 0 failures — see `BENCHMARK_RESULTS.md` |

## Carried-over checks (previous round, unchanged code paths)

| Check | Result | Count |
|---|---|---|
| Lint (`ruff check src tests`) | **PASS** | — |
| Format (`ruff format --check`) | **PASS** | 61 files |
| Live Discord smoke (gateway connect + command registration, as Goku#8715) | **PASS** | coordinator, 2026-10-02 |
| Live voice-channel playback | **NOT_RUN** | requires a human in a VC; left for the user |
| Qwen TTS smoke / end-to-end local speech | **PASS** | previous-round evidence; artifacts retained |

## Notes

- The unit-count difference (181 vs 180 + 1 skip) is exactly the
  discord-gated routing test: with discord absent it skips via
  `importorskip` instead of running — this is the P1-14 proof, not a lost
  test.
- Gate R7 used `.[media,dev]` in the clone (no discord extra) for the
  no-discord proof; the RVC smoke passed only after supplying the documented
  local assets (legacy `.venv` runtime symlink, `hubert_base.pt`, the missing
  `goku.yaml`, partially tracked `pth`/`index` files). The review's point
  that tracked assets are incomplete is CONFIRMED; `scripts/setup.sh` now
  prints the exact required list (see `IMPLEMENTATION_REPORT.md`).
- Benchmarks are reported separately in `BENCHMARK_RESULTS.md`.
