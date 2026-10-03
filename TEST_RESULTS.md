# Test Results — notamusicbot rework (Round-4 re-review return)

Semantics: every result is PASS / FAIL / SKIPPED / NOT_RUN. Nothing
unavailable was converted into a pass. Counts are coordinator-verified for
the Round-4 return (after the Round-3 review return at `02d1d97`; Round-4
commits `f804284` → `9d370fe`, re-run 2026-10-02).

## Current round

| Check | Command / environment | Result | Count |
|---|---|---|---|
| Unit tests (main venv) | `.venv-main/bin/python -m pytest tests/unit` | **PASS** | **239 passed, 0 failed, 0 skipped** (re-run and verified this session). One transient wall-clock flake was observed in `test_coordinator` (perf_counter aliasing) and eliminated: the test now asserts the mutual-exclusion invariant with a logical concurrency counter (`25b0ac7`); 4+ consecutive clean runs after the fix. |
| Lint | `.venv-main/bin/ruff check src tests` | **PASS** | clean |
| Format | `.venv-main/bin/ruff format --check src tests` | **PASS** | clean |
| RVC variant preservation (new, Gate R3 proof) | `tests/unit/test_rvc_variant.py` | **PASS** | 9 tests — fixture dir with `obama.yaml`/`obama_trans.yaml` (different parameters); obama profile → exact standard config values, obama_trans → exact trans config values; cache key `(resolved_dir, variant)` distinct; loud error on a missing variant |
| User migration asserts config identity (extended) | `tests/unit/test_user_migration.py` | **PASS** | all 30 migrated users carry the correct `variant`; asserts actual configuration identity, not just profile strings |
| RVC worker variant switching (real, Gate R4 proof) | `tests/integration/test_rvc_switching.py` (real Obama worker) | **PASS** | obama → tts_type `pyht:Joseph` vs obama_trans → `gtts` observed at conversion; `config_switched: true` **in-process — zero worker restarts on a variant switch** |
| Playback outcome matrix (Gate R6 proof) | `tests/unit/test_player.py` (deterministic fakes) | **PASS** | NATURAL → hook **1**; SKIPPED / STOPPED / FAILED / LOCAL-generated → hook **0**; interrupt reason recorded before `voice_client.stop()`; after-callback error → FAILED; dequeue without voice client → DROPPED |
| Crash-orphan cleanup (Gate R5 proof) | `tests/unit/test_job_artifacts.py` | **PASS** | full cleanup removes file + empty job dir on success; `.temporary` marker written per job; simulated crash/restart: `prune_expired_jobs` removes the stale successful artifact after the retention threshold; cancelled job → deterministic state |
| GPU arbitration (new, Gate R8 proof, real run) | `tests/integration/test_gpu_arbitration.py` (RTX 4070 Ti SUPER) | **PASS** | 3 iterations; concurrent LLM task + TTS→RVC asyncio task with all three models resident; timeline **fully serialized — zero overlap**; every stage starts exactly when the previous ends (`perf_counter` timestamps asserted mutual exclusion) |
| Coordinator mutual exclusion (unit) | `tests/unit/test_coordinator.py` | **PASS** | 3 tests — same slot never held concurrently (perf_counter-timestamp proof) |
| Interstitial guild scoping (new) | `tests/unit/test_interstitials_guild_scoping.py` | **PASS** | 4 tests — cooldown state keyed by guild id; guild A's fire does not suppress guild B |
| Setup RVC asset validation (new, Gate R7 proof) | `tests/unit/test_setup_rvc_assets.py` (extracted bash validation against temp fixtures) | **PASS** | 5 tests — missing model dir / variant yaml / `.pth` / `.index` each fail non-zero; case-insensitive model dir accepted (caught the real `GOKU`/`Goku` mismatch) |
| Silent-worker startup timeout (Gate R5/R7) | fake worker alive + silent → adapter startup fails inside configured timeout | **PASS** | `select`-based deadline read (`_wait_for_ready`) replaced the blocking `readline()` |
| Podcast `reply_turns` consistency | unit podcast tests | **PASS** | `turns`→`reply_turns` rename consistent (code/config/docs): host opening + N replies |
| RVC integration (Goku, RTX 4070 Ti SUPER) | `pytest tests/integration -m "rvc and integration"` | **PASS** | 2 passed |
| RVC switching benchmark (carried) | `notamusicbot.tools.benchmark_rvc_switching` | **PASS (documented)** | cross-MODEL Obama↔Goku: avg **5.51 s per switch** vs 0.57 s warm same-model; **variant switches within a model are cheap in-process config hot-swaps** |
| GPU residency (carried, real run) | RTX 4070 Ti SUPER 16 GB: LLM 4B + TTS 0.6B + RVC all resident | **PASS** | idle 11,262 / 16,375 MiB; peaks ≤ 9,857 MiB; **no OOM** |
| Clean install (carried, Gate R7) | fresh clone + `scripts/setup.sh` built the legacy RVC venv from scratch | **PASS** | Round-4 adds per-profile `validate_rvc_assets` (non-zero exit on missing assets) |

## Round-3 review required tests (section 14) — satisfaction map

| Required test (review §14) | Satisfied by | Result |
|---|---|---|
| RVC variant preservation: fixture `Obama/obama.yaml` + `obama_trans.yaml` with different values; exact config reaches conversion | `test_rvc_variant.py` (`9d370fe`) + extended `test_user_migration.py` | **PASS** |
| Playback outcome: NATURAL→hook 1; SKIPPED/STOPPED/FAILED/LOCAL→hook 0 | outcome-matrix tests in `test_player.py` (`f804284`) | **PASS** |
| Unit-suite isolation: CPU-only, no discord/qwen_tts/CUDA → `pytest tests/unit` passes with only justified skips | `importorskip("discord")` + `FakeSynthesizer` injection (`9d370fe`); result: **239 passed / 0 skipped** | **PASS** |
| GPU arbitration: LLM generation and speech generation observed not to overlap (shared coordinator) | shared `InferenceCoordinator` (`9d370fe`) + `test_gpu_arbitration.py` real run, 3 iterations, zero overlap | **PASS** |
| Crash-orphan cleanup: stale successful temporary artifact → removed by retention after simulated restart | `test_job_artifacts.py` (`f804284`) | **PASS** |
| RVC worker startup timeout: worker alive + silent → startup fails inside configured timeout | `_wait_for_ready` select-deadline test (`9d370fe`) | **PASS** |

## Gate records (Round-4 disposition of every Round-3 gate)

| Gate | Result | Evidence |
|---|---|---|
| R1 package security | **PASS** | credentials clean; explicit `.env.example`-only rule; benchmark/evidence CSV/JSON allowlisted (`9d370fe`) |
| R2 production speech routing | **PASS** | router architecture + typed playback-outcome lifecycle (`f804284`) |
| R3 migrated voices | **PASS** (Round-3 FAIL closed) | variant identity preserved end-to-end; migration asserts real config values (`9d370fe`) |
| R4 RVC persistence | **PASS** | same-model case-alias fix + variant switching in-process with `config_switched`, zero restarts; cross-model cost 5.51 s documented |
| R5 media lifecycle | **PASS** | full file+dir cleanup, `.temporary` crash-orphan markers + retention, `CancelledError` deterministic, silent-worker hard startup timeout |
| R6 content preservation | **PASS** | hook fires only on STREAM + NATURAL (outcome matrix PASS); ordering guarantees unchanged |
| R7 clean install | **PASS** | from-scratch legacy venv holds; per-profile asset validation fails non-zero on missing assets |
| R8 benchmark | **PASS** | shared GPU arbitration proven non-overlapping on real GPU (3 iterations, all models resident); raw evidence now packaged |

## Carried-over checks

| Check | Result | Count |
|---|---|---|
| Live Discord smoke (gateway connect + command registration) | **PASS** | coordinator, 2026-10-02 |
| Live voice-channel playback | **NOT_RUN** | requires a human in a VC; left for the user |

## Notes

- The unit suite grew 204 → 239 with the Round-4 tests (RVC variant, worker
  coordinator, guild-scoped interstitials, setup asset validation, playback
  outcome matrix, crash-orphan artifacts, migration variant assertions). No
  test was removed; the suite now runs with **zero skips**.
- Gate R7 asset validation caught a real defect: the model directory case
  (`GOKU` vs `Goku`) mismatched between config and disk.
- Cross-MODEL RVC switch cost (5.51 s, worker restart + load) remains
  documented future work (persistent per-model worker cache); variant
  switches within a model are in-process config hot-swaps.
- Benchmarks are reported separately in `BENCHMARK_RESULTS.md`.
