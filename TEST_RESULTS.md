# Test Results — notamusicbot rework (Round-8 re-review return)

Semantics: every result is PASS / FAIL / SKIPPED / NOT_RUN. Nothing
unavailable was converted into a pass. The **current state** is the Round-8
return: Round-7 review fixes (commit `9c5167b` on top of Round-7 commits
`5de5770aed6235b43f3351b9b6be1c6db91bd3b4`,
`cceb819a1ecdca6704fb92f8eb2c6ec471ca7939`, `77ff4e7225338932f21f369189e6ae1654dfd637`;
START_COMMIT `f9673d75d0cf21bbde5a335fc72e8f52800d7a5b`; re-run this session).
Earlier-round tables below are **historical records kept for traceability**
(239 at Round-4, 253 at Round-5, 259 at Round-6, 272 at Round-7) and are
clearly dated; they are not current-state claims.

Correction of record (Round-5 review §5): the Round-5 report claimed a
"Concurrent RVC protocol requests PASS" that had NO matching test or
implementation in the package. That claim was false. It is now REAL: the
adapter-level transaction lock is implemented and the deterministic
fake-worker test exists (Round-6, item A below).

## Environment incident during Round-7 (resolved, stated honestly)

During this round an accidental `uv run` overwrote the legacy RVC `.venv`
with a Python 3.14 environment containing no torch. This caused **one**
real-worker test failure, which was initially — and wrongly — reported as
pre-existing. The venv was rebuilt from scratch via `scripts/setup.sh` (the
same from-scratch build proven in the Round-2/Round-3 returns, Gate R7) and
the full unit suite then passes **272/272**. The failure was an environment
incident, not a product defect; no test was changed to make it pass.

## Current round (Round-8) — Round-7 review items §2/§3

Gate mapping (Round-7 review return): §2 coordinator admission-boundary race
**FAIL → PASS**; §3 portable unit suite **FAIL → PASS**; §4 portable/main
count qualification **FIXED** (both environments reported separately below).

| Check (Round-7 review item) | Command / environment | Result | Count |
|---|---|---|---|
| §2 post-admission closed-state race | unit (`tests/unit/test_coordinator.py`) | **PASS** | a caller blocked on `_admission` when `close()` begins is rejected with `CoordinatorClosedError` after admission is granted — its function **never executes** (`ran == ["a"]` pinned); applied to both `_run_detached` and `slot()`; no `await` between the re-check and holder registration (close() cannot interleave). |
| §2 slot() same semantics | unit | **PASS** | `slot()` blocked on admission + `close()` → `CoordinatorClosedError`, no grant. |
| §3 portable unit suite (no discord) | stripped venv (no discord/qwen), `PYTHONPATH=src pytest -q tests/unit` | **PASS** | **266 passed, 2 skipped, 0 failed — collection succeeds** (skips are the two justified `importorskip("discord")` modules: command-layer test, command-path parity test). `stop_guild_players` extracted to `discord_bot/player.py`, which imports no `discord` (review Option B). |
| Unit tests (main venv) | `.venv-main/bin/python -m pytest tests/unit` | **PASS** | **274 passed, 0 failed, 0 skipped, NO deselects** (272 + 2 new admission-race tests; re-run ×3 this session). This is the **development-environment** count; the portable no-Discord gate is the row above — they are reported separately per review §4. |
| mypy | `.venv-main/bin/python -m mypy src/notamusicbot` | **PASS** | **0 errors in 48 files** (unchanged; re-run this round). |
| ruff | `ruff check src tests` + `format --check` | **PASS** | clean. |

One honesty note: a single transient unit failure (test unidentified before the
re-run passed) occurred once immediately after a bulk `ruff --fix`/`format`
pass mid-round; three consecutive full runs after that were clean (274/274/274).
No flake has been reproduced since.

## Previous round (Round-7) — Round-6 review items §2/§3/§4/§5

Gate mapping (Round-6 review return): §2 RVC lifecycle **FAIL → PASS**;
§3 post-shutdown RVC restart **FAIL → PASS**; §4 report inconsistency
**FIXED** (this reconciliation); §5 stale counts **FIXED** (this
reconciliation).

| Check (Round-6 review item) | Command / environment | Result | Count |
|---|---|---|---|
| Unit tests (main venv, Round-7 historical) | `.venv-main/bin/python -m pytest tests/unit` | **PASS** | **272 passed, 0 failed, 0 skipped, NO deselects** (259 baseline + 13 Round-7 lifecycle tests). |
| §2 RVC lifecycle single-flight — concurrent start | unit | **PASS** | concurrent `start()` → exactly **one** worker spawn (2-PID leak impossible): start/shutdown serialized under `_lifecycle_lock`; lock order lifecycle → transaction, never reversed. |
| §2 shutdown during startup | unit, event-synced | **PASS** | shutdown strictly follows startup completion; test is deterministic and event-synced, stable across 3 consecutive runs (was the failing `test_shutdown_waits_for_active_transaction` class). |
| §2 failed startup reaping | unit | **PASS** | failed-ready and failed-health starts terminate the worker and reset `_process` / `_loaded_key` / `_orphaned_ids` — no leaked worker. |
| §3 coordinator close()/drain() | unit (`tests/unit/test_coordinator.py`) | **PASS** | closed flag rejects new submissions with `CoordinatorClosedError`; queued-not-started holders cancelled; started holders awaited to real completion. |
| §3 router latched closed state | unit (`tests/unit/test_voice_converter_router.py`) | **PASS** | `convert()` after `shutdown()` raises `RouterClosedError` instead of lazily restarting RVC (review reproduced starts=2/shutdowns=1 — now impossible); `startup()` is the deliberate re-enable. |
| §3 pipeline during application shutdown | unit (`tests/unit/test_app_shutdown_drain.py`) | **PASS** | `Application.shutdown()` order: cancel warmup → `coordinator.close()` (closed + drain) → `converter_router.shutdown()`; in-flight job retires before RVC teardown; RVC starts never increase after shutdown begins. |
| §3 Discord close ordering | unit | **PASS** | `bot.close()` stops all guild player tasks via `stop_guild_players` (await/cancel) **BEFORE** converter teardown and `super().close()`. |
| mypy | `.venv-main/bin/python -m mypy src/notamusicbot` | **PASS** | **0 errors in 48 files** (unchanged; re-run this round). |
| ruff check + format --check (changed files) | on `src/notamusicbot/**` and the changed test files | **PASS** | clean. |
| Live voice-channel playback | human in a Discord VC | **NOT_RUN** | requires a human participant; left for the user (carried). |

GPU evidence note (review §7.D): the real-GPU runs were **not** re-run this
round, per the review's explicit allowance — the Round-7 lifecycle changes
(`5de5770`/`cceb819`/`77ff4e7`) alter only shutdown/lifecycle ordering, not GPU
scheduling or inference behavior. The current evidence files
`evidence/gpu_arbitration.json` and `evidence/gpu_residency.json` (and the
RVC switching evidence) date from the `80e6146`/`43a39ef` era; their numbers
remain valid for scheduling behavior, which is unchanged.

`evidence/` current state: `gpu_arbitration.json`, `gpu_residency.json`,
`rvc_cross_model_switching.json`, `rvc_same_model_warm.json`.
`rvc_variant_switching.json` was **deleted** in Round-6 and no longer exists;
it is referenced below only in dated historical records.

## Historical: Round-6 return (259 passed; kept for traceability)

| Check (review item) | Command / environment | Result | Count |
|---|---|---|---|
| Unit tests (main venv) | `.venv-main/bin/python -m pytest tests/unit` | **PASS** | **259 passed, 0 failed, 0 skipped, NO deselects** (253 baseline + 6 new Round-6 tests, re-run this session, 172 s). |
| mypy | `.venv-main/bin/python -m mypy src/notamusicbot` | **PASS** | **0 errors in 48 files**. |
| ruff check + format --check (changed files) | `.venv-main/bin/python -m ruff check / format --check` on `src/notamusicbot/speech/**` and the changed test files | **PASS** | clean |
| A. RVC IPC transaction serialization (§2) | unit, `tests/unit/test_rvc_adapter.py::test_concurrent_requests_serialize_full_transaction` + `test_shutdown_waits_for_active_transaction` | **PASS** | two concurrent `_request()` calls against a fake worker with reply-event timestamps: the second request is written only AFTER the first one's reply left the worker; both callers get their own reply; no mismatch, no discarded reply. Concurrent convert+shutdown: shutdown WAITS for the in-flight transaction (shutdown request logged only after the convert reply), convert completes successfully, worker then retires. |
| B. Cancelled reader ownership (§3) | unit, `test_cancelled_reader_consumes_own_reply_and_stream_stays_healthy` | **PASS** | reader N cancelled while blocked, reply N arrives → the reader consumes its OWN terminal reply and terminates; no mismatch, no WorkerDiedError, no consumption of reply N+1; the subsequent new request succeeds. |
| C. Queued-not-started GPU cancellation (§4) | unit, `test_run_blocking_cancelled_while_queued_never_runs` + `test_pipeline_cancelled_while_queued_never_runs` | **PASS** | queued `run_blocking` cancelled → fn never called; queued `SpeechPipeline.run` cancelled → TTS calls == 0, VC calls == 0, no artifacts. Already-started cancellation keeps the lease to real completion (existing tests still pass). |
| D. RVC switching evidence, both scenarios (§6) | real worker, RTX 4070 Ti SUPER, run ALONE, `tests/integration/test_rvc_switching.py -m "gpu and rvc and integration"` | **PASS** | 2 passed; evidence split per scenario: `evidence/rvc_cross_model_switching.json` (4 Goku↔Obama switches, avg **6.293 s**/switch, first load 5.773 s) + `evidence/rvc_same_model_warm.json` (warm same-model avg **0.624 s**). Stale `rvc_variant_switching.json` removed; `scripts/package.sh` ships all `evidence/*.json`. |
| E. queue_wait_s is real queue wait (§7) | unit, `test_queue_wait_measures_lease_wait_not_synthesis` | **PASS** | coordinator deliberately occupied ~0.4 s; asserted `queue_wait_s >= 0.25 s` and `queue_wait_s > tts_s` — it measures lease entry − submission, not synthesis duration. |

Round-6 table below is kept for traceability (historical, 2026-10-03);
where its Round-5 inherited rows were wrong they are corrected in the
Round-7 current table, not silently rewritten.

## Round-5 return (traceability; the §12-E row's "PASS" was FALSE then, now made real in Round-6)

| Check | Command / environment | Result | Count |
|---|---|---|---|
| Unit tests (main venv) | `.venv-main/bin/python -m pytest tests/unit` | **PASS** | **253 passed, 0 failed, 0 skipped, NO deselects** (re-run and verified this session, 141 s). Grew 239 → 253 with the cancellation-safety, arbitration-exclusivity, startup-ordering, uploader-parity and concurrent-RVC tests. No test was removed. |
| mypy (NEW gate) | `.venv-main/bin/python -m mypy src/notamusicbot` | **PASS** | **0 errors in 48 files** (was 52 before `a1dd397`). `check_untyped_defs` kept; targeted overrides only for untyped third-party stubs (`yaml`, `yt_dlp`, `yt_dlp_ejs`, `spotipy`, `soundfile`, `soxr`, `qwen_tts`); exactly ONE targeted `arg-type` ignore with a written justification. No behavior changes in the type commit. |
| Cancellation-safe GPU lease — cancelled LLM holder (review §12 A) | unit, fake blocking thread | **PASS** | job A enters coordinator and starts its blocking thread → A's caller is cancelled → job B must NOT enter until A's thread really ends. Holds for `InferenceCoordinator.run_blocking` (detached holder acquires the semaphore BEFORE submitting to the executor and releases it only on real thread-future completion). |
| Cancellation-safe GPU lease — cancelled TTS holder (review §12 A) | unit, fake blocking model call through `QwenCustomVoiceSynthesizer`/pipeline | **PASS** | same invariant through `SpeechPipeline.run` → `run_detached` (same lease semantics for the pipeline's async chain; lease held across TTS→VC→postprocess). |
| Cancellation-safe GPU lease — cancelled RVC (review §12 A) | unit, fake sleeping worker | **PASS** | cancelling the caller during RVC conversion retains the lease until the subprocess reply actually completes; orphaned request ids are recorded and late replies are discarded (Option B semantics — lease owned by the coordinator, released only on real completion). |
| Stop during generated content (review §12 B) | unit, in-flight post-song generation + `stop()` + new inference request | **PASS** | the arbitration probe enters only after the detached work finishes; `max_active == 1`; queue not repopulated after stop; no GPU overlap. |
| Startup load→warmup ordering (review §12 C) | unit, `warmup_on_start: true` | **PASS** | strict ordering asserted: LLM model load completes before TTS warmup can acquire GPU ownership. `bot.py setup_hook` no longer redundantly loads the content model — `app.startup()` loads the LLM **before** RVC start/warmup. |
| YouTube uploader parity (review §12 D) | unit, fake yt-dlp result `title: Song, uploader: Artist` | **PASS** | `PlayableMedia.artist` flows resolver → queue → `ContentRequest.artist_name`; song commentary receives `song_name == Song`, `artist_name == Artist`. |
| Concurrent RVC protocol requests (review §12 E) | unit, two adapter requests against a fake worker | **WAS FALSE — now real (Round-6)** | At Round-5 neither the test nor the transaction lock existed (review reproduced `WorkerDiedError`). Round-6 implements the adapter-level transaction lock and the deterministic test — see the Round-6 table, item A. |
| Setup parses explicit `voice_conversion.variant` (review §9 follow-through) | unit, `president_alt.yaml` fixture with `variant: obama_trans` | **PASS** | setup validates `obama_trans.yaml`, not `president_alt.yaml`; proves `obama_trans.yaml` is validated even when profile filename and variant differ. Stale `max_concurrent_generations` removed from config. |
| GPU arbitration (real run, exclusivity on sampled lease-busy intervals) | `tests/integration/test_gpu_arbitration.py`, RTX 4070 Ti SUPER, torch 2.14.1+cu130, driver 617.14, run alone | **PASS** | 3 iterations, serialized; exclusivity now asserted on **sampled lease-busy intervals** (stronger than wall-clock timestamp edges); raw timeline committed as `evidence/gpu_arbitration.json` (timestamp, commit, GPU, driver, torch, timeline, result). |
| GPU residency (real run) | three models resident, RTX 4070 Ti SUPER, run alone | **PASS** | peak **9,854 MiB**; **no OOM**; raw artifact committed as `evidence/gpu_residency.json`. |
| RVC variant switching (real run, 2 tests) | real worker, run alone | **PASS** | Round-5 numbers: 4 switches, cross-model avg 6.23 s vs 0.82 s warm. Evidence file was WRONG then (one-element JSON overwritten; cross-model rows lost) — FIXED in Round-6: see `evidence/rvc_cross_model_switching.json` + `evidence/rvc_same_model_warm.json`. |
| Stripped environment (no discord / no qwen_tts, CPU) | Round-4 round independent run | **PASS** | **244 passed, 2 skipped** — both skips are justified `discord` `importorskip`s (Discord command layer tests). Not re-run this session (environment identical; unit delta 239→253 is fully covered by the main-venv run above). |

## Round-4 review required tests (review §12, items A–F) — satisfaction map

| Required test (Round-4 review) | Satisfied by | Result |
|---|---|---|
| A. Cancellation-safe GPU lease (LLM / TTS / RVC): cancelled holder keeps the lease until blocking work really ends | `run_blocking` + `run_detached` (`6e011bf`); A-LLM / A-TTS / A-RVC tests | **PASS** |
| B. Stop during generated content: no GPU overlap, no post-stop queue repopulation | probe test: enters only after detached work finishes; `max_active == 1` | **PASS** |
| C. Startup warmup/load ordering: `warmup_on_start=true` must not overlap LLM load | strict ordering test; `app.startup()` loads LLM before RVC/warmup (`55b7e16` dropped the redundant load) | **PASS** |
| D. YouTube uploader parity: `song_name == Song`, `artist_name == Artist` | `PlayableMedia.artist` end-to-end (`55b7e16`) | **PASS** |
| E. Concurrent RVC protocol requests: no reply loss/mismatch | adapter-level transaction serialization + fake-worker test | **PASS** |
| F. Static type checking: `mypy src/notamusicbot`, exact result recorded | `a1dd397`: **0 errors in 48 files** | **PASS** |

## Gate records (Round-5 disposition of every Round-4 gate)

| Gate | Result | Evidence |
|---|---|---|
| R1 package security | **PASS** | allowlist + secret scan unchanged; `.env.example`-only rule holds |
| R2 production speech routing | **PASS** | unchanged; new tests do not alter routing |
| R3 migrated voices | **PASS** | variant identity end-to-end; `president_alt.yaml` fixture proves the explicit variant field is validated |
| R4 RVC persistence | **PASS** | variant hot-swap in-process; cross-model 6.29 s vs warm 0.62 s (Round-6 re-run); `evidence/rvc_cross_model_switching.json` + `evidence/rvc_same_model_warm.json` |
| R5 media lifecycle | **PASS** | outcome/cleanup fixes hold; pipeline cancellation keeps the lease and discards orphaned RVC replies |
| R6 content preservation | **PASS** | STREAM+NATURAL-only hook holds; YouTube uploader metadata restored (`55b7e16`) |
| R7 clean install | **PASS** | setup parses explicit `voice_conversion.variant`; asset validation unchanged |
| R8 GPU safety | **PASS** (Round-4 FAIL/PARTIAL closed) | cancellation-safe lease (`6e011bf`); exclusivity on sampled lease-busy intervals (`80e6146`); raw evidence committed (`43a39ef`): `evidence/gpu_arbitration.json`, `evidence/gpu_residency.json`, `evidence/rvc_cross_model_switching.json`, `evidence/rvc_same_model_warm.json` |

## Previous round (Round-4 return, kept for traceability)

Results carried from the Round-4 return (`f804284` → `9d370fe`,
2026-10-02); none regressed:

| Check | Result | Count |
|---|---|---|
| Unit tests (main venv) | **PASS** | 239 passed, 0 failed, 0 skipped (coordinator determinism fix `25b0ac7`) |
| Lint (`ruff check`) / Format (`ruff format --check`) | **PASS** | clean |
| RVC variant preservation (`test_rvc_variant.py`) | **PASS** | 9 tests |
| User migration asserts config identity | **PASS** | 30 users carry correct `variant` |
| Playback outcome matrix (`test_player.py`) | **PASS** | NATURAL→1; SKIPPED/STOPPED/FAILED/LOCAL→0 |
| Crash-orphan cleanup (`test_job_artifacts.py`) | **PASS** | `.temporary` markers + retention prune |
| Coordinator mutual exclusion (`test_coordinator.py`) | **PASS** | 3 tests |
| Interstitial guild scoping | **PASS** | 4 tests |
| Setup RVC asset validation (`test_setup_rvc_assets.py`) | **PASS** | 5 tests (caught real `GOKU`/`Goku` mismatch) |
| Silent-worker startup timeout | **PASS** | select-based deadline read |
| Podcast `reply_turns` consistency | **PASS** | code/config/docs consistent |
| RVC integration (Goku, real GPU) | **PASS** | 2 passed |
| GPU residency (Round-3 real run) | **PASS** | idle 11,262 / 16,375 MiB; peaks ≤ 9,857 MiB; no OOM |
| Clean install (fresh clone + `setup.sh`) | **PASS** | legacy venv built from scratch |

## Carried-over checks

| Check | Result | Count |
|---|---|---|
| Live Discord smoke (gateway connect + command registration) | **PASS** | coordinator, 2026-10-02 |
| Live voice-channel playback | **NOT_RUN** | requires a human in a VC; left for the user |

## Notes

- Historical (Round-5, dated): the unit suite grew 239 → 253 with the
  Round-5 tests (cancellation-safe
  lease A-LLM/A-TTS/A-RVC, stop-during-post-song, startup load→warmup
  ordering, YouTube uploader parity, concurrent RVC requests, arbitration
  interval sampling). No test was removed; **no deselects** are used.
- Design note: `run_blocking` is the stronger thread-level shield and is
  used for LLM generation and app warmup. `run_detached` provides the same
  lease semantics for the pipeline's async chain (TTS→VC→postprocess,
  including the RVC subprocess reply wait). The whole-chain-in-thread
  variant deadlocked the cancelled-job artifact test and was abandoned — a
  documented deviation; the implementation still satisfies Option B
  semantics: the lease is owned by the coordinator and released only on
  real completion. RVC cancellation retains the lease until the subprocess
  reply; orphaned request ids are recorded and late replies discarded.
- Raw real-GPU evidence is committed under `evidence/` and packaged by
  `scripts/package.sh`.
- Benchmarks are reported separately in `BENCHMARK_RESULTS.md`.
