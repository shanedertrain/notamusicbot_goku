# Test Results — notamusicbot rework (Round-5 re-review return)

Semantics: every result is PASS / FAIL / SKIPPED / NOT_RUN. Nothing
unavailable was converted into a pass. Counts are coordinator-verified for
the Round-5 return (after the Round-4 review return; Round-5 commits
`6e011bf` → `a1dd397`, re-run 2026-10-03).

## Current round (Round-5)

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
| Concurrent RVC protocol requests (review §12 E) | unit, two adapter requests against a fake worker | **PASS** | full write→matching-reply transaction serialized at the adapter; both replies reach the correct caller with no id loss/mismatch. |
| Setup parses explicit `voice_conversion.variant` (review §9 follow-through) | unit, `president_alt.yaml` fixture with `variant: obama_trans` | **PASS** | setup validates `obama_trans.yaml`, not `president_alt.yaml`; proves `obama_trans.yaml` is validated even when profile filename and variant differ. Stale `max_concurrent_generations` removed from config. |
| GPU arbitration (real run, exclusivity on sampled lease-busy intervals) | `tests/integration/test_gpu_arbitration.py`, RTX 4070 Ti SUPER, torch 2.14.1+cu130, driver 617.14, run alone | **PASS** | 3 iterations, serialized; exclusivity now asserted on **sampled lease-busy intervals** (stronger than wall-clock timestamp edges); raw timeline committed as `evidence/gpu_arbitration.json` (timestamp, commit, GPU, driver, torch, timeline, result). |
| GPU residency (real run) | three models resident, RTX 4070 Ti SUPER, run alone | **PASS** | peak **9,854 MiB**; **no OOM**; raw artifact committed as `evidence/gpu_residency.json`. |
| RVC variant switching (real run, 2 tests) | real worker, run alone | **PASS** | 4 switches; cross-model switch avg **6.23 s** vs **0.82 s** warm same-model; variant switches stay in-process hot-swaps; raw artifact committed as `evidence/rvc_variant_switching.json`. |
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
| R4 RVC persistence | **PASS** | variant hot-swap in-process; cross-model 6.23 s vs warm 0.82 s documented; `evidence/rvc_variant_switching.json` |
| R5 media lifecycle | **PASS** | outcome/cleanup fixes hold; pipeline cancellation keeps the lease and discards orphaned RVC replies |
| R6 content preservation | **PASS** | STREAM+NATURAL-only hook holds; YouTube uploader metadata restored (`55b7e16`) |
| R7 clean install | **PASS** | setup parses explicit `voice_conversion.variant`; asset validation unchanged |
| R8 GPU safety | **PASS** (Round-4 FAIL/PARTIAL closed) | cancellation-safe lease (`6e011bf`); exclusivity on sampled lease-busy intervals (`80e6146`); raw evidence committed (`43a39ef`): `evidence/gpu_arbitration.json`, `evidence/gpu_residency.json`, `evidence/rvc_variant_switching.json` |

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

- The unit suite grew 239 → 253 with the Round-5 tests (cancellation-safe
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
