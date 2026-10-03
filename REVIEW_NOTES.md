# Review notes — notamusicbot_goku rework (branch rework/qwen3-tts), Round-8 return

HEAD at evidence time: **END_COMMIT (code) = `04a0a19`, full SHA `04a0a19dccdb4e950a607870583459d8f13c96ca`** — Round-8 review fixes
(§2 `slot()` post-semaphore closed re-check; §3 `close()` drains active
`slot()` holders; both pinned by deterministic tests) on top of the
Round-8 commit `9c5167ba72c99075e2fef211654ffa08d7747ac6` (detached
post-admission closed-state re-check; Discord-free `stop_guild_players`
in `discord_bot/player.py`, restoring the portable no-Discord unit gate).
These evidence reports are appended on top by the coordinator and are
intentionally uncommitted at packaging time. Round-7 response base:
START_COMMIT `f9673d75d0cf21bbde5a335fc72e8f52800d7a5b`. Historical
Round-7 commits (closing the Round-6 review; SHAs verified with
`git rev-parse`):

- `5de5770aed6235b43f3351b9b6be1c6db91bd3b4` — Round-6 §2 P0 RVC lifecycle
  single-flight: `_lifecycle_lock` serializes start/shutdown (lock order
  lifecycle → transaction, never reversed); failed-startup reaping (worker
  terminated, `_process`/`_loaded_key`/`_orphaned_ids` reset); concurrent
  start → exactly one spawn; shutdown during startup strictly follows
  startup.
- `cceb819a1ecdca6704fb92f8eb2c6ec471ca7939` — Round-6 §3 P0:
  `InferenceCoordinator.close()`/`drain()` — closed flag rejects new
  submissions with `CoordinatorClosedError`, cancels queued-not-started
  holders, awaits started holders to real completion;
  `VoiceConverterRouter` latches closed state: `convert()` after
  `shutdown()` raises `RouterClosedError` instead of lazily restarting
  RVC; `startup()` is the deliberate re-enable.
- `77ff4e7225338932f21f369189e6ae1654dfd637` — Round-6 §3 P0:
  `Application.shutdown()` order — cancel warmup → `coordinator.close()`
  (closed + drain) → `converter_router.shutdown()`; `bot.close()` stops
  all guild player tasks via `stop_guild_players` **BEFORE** converter
  teardown and `super().close()`.

## Current state (Round-9 return)

- Unit suite (main venv): **276 passed / 0 failed / 0 skipped** (274 + 2 new
  `slot()` lifecycle tests; re-run this session).
- Portable stripped env (no discord/qwen): collection succeeds, justified
  skips only (265 passed / 3 skipped in the review sandbox; 266/2 where the
  legacy RVC runtime is present).
- mypy: **0 errors in 48 files**. ruff check + format: clean.
- Evidence files: `gpu_arbitration.json`, `gpu_residency.json`,
  `rvc_cross_model_switching.json`, `rvc_same_model_warm.json`.
- Where to look: `git diff --stat f9673d7..04a0a19`.
- Live Discord voice playback: NOT_RUN (human VC test). Worker-cache future
  work (~6.3 s cross-model switch, ~0.6 s warm) unchanged.
- Round-6 items disposition (all closed in Round-7):

- §2 P0 RVC lifecycle races — **FAIL → PASS** (`5de5770`; concurrent-start
  spawn==1, shutdown-during-startup event-synced/deterministic/3× stable,
  failed-startup reaping tests)
- §3 P0 post-shutdown RVC restart — **FAIL → PASS** (`cceb819` + `77ff4e7`;
  review reproduced starts=2/shutdowns=1, now impossible: drain +
  `RouterClosedError`)
- §4 P1 evidence-document inconsistency — **FIXED** (this reconciliation:
  one coherent revision across all four evidence documents)
- §5 P2 stale counts — **FIXED** (current state 276 passed; 239/244/253/259/272
  only in dated per-round historical tables)
- §7.D GPU re-run — **NOT required / NOT_RUN by design**: lifecycle changes
  do not alter GPU scheduling or inference behavior, so the
  arbitration/residency/RVC-switching evidence from the `80e6146`/`43a39ef`
  era remains valid and was not re-run.
- Round-6 §8 shutdown-drain hardening — **IMPLEMENTED beyond the minimum**:
  coordinator drain + guild player enumeration were planned follow-up; both
  are now implemented and tested.

## Environment incident during Round-7 (resolved, stated honestly)

An accidental `uv run` overwrote the legacy RVC `.venv` with a Python 3.14
environment without torch, causing one real-worker test failure that was
wrongly called pre-existing. The venv was rebuilt from scratch via
`scripts/setup.sh` (the from-scratch build proven in Round 2) and the full
suite passes 272/272. Environment incident, not a product defect.

## Current test/evidence state (Round-7)

- Unit suite (main venv, this session): **272 passed / 0 failed / 0
  skipped, no deselects**. mypy: **0 errors in 48 files**. ruff check +
  format --check clean.
- Live voice-channel playback: **NOT_RUN** — needs a human in a Discord
  VC; left for the user (carried).

## Remaining known items (stated honestly)

1. **Live voice-channel playback NOT_RUN** — needs a human in a Discord
   VC; left for the user.
2. **Persistent per-model RVC worker cache not implemented** — future work.
   A cross-MODEL (Obama↔Goku) switch costs ~6.3 s (worker restart + model
   load) vs ~0.6 s warm same-model; variant switches **within** a model
   are in-process hot-swaps (`config_switched: true`, zero restarts).
3. **Credential rotation** (carried) — recommended for
   PLAY_HT/OPENAI/GEMINI-era keys because the first (pre-Gate-R1) review
   package left the dev machine; the leaked zip was replaced.
4. **Smaller content LLM** (carried, no code change required) — optional,
   config-only (`content.model_id`); residency evidence shows the 4B model
   is safe on the target GPU.

## Historical: Round-5 return record

HEAD at the Round-5 evidence time was
`a1dd397fc3f62f3c5695effd83bca7d5a44ad915`
(`fix(types): make mypy gate pass for src/notamusicbot (52 -> 0 errors)`);
Round-5 response base `80d213e` (Round-4 evidence reports). Round-5 commits
(closing the Round-4 review; SHAs verified with `git rev-parse`):

- `6e011bf` — P0 cancellation-safe GPU lease (Round-4 Option B):
  `InferenceCoordinator.run_blocking` — a detached holder acquires the
  semaphore **BEFORE** submitting to the executor and releases it only on
  real thread-future completion; caller cancellation returns immediately
  via a shielded await while the lease persists. Companion `run_detached`
  gives the same lease semantics for the pipeline's async chain —
  `SpeechPipeline.run` holds the lease across TTS→VC→postprocess including
  the RVC subprocess reply wait. LLM generate + app warmup use
  `run_blocking`. RVC adapter records orphaned request ids and discards
  late replies.
- `80e6146` — arbitration exclusivity asserted on **sampled lease-busy
  intervals**; ships `evidence/gpu_arbitration.json`.
- `43a39ef` — ships `evidence/gpu_residency.json` +
  `evidence/rvc_variant_switching.json`.
- `55b7e16` — P1 YouTube uploader parity (`PlayableMedia.artist` flows
  resolver→queue→`ContentRequest.artist_name`); P2 `setup.sh` parses the
  explicit `voice_conversion.variant` (`president_alt.yaml` fixture proves
  `obama_trans.yaml` is validated); stale `max_concurrent_generations`
  removed; runtime tests reorganized (command registration →
  `tests/integration`, live users.json proof →
  `tests/evidence/test_users_migration_evidence.py`); redundant
  `load_content_model` dropped from `bot.py setup_hook` since
  `app.startup()` loads the LLM **before** RVC start/warmup.
- `a1dd397` — mypy gate: **52 → 0 errors in 48 files**; targeted overrides
  only for untyped third-party (`yaml`, `yt_dlp`, `yt_dlp_ejs`, `spotipy`,
  `soundfile`, `soxr`, `qwen_tts`); exactly ONE targeted `arg-type` ignore
  with written justification; no behavior changes; `check_untyped_defs`
  kept.

## Design note — run_blocking vs run_detached

`run_blocking` is the stronger thread-level shield (used for LLM generation
and app warmup). `run_detached` provides the same lease semantics for the
pipeline's async chain. The whole-chain-in-thread variant deadlocked the
cancelled-job artifact test and was abandoned — a **documented deviation**;
the implementation still satisfies the review's Option B semantics: the
lease is owned by the coordinator and released only on real completion.
RVC cancellation retains the lease until the subprocess reply; orphaned
request ids are recorded and late replies discarded.

## Historical: Round-4 review items — disposition (all FIXED, Round-5)

- P0 cancellation safety — **FIXED** (tests A-LLM / A-TTS / A-RVC: the
  cancelled holder keeps the lease until the blocking work really ends)
- §3 startup ordering — **FIXED** (strict load→warmup ordering, test PASS)
- §4 uploader parity — **FIXED** (`PlayableMedia.artist` end-to-end)
- §5 serialization (RVC IPC + startup) — **FIXED** via the lease-held-chain
  design
- §6 mypy — **PASS** (0 errors in 48 files)
- §7 test reorg — **FIXED** (accurate stripped-env numbers: 244 passed /
  2 skipped, justified `discord` `importorskip`s)
- §8 evidence JSONs — **COMMITTED** (`gpu_arbitration`, `gpu_residency`,
  `rvc_variant_switching`)
- §9 `reply_turns`, §10 per-guild interstitials, §12 `.env.example` rule —
  already fixed in Round 4, verified unchanged
- Stale `max_concurrent_generations` — **REMOVED**

All gates **PASS** in the historical Round-5 disposition (R1–R8; R8's
Round-4 FAIL/PARTIAL was closed by the cancellation-safe lease +
sampled-interval exclusivity evidence; see the Round-7 gate table in
`IMPLEMENTATION_REPORT.md` for the current disposition). Historical unit
suite: **253 passed / 0 failed / 0 skipped, no deselects** at Round-5 (was
239 at the Round-3 return; current Round-7 count is 272). Real GPU runs
(RTX 4070 Ti SUPER, torch 2.14.1+cu130, driver 617.14, each run alone):
`test_gpu_arbitration` PASS (3 iterations, serialized, evidence JSON),
`test_gpu_residency` PASS (peak 9,854 MiB, no OOM), `test_rvc_switching`
2 PASS (4 switches, avg 6.23 s cross-model vs 0.82 s warm same-model at the
time; current files carry the Round-6 re-run numbers). The Round-4 review
§12 required-test satisfaction map (A–F) is in `TEST_RESULTS.md` — all
PASS.

## Remaining known items (stated honestly)

1. **Live voice-channel playback NOT_RUN** — needs a human in a Discord
   VC; left for the user.
2. **Persistent per-model RVC worker cache not implemented** — future work.
   A cross-MODEL (Obama↔Goku) switch costs ~6.23 s (worker restart + model
   load) vs ~0.82 s warm same-model; variant switches **within** a model
   are in-process hot-swaps (`config_switched: true`, zero restarts).
3. **Credential rotation** (carried) — recommended for
   PLAY_HT/OPENAI/GEMINI-era keys because the first (pre-Gate-R1) review
   package left the dev machine; the leaked zip was replaced.
4. **Smaller content LLM** (carried, no code change required) — optional,
   config-only (`content.model_id`); residency evidence shows the 4B model
   is safe on the target GPU.

## Where to look

- `IMPLEMENTATION_REPORT.md` — Round-7 commits, Round-6 defect
  cross-reference, current gate table, START_COMMIT/END_COMMIT, verbatim
  `git diff --stat f9673d7..77ff4e7`
- `TEST_RESULTS.md` — Round-7 per-check table + dated historical per-round
  tables + Round-4 §12 (A–F) required-test satisfaction map
- `evidence/gpu_arbitration.json`, `evidence/gpu_residency.json`,
  `evidence/rvc_cross_model_switching.json`,
  `evidence/rvc_same_model_warm.json` — raw real-GPU artifacts (current;
  `rvc_variant_switching.json` was deleted in Round-6)
- `BENCHMARK_RESULTS.md` — RTF interpretation (RTF > 1 = slower than real
  time)
- `DEPENDENCY_NOTES.md` — direct-deps table, Reddit correction,
  legacy-venv setup deps
