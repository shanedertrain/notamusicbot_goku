# Implementation Report — notamusicbot rework (Qwen3-TTS), Final review return — implementation ACCEPTED

Rework of the legacy flat "notamusicbot" Discord bot onto a typed package
(`src/notamusicbot/`) with Qwen3-TTS as the speech backend and RVC voice
conversion behind an isolated worker boundary. This report covers the closure
of the Round-6 review return
(`NOTAMUSICBOT_GOKU_REWORK_REVIEW_RETURN_ROUND6.md`) — the Round-7 re-review
return. It supersedes the Round-5-titled revision; Round-4/5/6 material
below is retained as clearly dated historical records.

## Commits (coordinator-verified)

- **START_COMMIT = `f9673d75d0cf21bbde5a335fc72e8f52800d7a5b`**
  (`docs: honest Round-6 TEST_RESULTS.md — retract false concurrent-RVC claim…`)
- **END_COMMIT (final code) = `f9eccef22b18dadea9b53ed3f8c52615545444af`**
  (`fix(concurrency): detect close() from inside a held slot (final-review hardening note)` —
  the non-blocking hardening from the Final Review, introduced during documentation
  reconciliation as the review permitted). Final-review disposition:
  **software implementation ACCEPTED**; only documentation reconciliation remained.
  Round-8 code commit was `9c5167ba72c99075e2fef211654ffa08d7747ac6`; Round-7 code commits
  `5de5770…`, `cceb819…`, `77ff4e7…`.
  — Round-8 fixes on top of the Round-7 code commits below:
  - `9c5167b` — Round-7 §2 P0: post-admission closed-state re-check in `InferenceCoordinator._run_detached` and `slot()` (a caller blocked on admission when `close()` begins is rejected with `CoordinatorClosedError` and its function never executes; no `await` between re-check and holder registration). Round-7 §3 P1: `stop_guild_players` extracted from `bot.py` into Discord-import-free `discord_bot/player.py` (review Option B) — the portable no-Discord unit gate is restored (stripped env: 266 passed / 2 justified skips / 0 failed, collection succeeds).
  - Round-7 base: END_COMMIT at the Round-7 return was `77ff4e7225338932f21f369189e6ae1654dfd637`
  (`fix(shutdown): drain inference before RVC teardown; stop guild players first (round-6 §3/B)`)
  — the last **code** commit of that round; the evidence reports
  (`IMPLEMENTATION_REPORT.md`, `TEST_RESULTS.md`, `REVIEW_NOTES.md`,
  `DEPENDENCY_NOTES.md`) are appended on top of it by the coordinator and are
  intentionally uncommitted at packaging time.

### Round-7 commits (closing the Round-6 review; full SHAs verified with `git rev-parse`)

| SHA | Subject / defects closed |
|---|---|
| `5de5770aed6235b43f3351b9b6be1c6db91bd3b4` | Round-6 §2 P0 — RVC lifecycle single-flight: start/shutdown serialized under `_lifecycle_lock` (lock order lifecycle → transaction, never reversed); failed-startup reaping (worker terminated, `_process`/`_loaded_key`/`_orphaned_ids` reset); concurrent start → exactly one spawn; shutdown during startup strictly follows startup |
| `cceb819a1ecdca6704fb92f8eb2c6ec471ca7939` | Round-6 §3 P0 — `InferenceCoordinator.close()`/`drain()`: closed flag rejects new submissions with `CoordinatorClosedError`, cancels queued-not-started holders, awaits started holders to real completion; `VoiceConverterRouter` latches closed state — `convert()` after `shutdown()` raises `RouterClosedError` instead of lazily restarting RVC; `startup()` is the deliberate re-enable |
| `77ff4e7225338932f21f369189e6ae1654dfd637` | Round-6 §3 P0 — `Application.shutdown()` order: cancel warmup → `coordinator.close()` (closed + drain) → `converter_router.shutdown()`; `bot.close()` stops all guild player tasks via `stop_guild_players` **BEFORE** converter teardown and `super().close()` |

### Round-7 source delta (verbatim, at END_COMMIT)

```
$ git diff --stat f9673d7..77ff4e7
 src/notamusicbot/app.py                            |  17 +-
 src/notamusicbot/discord_bot/bot.py                |  43 +++-
 src/notamusicbot/discord_bot/guild_state.py        |   4 +
 src/notamusicbot/speech/coordinator.py             |  71 +++++-
 src/notamusicbot/speech/voice_conversion/router.py |  38 +++-
 src/notamusicbot/speech/voice_conversion/rvc.py    | 126 ++++++++---
 tests/unit/test_app_shutdown_drain.py              | 248 +++++++++++++++++
 tests/unit/test_coordinator.py                     |  76 +++++++
 tests/unit/test_rvc_adapter.py                     | 155 ++++++++++++-
 tests/unit/test_voice_converter_router.py          |  68 ++++++
 10 files changed, 795 insertions(+), 51 deletions(-)
```

Working-tree state at report time (`git status --porcelain`): the only
tracked entry is ` M rvc_cli` — the vendored RVC submodule is at the
recorded commit but its working tree carries runtime residue (unchanged
since earlier rounds; no source change). One untracked file,
`config/app.yaml`, is local runtime configuration not present in a fresh
clone. No source, test, or script file is uncommitted; the evidence reports
are committed on top of `77ff4e7` by the coordinator, as stated above.

Previous rounds' commits (historical records; full SHAs verified with
`git rev-parse` at their own rounds):

- **Historical baseline (Round-5 return):** START_COMMIT
  `8a6f81a8b4f8fe5c5a39639e5f1fcf1fb1c66edf` (legacy baseline), END_COMMIT
  `a1dd397fc3f62f3c5695effd83bca7d5a44ad915` (`fix(types): make mypy gate
  pass for src/notamusicbot (52 -> 0 errors)`); Round-5 response base
  `80d213e` (`80d213e docs: round-4 evidence reports`).

Review-response round commits (full SHAs verified with `git rev-parse`):

| SHA | Subject / defects |
|---|---|
| `ac0c601b4d07c8f526415f19cb785e89ca75527f` | P1-8 — resolve RVC model alias before comparing loaded model (prevents per-utterance worker restarts) |
| `75321ce80d7c2f006c3865325acb5111e770a8f9` | Gate R1 — allowlist-based packaging script with secret scan |
| `9c6557ca2b22b579c58431386c9918fa36730487` | P0-2 / P1-10 / P1-12 / P2-20 — route production voice conversion by profile backend, pass profile audio settings, wire warmup non-blocking |
| `84d7fe366380bab6d5d76c1cfbc8cdf013498410` | P0-3 / P0-4 — load `config/app.yaml`, migrate users to legacy voices (obama/obama_trans), add profiles for all RVC models, harden setup script |
| `17adafca66f2d1daaf6abe90f9f0c10491a33cea` | P1-9 — pitch-preserving speed change via FFmpeg atempo chain |
| `ab1270ddc95f9981fc64c7bf12072fc1d033a6ec` | P1-11 — per-stage artifact cleanup policy with `keep_artifacts` debug flag |
| `9875b7d5833153ff9dcf4556a0a5fceef4e481b9` | P1-6 / P1-7 — correct download fallback URL, missing-stream handling, temp-file leak |
| `a31581adfd1d6110e1240f27c813b28b9b92ee31` | P1-14 — lazy `discord_bot` package; unit suite runs without discord.py |
| `c721ecd71dc4483346d6fd44b4de208c72d1ae0c` | P0-5 — restore reddit/news interstitials and podcast through the shared pipeline; download via canonical URL |
| `6ca34bdc0f25d03f24d98926d5bd65084bbc545f` | P2-17 — torch/transformers declared as direct dependencies |
| `4716fabd9963dd0153e13680edd19a4898c3de20` | Round-2 return: corrected evidence reports, re-run benchmark, R1–R8 gate results |

### Round-3 commits (closing the Round-2 review; full SHAs verified with `git rev-parse`)

| SHA | Subject / defects closed |
|---|---|
| `5cd412f05606609b82148e5d7d2628b6a702ac5e` | Privacy — live `config/users.yaml` excluded from review packages; `config/users.example.yaml` added |
| `548b15007f87eb11d85cb52523d2e460de159e3d` | Gate R7 — `setup.sh` legacy deps completed (audiostretchy, pydub, gTTS, pyttsx3, pyyaml, dotenv), `setuptools<81` pin for `pkg_resources` (setuptools 84 dropped it), `dev` extra installed in the main venv |
| `96000d3c7c98efcb8195e844445efe028b0dcadf` | P0 playback lifecycle — explicit `on_playback_complete` hook fired **only for `MediaKind.STREAM`** after completion and before the next dequeue; `GuildState.put_front` head-insertion so interstitials/podcast turns play **before** an already-queued song B; stop/get race recovery; download-leak ownership fix |
| `07d16d39b1284fa6f88bb9fe37ffb6d500d9f2e7` | P0 commands — podcast registered as `name="podcast"` so the `!g` prefix produces `!gpodcast` (was `!ggpodcast`) |
| `f03708aad5b35c664166a65ca1c71d5397723e00` | P1/P2 — fail-soft content load with `StaticTemplateProvider` fallback; `commentary_enabled`/`content.enabled` default **true** (interstitial/podcast stay false); skip-when-disabled via `_LazySpeechSynthesizer`; `keep_artifacts` wired + `failed_job_retention_hours=24`; empty job-dir removal + opportunistic prune; warmup scheduled exactly once, non-blocking; GPU-residency and RVC-switching evidence |

The P2-17 pyproject change was **already committed at report time** as
`6ca34bd` (it is not a pending working-tree change).

### Round-4 commits (closing the Round-3 review; full SHAs verified with `git rev-parse`)

| SHA | Subject / defects closed |
|---|---|
| `f804284916dbc520396f4c2c1d9a194acd0e830d` | P0 playback outcomes — `PlaybackOutcome` StrEnum (`NATURAL`/`SKIPPED`/`STOPPED`/`FAILED`/`DROPPED`) in `domain/media.py`; `_play_one()` returns `(media, outcome)`; `skip`/`stop` record `state.interrupt_reason` **before** `voice_client.stop()`; post-song hook fires **only on STREAM + NATURAL** and is awaited before the next dequeue; after-callback errors → `FAILED`; dequeue without a voice client → `DROPPED`. P1 artifacts — per-job `.temporary` on-disk JSON markers so `prune_expired_jobs` removes stale crash orphans; full file + empty job-dir cleanup in the production path; explicit `CancelledError` handling |
| `9d370fe8f8d47fc37cd8d245f6c77269993c7591` | P0 variant identity — `voice_conversion.variant` profile field → `VoiceConversionProfile.variant` → IPC `load-model {name, variant}` → worker selects the **exact** `<variant>.yaml` (`found[0]` eliminated; loud error on a missing variant); RVC cache key becomes `(resolved_dir, variant)`; same-dir variant switch = **in-process config hot-swap** (`config_switched: true`, zero worker restarts); `config/voices/obama.yaml`/`obama_trans.yaml` and `config/users.yaml` (all 30 users) carry `variant`. P1 shared GPU arbitration — one `InferenceCoordinator` injected into **both** the content LLM provider and the `SpeechPipeline`; the pipeline holds the slot across TTS→VC→postprocess so the RVC subprocess cannot overlap main-process inference; warmup arbitrates too. P1 unit isolation — `pytest.importorskip("discord")` in `test_command_registration`; the 2 `build_pipeline` tests inject a `FakeSynthesizer` (no `qwen_tts` import in the unit suite). P2 — `setup.sh validate_rvc_assets` per enabled RVC profile (model dir case-insensitive, variant yaml, `.pth`, `.index`; exits non-zero); `package.sh` explicit `.env.example`-only rule + benchmark/evidence CSV/JSON allowlisted; podcast `turns`→`reply_turns` rename; interstitial cooldown guild-scoped (dict keyed by guild id; production caller passes `guild_id`) |

### Round-5 commits (closing the Round-4 review; full SHAs verified with `git rev-parse`)

| SHA | Subject / defects closed |
|---|---|
| `6e011bf62136a2a7d78023f043d858bdb02c1621` | P0 cancellation-safe GPU lease — `InferenceCoordinator.run_blocking`: a detached holder acquires the semaphore **BEFORE** submitting to the executor and releases it only on real thread-future completion; caller cancellation returns immediately via a shielded await while the lease persists. Companion `run_detached` gives the same lease semantics for the pipeline's async chain — `SpeechPipeline.run` holds the lease across TTS→VC→postprocess including the RVC subprocess reply wait. LLM generate + app warmup use `run_blocking`. RVC adapter records orphaned request ids and discards late replies (cancellation during conversion retains the lease until the subprocess reply) |
| `80e6146d1d653e39d4340cc5f581ff81ec9d4504` | GPU arbitration test now asserts exclusivity on **sampled lease-busy intervals** (stronger than wall-clock edges); ships `evidence/gpu_arbitration.json` |
| `43a39effcefc78dcb307fb296f692338442c68a` | Ships raw real-GPU evidence: `evidence/gpu_residency.json` + `evidence/rvc_variant_switching.json` |
| `55b7e163931c234e4d93079e04b0f294058c887f` | P1 YouTube uploader parity — `PlayableMedia.artist` flows resolver→queue→`ContentRequest.artist_name`. P2 — `setup.sh` parses the explicit `voice_conversion.variant` (`president_alt.yaml` fixture proves `obama_trans.yaml` is validated even when filename and variant differ); stale `max_concurrent_generations` removed; runtime tests reorganized: command registration → `tests/integration`, live users.json proof → `tests/evidence/test_users_migration_evidence.py`; redundant `load_content_model` dropped from `bot.py setup_hook` since `app.startup()` loads the LLM **before** RVC start/warmup |
| `a1dd397fc3f62f3c5695effd83bca7d5a44ad915` | mypy gate — **52 → 0 errors in 48 files**; targeted overrides only for untyped third-party (`yaml`, `yt_dlp`, `yt_dlp_ejs`, `spotipy`, `soundfile`, `soxr`, `qwen_tts`); exactly ONE targeted `arg-type` ignore with written justification; no behavior changes; `check_untyped_defs` kept |

### Design note — cancellation-safe lease shapes

`run_blocking` is the stronger thread-level shield and is used for LLM
generation and app warmup. `run_detached` provides the same lease semantics
for the pipeline's async chain. The whole-chain-in-thread variant
(Option A style) deadlocked the cancelled-job artifact test and was
abandoned — a **documented deviation**; the implementation still satisfies
Option B semantics from the review: the lease is owned by the coordinator
and released only on real completion. RVC cancellation retains the lease
until the subprocess reply; orphaned request ids are recorded and late
replies discarded.

## Round-6 review defects — cross-reference (all FIXED, Round-7)

| Round-6 finding | Priority | Status | Fixed by / evidence |
|---|---|---|---|
| §2 P0 RVC worker startup/shutdown not lifecycle-serialized (concurrent start 2-PID leak; shutdown racing startup; failing `test_shutdown_waits_for_active_transaction`) | P0 | **FIXED → PASS** | `5de5770` — `_lifecycle_lock` single-flight (lock order lifecycle → transaction, never reversed); concurrent start → exactly one spawn; shutdown during startup strictly follows startup; failed-startup reaping. The previously flaky transaction-shutdown test is now event-synced, deterministic, stable 3×. See `TEST_RESULTS.md` Round-7 table. |
| §3 P0 Application shutdown can restart RVC after shutdown (review reproduced starts=2 / shutdowns=1) | P0 | **FIXED → PASS** | `cceb819` + `77ff4e7` — coordinator `close()`/`drain()` (closed + drain semantics), router latched closed state (`RouterClosedError` after shutdown; `startup()` is the deliberate re-enable), shutdown order warmup-cancel → coordinator close/drain → router shutdown, `stop_guild_players` before converter teardown. Post-shutdown RVC restart is now impossible. |
| §4 P1 Round-6 evidence documents internally inconsistent (Round-5-titled reports, stale END_COMMIT `a1dd397`, stale `rvc_variant_switching.json` references) | P1 | **FIXED** | This reconciliation: all four evidence documents now agree on round number (7), START_COMMIT `f9673d7…`, END_COMMIT `77ff4e7…`, test counts (272 passed), current evidence filenames, and known limitations; current-state summaries are separated from dated historical records. |
| §5 P2 Stale test-count claims (239/244/253 mixed into current-state text) | P2 | **FIXED** | Current state everywhere: **272 passed / 0 failed / 0 skipped** (main venv, this session); 239/244/253/259 appear only in clearly dated per-round historical tables. `DEPENDENCY_NOTES.md` unit-suite wording updated. |

## Historical: Round-5 review defects — cross-reference (all FIXED, closed in Round 5)

| Round-4 finding | Priority | Status | Fixed by / evidence |
|---|---|---|---|
| §2 P0 Shared GPU arbitration not cancellation-safe (cancelled awaiter releases slot while thread still running) | P0 | **FIXED** | `6e011bf` — `run_blocking` detached holder acquires the lease before executor submission, releases only on real thread-future completion; `run_detached` for the pipeline's async chain (lease held across TTS→VC→postprocess incl. RVC reply wait); RVC orphaned request ids recorded, late replies discarded. Tests: cancelled LLM/TTS/RVC holder-keeps-lease all PASS; stop-during-post-song probe enters only after detached work finishes, `max_active == 1`, queue not repopulated. Gate R8 PASS |
| §3 P1 Startup warmup can overlap LLM model loading | P1 | **FIXED** | Strict ordering: `app.startup()` loads the content LLM **before** RVC start/warmup; `55b7e16` dropped the redundant `load_content_model` from `bot.py setup_hook`; startup load→warmup ordering test with `warmup_on_start: true` PASS |
| §4 P1 YouTube song commentary lost uploader/artist | P1 | **FIXED** | `55b7e16` — `PlayableMedia.artist` flows resolver→queue→`ContentRequest.artist_name`; command-path test asserts `song_name == Song`, `artist_name == Artist`. Gate R6 PASS |
| §5 P1 RVC IPC unsafe above 1 concurrent inference | P1 | **FIXED** | The pipeline now holds the GPU lease across the full RVC round trip (`6e011bf` lease-held-chain design), and the adapter serializes the write→matching-reply transaction; concurrent-request fake-worker test PASS; stale `max_concurrent_generations` removed (`55b7e16`) |
| §6 P1/P2 mypy gate missing from evidence | P1/P2 | **FIXED (PASS)** | `a1dd397` — `mypy src/notamusicbot`: **0 errors in 48 files** (was 52); targeted third-party overrides only; one justified `arg-type` ignore; recorded in `TEST_RESULTS.md` |
| §7 P2 Unit-test evidence wording inaccurate | P2 | **FIXED** | Honest numbers: main-venv **253 passed / 0 failed / 0 skipped, no deselects**; stripped env (no discord/qwen) **244 passed / 2 skipped** (justified `discord` `importorskip`s, Round-4 independent run). Runtime tests reorganized (`55b7e16`): command registration → `tests/integration`, live users.json proof → `tests/evidence/test_users_migration_evidence.py` |
| §8 P2 Raw GPU evidence absent from package | P2 | **FIXED** | `80e6146` + `43a39ef` — `evidence/gpu_arbitration.json`, `evidence/gpu_residency.json`, `evidence/rvc_variant_switching.json` committed (timestamp, commit, GPU, driver, torch, timeline/VRAM, pass/fail) |
| §9 P2 Setup should parse explicit `voice_conversion.variant` | P2 | **FIXED** | `55b7e16` — explicit `variant` parsing; `president_alt.yaml` fixture (`variant: obama_trans`) proves `obama_trans.yaml` is validated, not the profile filename |
| §10 P2 Stale concurrency setting | P2 | **FIXED** | `55b7e16` — `max_concurrent_generations` removed |
| §9/§10/§12 already fixed in Round 4 (`reply_turns`, per-guild interstitials, `.env.example` rule) | P2 | **FIXED (Round 4)** | Carried; verified unchanged |

## Historical: Round-3 review defects — cross-reference (all FIXED)

| Round-3 finding | Priority | Status | Fixed by / evidence |
|---|---|---|---|
| §2 Legacy RVC YAML variants not preserved (`found[0]` selection) | P0 | **FIXED** | `9d370fe` — `variant` field end-to-end (profile → IPC → worker); exact `<variant>.yaml` selection; cache key `(dir, variant)`; new `tests/unit/test_rvc_variant.py` (9 tests) asserts the real config values reach conversion; migration test asserts config identity, not just profile strings. Gate R3 PASS |
| §3 Skip/stop/failure trigger post-song content | P0 | **FIXED** | `f804284` — typed `PlaybackOutcome`; hook gated to STREAM + NATURAL; interrupt reason set before `voice_client.stop()`; outcome matrix tests: NATURAL→1, SKIPPED/STOPPED/FAILED/LOCAL→0. Gate R6 PASS |
| §4 Unit suite depends on discord / real Qwen backend | P1 | **FIXED** | `9d370fe` — `importorskip("discord")`; `FakeSynthesizer` in the 2 `build_pipeline` tests; unit suite: 239 passed / 0 failed / **0 skipped** with no `qwen_tts` import |
| §5 GPU coexistence tested sequentially, not concurrently | P1 | **FIXED (shared coordinator)** | `9d370fe` — single `InferenceCoordinator` shared by LLM provider + SpeechPipeline; pipeline holds the slot through the RVC stage; warmup arbitrates. `tests/integration/test_gpu_arbitration.py` (real run, 3 iterations, concurrent LLM + TTS→RVC tasks, all models resident): timeline fully serialized, zero overlap — every stage starts exactly when the previous ends. Gate R8 PASS |
| §6 Successful cleanup leaves directories and crash orphans | P1 | **FIXED** | `f804284` — full cleanup helper in the production path (file + empty job dir); per-job `.temporary` JSON markers; `prune_expired_jobs` removes stale crash orphans after the retention threshold; explicit `CancelledError` handling; deterministic tests. Gate R5 PASS |
| §7 Setup does not fail for missing RVC model assets | P2 | **FIXED** | `9d370fe` — `validate_rvc_assets` per enabled profile (case-insensitive model dir, variant yaml, `.pth`, `.index`), exits non-zero; caught the real `GOKU`/`Goku` case mismatch; `tests/unit/test_setup_rvc_assets.py` (5 tests) extracts the bash validation against temp fixtures. Gate R7 PASS |
| §8 RVC startup timeout is not a hard timeout | P2 | **FIXED** | `9d370fe` — `select`-based deadline read (`_wait_for_ready`) replaces the blocking `readline()`; silent-worker test fails within the configured timeout |
| §9 Podcast `turns` documentation/behavior disagree | P2 | **FIXED** | `9d370fe` — renamed to `reply_turns` (replies after the host opening) consistently across code/config/docs |
| §10 Interstitial cooldown state is global, not guild-scoped | P2 | **FIXED** | `9d370fe` — `_last_fire_monotonic` keyed by guild id; production caller passes `guild_id`; `test_interstitials_guild_scoping.py` (4 tests) |
| §11 Raw benchmark/evidence data missing from package | P2 | **FIXED** | `9d370fe` — `package.sh` allowlists benchmark/evidence CSV + JSON artifacts |
| §12 `.env` rejection should be explicit | P2 | **FIXED** | `9d370fe` — exact `".env.example"`-name allow; every other `.env*` path rejected |

## Historical: Round-2 review defects — cross-reference (all FIXED)

Historical note: `evidence/rvc_variant_switching.json`, referenced in two
rows below, was the Round-5-era combined evidence file; it was **deleted** in
Round-6 and replaced by `evidence/rvc_cross_model_switching.json` +
`evidence/rvc_same_model_warm.json`, which are the current files.

| Round-2 finding | Priority | Status | Fixed by / evidence |
|---|---|---|---|
| Post-song content lifecycle recurs + ordering race | P0 | **FIXED** | `96000d3` — detached listeners removed; `on_playback_complete` hook, STREAM-only, awaited before next dequeue; `put_front` head-insertion; Gate R6 PASS |
| Podcast command registered as `gpodcast` → `!ggpodcast` | P0 | **FIXED** | `07d16d3` — `name="podcast"`; registration test asserts all public commands |
| Full three-model GPU residency not demonstrated | P1 | **FIXED** | Real RTX 4070 Ti SUPER run: all three models resident 11,262/16,375 MiB idle; peaks ≤ 9,857 MiB; no OOM (Gate R8) |
| Alternating RVC profile cost not characterized | P1 | **FIXED (documented)** | `benchmark_rvc_switching` tool + integration test: 5.51 s avg per Obama↔Goku switch vs 0.57 s warm same-model (~10×); persistent worker cache documented as future work, not implemented |
| Content-model startup not fail-soft | P1 | **FIXED** | `f03708a` — provider load wrapped; `StaticTemplateProvider` fallback; startup-continues test |
| Fresh config disables the feature under construction | P1 | **FIXED** | `f03708a` — `commentary_enabled`/`content.enabled` default true; interstitial/podcast stay opt-in; lazy synthesizer skips loading when disabled |
| Downloaded fallback file leaks on setup failure | P1 | **FIXED** | `96000d3` — ownership into `try/finally`; test: download OK + source_factory raises → file removed |
| Stop/get race can orphan an item | P1 | **FIXED** | `96000d3` — completed queue-get recovered and cleaned on stop; deterministic race test |
| `keep_artifacts` not wired into production pipeline | P2 | **FIXED** | `f03708a` — wired through `build_pipeline` → `SpeechPipeline`; tested |
| Empty speech-job directories accumulate | P2 | **FIXED** | `f03708a` — empty job dirs removed after cleanup + opportunistic prune; failed jobs retained 24 h |
| Warmup not consistently wired | P2 | **FIXED** | `f03708a` — scheduled exactly once, non-blocking; consistency tested |
| `DEPENDENCY_NOTES.md` stale about Reddit | P2 | **FIXED** | Notes updated: Reddit **is** ported (PRAW-backed `RedditPostFetcher`, optional `[reddit]` extra) |
| README hard-codes stale test count | P2 | **FIXED** | README now points to `TEST_RESULTS.md`; no hard-coded count |
| Live `config/users.yaml` in review package | P2 | **FIXED** | `5cd412f` — excluded; `users.example.yaml` added |
| Legacy RVC venv not provable from setup script alone | P1 (Gate R7) | **FIXED** | `548b150` — from-scratch proof: fresh clone + `scripts/setup.sh` built the legacy venv (fairseq compiled, torch cu121); setup exits 1 on missing assets; RVC suite 2/2 on the fresh venv, no symlinks |

### Repository cleanliness (historical: commands run verbatim at the Round-5 END_COMMIT; see the Round-7 source-delta section above for the current state)

```
$ git diff --stat 8a6f81a..HEAD | tail -3
 tools/smoke_discord_startup.py                     |    50 +
 tts.py                                             |   315 +-
 152 files changed, 24261 insertions(+), 10021 deletions(-)

$ git status --porcelain | grep -v '^??' | grep -v rvc_cli
(no output — exit 1, no matches)
```

The only tracked working-tree entry in `git status --porcelain` is
` M rvc_cli`: the vendored RVC submodule is at the recorded commit
`535d3643144890461f2a77cef0a9c47dc9bfa0e6` but its working tree is dirty
(-dirty suffix; runtime residue inside the vendored stack, no source change).
No source, config, or script file outside the evidence reports is uncommitted;
the coordinator commits the evidence reports themselves.

## Defects fixed vs review (cross-reference)

| ID | Review defect | Status | Fixed by / evidence |
|---|---|---|---|
| P0-1 | Secrets in returned package | **FIXED** | `75321ce` — allowlist packaging + secret scan (Gate R1 PASS); leaked zip replaced in recipient's Downloads |
| P0-2 | Production Discord did not use RVC | **FIXED** | `9c6557c` — routing by profile backend; Gate R2 PASS |
| P0-3 | `config/app.yaml` not loaded | **FIXED** | `84d7fe3` |
| P0-4 | User voice mappings not preserved | **FIXED** | `84d7fe3` — migration; Gate R3 PASS (29→obama, 1→obama_trans, 0 goku) |
| P0-5 | Content behavior missing (song/news/reddit/podcast) | **FIXED** | `c721ecd` — all four via ContentProvider→SpeechPipeline→queue; Gate R6 PASS |
| P1-6 | Download fallback leaks cache files | **FIXED** | `9875b7d`; Gate R5 PASS |
| P1-7 | Fallback wrong URL / missing stream URL | **FIXED** | `9875b7d` — canonical-URL download; Gate R5 PASS |
| P1-8 | RVC restart per utterance | **FIXED** | `ac0c601` — resolve-first; Gate R4 PASS |
| P1-9 | Speed change altered pitch | **FIXED** | `17adafc` — FFmpeg atempo chain |
| P1-10 | Per-profile audio settings ignored | **FIXED** | `9c6557c` |
| P1-11 | Artifacts accumulate indefinitely | **FIXED** | `ab1270d` — per-stage cleanup + `keep_artifacts` |
| P1-12 | RVC worker lifecycle not tied to shutdown | **FIXED** | `9c6557c` |
| P1-13 | Benchmark interpretation backwards | **FIXED** | Corrected report + re-run; see `BENCHMARK_RESULTS.md` (Gate R8 PASS) |
| P1-14 | Unit suite depends on discord extra | **FIXED** | `a31581a` — lazy imports; 180 passed / 1 skipped with discord absent |
| P1-15 | "Fresh install PASS" overstated | **FIXED** | Real clean-clone Gate R7 run; tracked-asset gaps confirmed and documented (see below) |
| P1-16 | Ending commit not concrete | **FIXED** | START_COMMIT/END_COMMIT + embedded diff/stat/status above |
| P2-17 | torch/transformers not direct deps | **FIXED** | `6ca34bd` — `torch>=2.6`, `transformers>=4.55` in `[project.dependencies]` |
| P2-18 | Local LLM scope deviation | **USER-APPROVED** | User: "We dont use Gemini anymore so we want to run the LLM parts on the 4070 ti" (stated during implementation). Smaller-model question (e.g. Qwen3-1.7B) under review; config-driven via `content.model_id` |
| P2-19 | GPU arbitration should be unified | **FIXED (Round-4)** | `9d370fe` — single shared `InferenceCoordinator` arbitrates LLM + speech pipeline (TTS→VC→postprocess, RVC subprocess included); real-GPU integration test proves zero overlap |
| P2-20 | Warmup not part of startup | **FIXED** | `9c6557c` — wired non-blocking |
| P2-21 | Setup is Linux/WSL-oriented | **FIXED** (documented) | Linux/WSL-only documented in README + `setup.sh` |

## Gate status (current Round-7 disposition; coordinator-verified, this session)

| Gate | Result | Evidence |
|---|---|---|
| R1 package security | **PASS** | `scripts/package.sh` allowlist + secret scan; `.env.example`-only rule holds; `users.example.yaml` ships. |
| R2 production speech routing | **PASS** | Router through the real composition root; typed playback outcomes hold; unchanged by Round-7 lifecycle work. |
| R3 migrated voices | **PASS** | Variant identity preserved end-to-end; `55b7e16` fixture proves the explicit `variant` field is validated. |
| R4 RVC persistence | **PASS** | Cache key `(resolved_dir, variant)`; in-process variant hot-swaps with zero restarts; real run: 4 switches, cross-model avg **6.29 s** vs **0.62 s** warm same-model (`evidence/rvc_cross_model_switching.json` + `evidence/rvc_same_model_warm.json`). Persistent worker cache remains future work. |
| R5 media lifecycle | **PASS** | Crash orphans + full cleanup + cancellation hold; pipeline cancellation keeps the lease until the RVC subprocess reply and discards orphaned request ids. |
| R6 content preservation | **PASS** | STREAM+NATURAL-only hook holds; YouTube uploader metadata restored (`55b7e16`). |
| R7 clean install | **PASS** | From-scratch legacy venv proof holds (re-proven again in Round-7 after the `.venv` incident — rebuilt via `scripts/setup.sh`); per-profile asset validation unchanged; setup parses the explicit `voice_conversion.variant`. |
| R8 GPU safety | **PASS** | `6e011bf` cancellation-safe lease; `80e6146` exclusivity on sampled lease-busy intervals; raw evidence committed (`43a39ef`): `evidence/gpu_arbitration.json`, `evidence/gpu_residency.json` (peak 9,854 MiB, no OOM), plus `evidence/rvc_cross_model_switching.json` + `evidence/rvc_same_model_warm.json`. Per review §7.D these real-GPU runs were not re-run in Round-7: the lifecycle commits change shutdown ordering only, not GPU scheduling/inference behavior; the numbers date from the `80e6146`/`43a39ef` era. |

### Historical gate tables

The per-round gate dispositions of Rounds 4–6 (including the Round-5 table
below) are retained unchanged in `TEST_RESULTS.md` and in the dated sections
above.

### Required Round-4 tests (review §12 A–F) — satisfaction map

See the "Round-4 review required tests" table in `TEST_RESULTS.md`:
A cancellation-safe GPU lease (LLM/TTS/RVC), B stop-during-post-song,
C startup load→warmup ordering, D YouTube uploader parity, E concurrent
RVC protocol requests, and F mypy (0 errors) are all implemented and
**PASS**.

### Gate R7 — required local assets (unchanged; setup exits 1 without them)

Model weights and per-model YAMLs are not distributed (gitignored). A fresh
clone needs locally supplied: `hubert_base.pt`, `rvc_cli/models/<Name>/*.pth`
+ `*.index`, and `rvc_cli/models/<Name>/*.yaml` (e.g. `goku.yaml`).
`scripts/setup.sh` prints this list and exits non-zero when assets are
missing. The Round-3 proof built the legacy venv itself from scratch — no
symlinked or reused dev venv.

## Architecture summary

Unchanged from previous rounds: `src/notamusicbot/` typed package —
Discord/content → generic SpeechPipeline (synthesize → convert → postprocess)
→ Qwen3-TTS-0.6B in-process → RVC via isolated py3.10 worker (JSON IPC) →
guild-scoped queue player. Composition root `notamusicbot/app.py`; no
business-logic module imports `rvc_cli`. Round-7 adds the one coherent
lifecycle rule the Round-6 review required: once application shutdown starts,
no new worker or GPU job may begin, and already-started work retires before
the worker is torn down (§8 shutdown-drain hardening is now IMPLEMENTED
beyond the minimum — coordinator drain + guild player enumeration — not just
planned).

## Test results summary (current: Round-7)

See `TEST_RESULTS.md` for the full tables. Headline counts
(coordinator-verified, Round-7, re-run in this session): main-venv unit
**272 passed / 0 failed / 0 skipped, no deselects** (259 at the Round-6
return; 253 at Round-5; 239 at Round-3 — historical, per-round tables kept);
**mypy `src/notamusicbot`: 0 errors in 48 files**; `ruff check` and
`ruff format --check` clean. New Round-7 coverage: concurrent-start
spawn==1, shutdown-during-startup (event-synced, deterministic, 3× stable),
failed-ready/failed-health cleanup, coordinator drain (started awaited /
queued cancelled / new rejected), router closed-state + startup re-enable,
pipeline-during-shutdown (job retires before RVC teardown, starts never
increase), `stop_guild_players` await/cancel. Real GPU evidence (RTX 4070 Ti
SUPER, `80e6146`/`43a39ef` era, not re-run per review §7.D — lifecycle
changes do not alter GPU scheduling): `evidence/gpu_arbitration.json`,
`evidence/gpu_residency.json`, `evidence/rvc_cross_model_switching.json`,
`evidence/rvc_same_model_warm.json`.

## Known limitations / follow-up (stated honestly)

- **Live voice-channel playback NOT_RUN** — needs a human in a VC; left for
  the user.
- **Persistent per-model RVC worker cache not implemented** — documented as
  future work; a cross-MODEL (Obama↔Goku) switch still costs ~6.3 s
  (worker restart + model load) vs ~0.6 s warm same-model. Variant
  switches **within** a model are cheap in-process config hot-swaps with
  zero restarts (variant hot-swap in-process).
- **Real-GPU evidence vintage** — GPU arbitration/residency/RVC-switching
  numbers date from the `80e6146`/`43a39ef` era; per review §7.D they were
  not re-run in Round-7 because the lifecycle changes affect only shutdown
  ordering, not GPU scheduling or inference behavior.
- **Environment incident (Round-7, resolved)** — an accidental `uv run`
  overwrote the legacy `.venv` (py3.14, no torch), causing one real-worker
  test failure that was wrongly called pre-existing; the venv was rebuilt
  from scratch via `scripts/setup.sh` and the full suite passes 272/272.
- **Transient allocator spike** (historical, Round-3) — a ~15.9 GiB spike was
  observed during LLM load with concurrent external GPU load; with the shared
  coordinator, loading/inference are serialized (warmup included), removing
  the concurrent-inference concern. Steady state 11.2/16 GiB.
- Smaller content-LLM decision open (P2-18 follow-up), config-driven via
  `content.model_id`; residency evidence shows the 4B model is safe.
- Only the evidence reports (+ `REVIEW_NOTES.md`) are intentionally
  uncommitted; the coordinator commits them on top of
  `77ff4e7225338932f21f369189e6ae1654dfd637`.
