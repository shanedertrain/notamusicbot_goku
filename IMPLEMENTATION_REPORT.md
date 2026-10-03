# Implementation Report — notamusicbot rework (Qwen3-TTS), Round-4 re-review return

Rework of the legacy flat "notamusicbot" Discord bot onto a typed package
(`src/notamusicbot/`) with Qwen3-TTS as the speech backend and RVC voice
conversion behind an isolated worker boundary. This report covers the closure
of the Round-3 review return
(`NOTAMUSICBOT_GOKU_REWORK_REVIEW_RETURN_ROUND3.md`) — the Round-4 re-review
return.

## Commits (coordinator-verified)

- **START_COMMIT = `8a6f81a8b4f8fe5c5a39639e5f1fcf1fb1c66edf`** (legacy baseline)
- **END_COMMIT = `9d370fe8f8d47fc37cd8d245f6c77269993c7591`**
  (`feat: preserve RVC YAML variants end-to-end, shared GPU arbitration, unit isolation, per-guild interstitials, setup asset validation`)
- Round-3 review return anchor: `02d1d970b5bb3a897398d10a13e6058f9c0caf6b`
  (`02d1d97 docs: round-3 evidence reports`)

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

## Round-3 review defects — cross-reference (all FIXED)

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

## Round-2 review defects — cross-reference (all FIXED)

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

### Repository cleanliness (commands run verbatim at END_COMMIT)

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

## Gate status (Round-4 disposition of Round-3 gates; coordinator-verified, 2026-10-02)

| Gate | Result | Evidence |
|---|---|---|
| R1 package security | **PASS** | `scripts/package.sh` allowlist + secret scan; live `config/users.yaml` excluded; `users.example.yaml` ships. Round-4 `9d370fe` makes the `.env.example` exemption explicit (exact name allow, all other `.env*` rejected) and allowlists benchmark/evidence CSV/JSON (review §11/§12). |
| R2 production speech routing | **PASS** | Router through the real composition root; Round-4 `f804284` refines the lifecycle further with typed playback outcomes. |
| R3 migrated voices | **PASS** (variant identity now preserved) | Round-3 FAIL closed by `9d370fe`: `voice_conversion.variant` preserved end-to-end; worker selects the exact `<variant>.yaml`; migration asserts actual config identity with real config values (obama → `obama.yaml`, obama_trans → `obama_trans.yaml`, distinct parameters verified by `test_rvc_variant.py`). |
| R4 RVC persistence | **PASS** (same-model + variant switching) | Case-alias fix holds; cache key `(resolved_dir, variant)`; `test_rvc_switching` extended with real Obama-worker variant switching (obama → `pyht:Joseph` vs obama_trans → `gtts`) with `config_switched: true` **in-process — zero worker restarts on a variant switch**. Cross-MODEL switch cost remains 5.51 s avg (documented limitation). |
| R5 media lifecycle | **PASS** (crash orphans + full cleanup + cancellation) | `f804284`: production path uses the full cleanup helper (file + empty job dir); `.temporary` JSON markers persist ownership; `prune_expired_jobs` removes stale crash orphans after retention; explicit `CancelledError` handling; deterministic tests. |
| R6 content preservation | **PASS** (outcome-gated hook) | `f804284`: `PlaybackOutcome` StrEnum; hook fires only for STREAM + NATURAL, awaited before next dequeue; outcome matrix NATURAL→1, SKIPPED/STOPPED/FAILED/LOCAL→0; ordering guarantees unchanged (`put_front`). |
| R7 clean install | **PASS** (per-profile asset validation) | From-scratch legacy venv proof holds (`548b150`); Round-4 `9d370fe` adds `validate_rvc_assets` per enabled profile (model dir case-insensitive, variant yaml, `.pth`, `.index`; non-zero exit) — it caught the real `GOKU`/`Goku` case mismatch; 5 unit tests against temp fixtures. |
| R8 benchmark | **PASS** (shared arbitration proven non-overlapping on real GPU) | `9d370fe`: one `InferenceCoordinator` arbitrates LLM + TTS→VC→postprocess (RVC subprocess included); real GPU integration test, 3 iterations, concurrent tasks, all three models resident — fully serialized timeline, zero overlap. Raw evidence now included in packages. |

### Required Round-4 tests (review §14) — satisfaction map

See the "Round-3 review required tests" table in `TEST_RESULTS.md`: variant
preservation fixture tests, playback-outcome matrix, unit-suite isolation,
GPU arbitration, crash-orphan cleanup, and the silent-worker startup timeout
are all implemented and **PASS**.

### Gate R7 — required local assets (unchanged; setup exits 1 without them)

Model weights and per-model YAMLs are not distributed (gitignored). A fresh
clone needs locally supplied: `hubert_base.pt`, `rvc_cli/models/<Name>/*.pth`
+ `*.index`, and `rvc_cli/models/<Name>/*.yaml` (e.g. `goku.yaml`).
`scripts/setup.sh` prints this list and exits non-zero when assets are
missing. The Round-3 proof built the legacy venv itself from scratch — no
symlinked or reused dev venv.

## Architecture summary

Unchanged from the previous round: `src/notamusicbot/` typed package —
Discord/content → generic SpeechPipeline (synthesize → convert → postprocess)
→ Qwen3-TTS-0.6B in-process → RVC via isolated py3.10 worker (JSON IPC) →
guild-scoped queue player. Composition root `notamusicbot/app.py`; no
business-logic module imports `rvc_cli`.

## Test results summary

See `TEST_RESULTS.md` for the full table. Headline counts
(coordinator-verified, Round-4; re-run in this session): main-venv unit
**239 passed / 0 failed / 0 skipped**; `ruff check` and `ruff format --check`
clean. New/extended suites: `test_rvc_variant` (9), `test_coordinator` (3),
`test_interstitials_guild_scoping` (4), `test_setup_rvc_assets` (5),
`test_user_migration` (variant assertions), `test_player` (outcome matrix).
Integration: `test_gpu_arbitration` (real GPU, serialized timeline, PASS) and
`test_rvc_switching` extended with real variant switching (in-process
`config_switched`, zero restarts).

## Known limitations / follow-up (stated honestly)

- **Live voice-channel playback NOT_RUN** — needs a human in a VC; left for
  the user.
- **Persistent per-model RVC worker cache not implemented** — documented as
  future work; a cross-MODEL (Obama↔Goku) switch still costs ~5.5 s (worker
  restart + model load). Variant switches **within** a model are cheap
  in-process config hot-swaps with zero restarts.
- **Transient allocator spike** (historical, Round-3) — a ~15.9 GiB spike was
  observed during LLM load with concurrent external GPU load; with the shared
  coordinator, loading/inference are serialized (warmup included), removing
  the concurrent-inference concern. Steady state 11.2/16 GiB.
- Smaller content-LLM decision open (P2-18 follow-up), config-driven via
  `content.model_id`; residency evidence shows the 4B model is safe.
- Only the evidence reports (+ `REVIEW_NOTES.md`) are intentionally
  uncommitted; the coordinator commits them.
