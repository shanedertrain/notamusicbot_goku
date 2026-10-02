# Implementation Report — notamusicbot rework (Qwen3-TTS), Round-3 re-review return

Rework of the legacy flat "notamusicbot" Discord bot onto a typed package
(`src/notamusicbot/`) with Qwen3-TTS as the speech backend and RVC voice
conversion behind an isolated worker boundary. This report covers the closure
of the Round-2 review return
(`NOTAMUSICBOT_GOKU_REWORK_REVIEW_RETURN_ROUND2.md`) — the Round-3 re-review
return.

## Commits (coordinator-verified)

- **START_COMMIT = `8a6f81a8b4f8fe5c5a39639e5f1fcf1fb1c66edf`** (legacy baseline)
- **END_COMMIT = `f03708aad5b35c664166a65ca1c71d5397723e00`**
  (`feat: fail-soft content load, default-enable commentary, artifact retention policy, GPU residency + RVC switching evidence`)
- Round-2 review return anchor: `4716fabd9963dd0153e13680edd19a4898c3de20`

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
| P2-19 | GPU arbitration should be unified | **ACCEPTED-LIMITATION** | Separate semaphores remain, documented; load-order peak ~15.9 GiB observed once with concurrent external GPU load; steady state 11.2/16 GiB; monitored |
| P2-20 | Warmup not part of startup | **FIXED** | `9c6557c` — wired non-blocking |
| P2-21 | Setup is Linux/WSL-oriented | **FIXED** (documented) | Linux/WSL-only documented in README + `setup.sh` |

## Gate status (Round-3 disposition of Round-2 gates; coordinator-verified, 2026-10-02)

| Gate | Result | Evidence |
|---|---|---|
| R1 package security | **PASS** (+users.yaml privacy fix) | `scripts/package.sh` allowlist + secret scan; Round-3 `5cd412f` excludes live `config/users.yaml` from packages and adds `config/users.example.yaml`. Credential rotation still recommended for the PLAY_HT/OPENAI/GEMINI-era keys. |
| R2 production speech routing | **PASS** (architecture + lifecycle fix) | Router through the real composition root (`tests/unit/test_app_production_routing.py`); Round-3 `96000d3` fixes when the production queue triggers generated content. |
| R3 migrated voices | **PASS** | 29 users → obama, 1 → obama_trans via `legacy_alias`; 0 goku; unknown model exits non-zero. |
| R4 RVC persistence | **PASS** (+alternating-cost evidence) | Resolve-first fix prevents same-model restarts; new alternating benchmark: 5.51 s avg per Obama↔Goku switch vs 0.57 s warm same-model. |
| R5 media lifecycle | **PASS** (failure-path + stop-race tests) | Download-leak ownership test and deterministic stop-race test added in `96000d3`. |
| R6 content preservation | **PASS** (recursion impossible, ordering guaranteed, command fixed) | Hook fires only for `MediaKind.STREAM`; `put_front` guarantees song A → segment → song B; ordering test shows zero hook calls for generated items and no leftover tasks; `07d16d3` fixes the podcast command. |
| R7 clean install | **PASS** (full from-scratch legacy venv via `setup.sh`) | Fresh clone + `scripts/setup.sh` built the legacy RVC venv from scratch (fairseq compiled, torch cu121); setup exits 1 on missing assets; with `hubert_base.pt` + `rvc_cli/models/Goku/goku.yaml` supplied, RVC integration passed 2/2 on the fresh venv — no symlinks. Setup defects found and fixed in `548b150`. |
| R8 benchmark | **PASS** (three-model residency + corrected semantics) | Corrected RTF interpretation + `benchmark/20261002T130047Z`; real GPU residency run (see TEST_RESULTS.md); alternating-profile benchmark added. |

### Required Round-3 tests (review §9) — satisfaction map

See the "Round-2 review required tests" table in `TEST_RESULTS.md`: every
required test (playback ordering, command surface, temp-file failure cleanup,
stop race, provider startup fallback, full GPU residency, alternating RVC
profiles) is implemented and **PASS**.

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
(coordinator-verified, Round-3): main-venv unit **204 passed / 0 failed**;
`ruff check` and `ruff format --check` clean. RVC integration 2 passed; RVC
switching benchmark: 5.51 s avg per Obama↔Goku switch vs 0.57 s warm
same-model; GPU residency run with no OOM.

## Known limitations / follow-up (stated honestly)

- **Live voice-channel playback NOT_RUN** — needs a human in a VC; left for
  the user.
- **Persistent per-model RVC worker cache not implemented** — documented as
  future work; until then an alternating-profile podcast turn costs ~5.5 s
  per switch (worker restart + model load). This follows the review's own
  sequencing guidance (do not optimize before the lifecycle fix).
- **Transient allocator spike** — a ~15.9 GiB spike can appear during LLM
  load (especially with concurrent external GPU load) and settles to the
  11.2 GiB steady state; no OOM in the recorded run. Monitored.
- GPU arbitration remains per-semaphore (P2-19, accepted-with-monitoring).
- Smaller content-LLM decision open (P2-18 follow-up), config-driven. The
  Round-3 residency evidence answers the underlying question: the 4B LLM
  coexists with TTS + RVC on 16 GB with no OOM, so no smaller model is
  required — the option remains config-only via `content.model_id`.
- Only the evidence reports (+ `REVIEW_NOTES.md`) are intentionally
  uncommitted; the coordinator commits them.
