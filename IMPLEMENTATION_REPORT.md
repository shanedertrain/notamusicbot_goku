# Implementation Report — notamusicbot rework (Qwen3-TTS), review-response round

Rework of the legacy flat "notamusicbot" Discord bot onto a typed package
(`src/notamusicbot/`) with Qwen3-TTS as the speech backend and RVC voice
conversion behind an isolated worker boundary. This report covers the
review-response round that closed the defects in
`NOTAMUSICBOT_GOKU_REWORK_REVIEW_RETURN.md`.

## Commits (coordinator-verified)

- **START_COMMIT = `8a6f81a8b4f8fe5c5a39639e5f1fcf1fb1c66edf`** (legacy baseline)
- **END_COMMIT = `6ca34bdc0f25d03f24d98926d5bd65084bbc545f`**
  (`fix(deps): declare torch and transformers as direct dependencies (review P2-17)`)

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

The P2-17 pyproject change was **already committed at report time** as
`6ca34bd` (it is not a pending working-tree change).

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

## Gate status (all coordinator-verified, 2026-10-02)

| Gate | Result | Evidence |
|---|---|---|
| R1 package security | **PASS** | `scripts/package.sh` allowlist + secret scan; earlier leaked review zip replaced in the recipient's Downloads. Credential rotation recommended for the PLAY_HT/OPENAI/GEMINI-era keys, since the first package left the dev machine. |
| R2 production speech routing | **PASS** | `tests/unit/test_app_production_routing.py` — spy-adapter proof through the real composition root: Discord commentary → SpeechPipeline → Qwen synthesizer → RVC adapter → profile postprocess → queue. |
| R3 migrated voices | **PASS** | Real legacy user file: 29 users → obama, 1 → obama_trans via documented weights alias `voice_profile_source: legacy_alias`; 0 goku; unknown model exits non-zero. |
| R4 RVC persistence | **PASS** | Two sequential conversions: 1 worker start, 1 load-model, 2 converts, 0 reloads — enforced by the resolve-first fix `ac0c601`; unit-covered. |
| R5 media lifecycle | **PASS** | Tests: downloaded fallback deleted after playback; webpage-URL download works with no stream URL. |
| R6 content preservation | **PASS** | song/news/reddit/podcast each via ContentProvider → SpeechPipeline → queue; interstitial + podcast default disabled. |
| R7 clean install | **PASS** | Fresh `git clone` + `git submodule update --init --recursive` + fresh uv py3.12 venv `.[media,dev]`: import OK, unit 180 passed / 1 skipped; RVC smoke 2 passed after documented local assets supplied. |
| R8 corrected benchmark | **PASS** | New run `benchmark/20261002T130047Z` after the P1-8 fix — see `BENCHMARK_RESULTS.md`. |

### Gate R7 — tracked-asset completeness (review point confirmed)

The review's point that tracked assets are incomplete is **CONFIRMED**:
`rvc_cli/models/*/yaml` files are gitignored, and model `pth`/`index` files
are only partially tracked. The clean clone needs these local assets supplied
before the RVC smoke passes:

- legacy `.venv` runtime (Python 3.10, fairseq 0.12.2, torch 2.5.0+cu121) — symlinked from the dev machine in the clean-clone test;
- `hubert_base.pt`;
- the per-model `goku.yaml` (and equivalents for other models);
- model `pth`/`index` files not covered by tracking.

`scripts/setup.sh` now prints this exact required-asset list.

## Architecture summary

Unchanged from the previous round: `src/notamusicbot/` typed package —
Discord/content → generic SpeechPipeline (synthesize → convert → postprocess)
→ Qwen3-TTS-0.6B in-process → RVC via isolated py3.10 worker (JSON IPC) →
guild-scoped queue player. Composition root `notamusicbot/app.py`; no
business-logic module imports `rvc_cli`.

## Test results summary

See `TEST_RESULTS.md` for the full table. Headline counts
(coordinator-verified): main-venv unit 181 passed / 0 failed; no-discord venv
(`[media,dev]`, discord absent) 180 passed / 1 skipped (proves P1-14); RVC
integration 2 passed; clean-clone unit 180 passed / 1 skipped, RVC smoke
2 passed.

## Known limitations / follow-up

- GPU arbitration remains per-semaphore (P2-19, accepted-with-monitoring).
- Live voice-channel playback still needs a human in a VC (left for the user).
- Smaller content-LLM decision open (P2-18 follow-up), config-driven.
- Only the four evidence reports (+ `REVIEW_NOTES.md`) are intentionally
  uncommitted; the coordinator commits them.
