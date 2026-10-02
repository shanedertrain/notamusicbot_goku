# Test Results — notamusicbot rework (Round-3 re-review return)

Semantics: every result is PASS / FAIL / SKIPPED / NOT_RUN. Nothing
unavailable was converted into a pass. Counts are coordinator-verified for
the Round-3 return (after the Round-2 review return at `4716fab`; Round-3
commits `5cd412f` → `f03708a`, re-run 2026-10-02).

## Current round

| Check | Command / environment | Result | Count |
|---|---|---|---|
| Unit tests (main venv) | `.venv-main/bin/python -m pytest tests/unit` | **PASS** | 204 passed, 0 failed |
| Lint | `.venv-main/bin/ruff check src tests` | **PASS** | clean |
| Format | `.venv-main/bin/ruff format --check src tests` | **PASS** | clean (after coordinator's final pass) |
| RVC integration (Goku, RTX 4070 Ti SUPER) | `pytest tests/integration -m "rvc and integration"` | **PASS** | 2 passed |
| RVC switching benchmark (new) | `pytest tests/integration/test_rvc_switching.py` + `notamusicbot.tools.benchmark_rvc_switching` | **PASS** | Obama↔Goku: avg **5.51 s per switch** (worker restart + load) vs 0.57 s warm same-model (~10×) |
| Playback ordering (Gate R6 proof) | production-style queue test: song A → seg-1 → song B → seg-2 | **PASS** | hook fired exactly once per STREAM item; **zero hook calls** for generated items; no leftover listener tasks |
| Command registration | asserts registered commands play / playnow / play_spotify / join / stop / skip / podcast | **PASS** | prefix `!g` → `!gpodcast` (was `!ggpodcast`) |
| Download-leak failure path | download succeeds → source_factory raises → temp file removed | **PASS** | — |
| Stop/get race | queue.get resolves while stop fires → item recovered and cleaned | **PASS** | — |
| Provider startup fallback | LLM load raises → StaticTemplateProvider fallback installed, startup continues | **PASS** | — |
| Artifact retention | `keep_artifacts` wired; empty job dirs removed; failed-job retention 24 h; warmup scheduled exactly once, non-blocking | **PASS** | — |
| GPU residency (Gate R8, real run) | RTX 4070 Ti SUPER 16 GB: LLM 4B + TTS 0.6B + RVC Goku all resident | **PASS** | idle 11,262 / 16,375 MiB; peaks — content 9,840, TTS 9,857, RVC 9,857, end-to-end 9,857 MiB; **no OOM** |
| Clean install (Gate R7) | fresh clone + `scripts/setup.sh` built the legacy RVC venv **from scratch** (fairseq compiled, torch cu121) | **PASS** | setup correctly exits 1 on missing assets; after supplying documented assets (`hubert_base.pt` + `rvc_cli/models/Goku/goku.yaml`, gitignored) the RVC integration suite passed **2/2 with the freshly built venv — no symlinks** |

## Round-2 review "required tests" (section 9) — satisfaction map

| Required test (review §9) | Satisfied by | Result |
|---|---|---|
| Playback/content ordering: song A → podcast segment → song B; generated audio triggers nothing | ordering test in the player suite (`96000d3`) | **PASS** |
| Command surface: play/playnow/play_spotify/join/stop/skip/podcast with `!g` → `!gpodcast` | command-registration test (`07d16d3`) | **PASS** |
| Temporary file failure cleanup: download OK, source creation fails, file removed | download-leak ownership test (`96000d3`) | **PASS** |
| Stop race: queue.get resolves while stop fires; item recovered/cleaned | stop-race test (`96000d3`) | **PASS** |
| Provider startup fallback: LLM load raises, startup continues on fallback | fail-soft startup test (`f03708a`) | **PASS** |
| Full GPU residency: LLM + TTS + loaded RVC → content → TTS → VC, no OOM, memory recorded | real GPU run (Gate R8 evidence above, `f03708a`) | **PASS** |
| Alternating RVC profiles: Obama→Goku→Obama→Goku switch cost | `tools/benchmark_rvc_switching.py` + `tests/integration/test_rvc_switching.py` (`f03708a`) | **PASS** — 5.51 s/switch avg documented |

## Gate records (Round-3 disposition of every Round-2 gate)

| Gate | Result | Evidence |
|---|---|---|
| R1 package security | **PASS** | credentials clean (+ Round-3 `5cd412f`: live `config/users.yaml` excluded from packages; `config/users.example.yaml` added) |
| R2 production speech routing | **PASS** | router architecture + Round-3 playback-lifecycle fix (`96000d3`) |
| R3 migrated voices | **PASS** | 29 → obama, 1 → obama_trans (unchanged) |
| R4 RVC persistence | **PASS** | case-normalization fix holds + Round-3 alternating-cost evidence (5.51 s/switch) |
| R5 media lifecycle | **PASS** | failure-path (download-leak) + stop-race tests added (`96000d3`) |
| R6 content preservation | **PASS** | recursion impossible (hook fires for STREAM only), ordering guaranteed (head-insertion `GuildState.put_front`), command fixed (`07d16d3`) |
| R7 clean install | **PASS** | full from-scratch legacy venv via `scripts/setup.sh` (`548b150`); setup defects found & fixed during the proof |
| R8 benchmark | **PASS** | three-model VRAM residency + corrected RTF semantics + alternating-profile benchmark |

## Carried-over checks

| Check | Result | Count |
|---|---|---|
| Live Discord smoke (gateway connect + command registration, as Goku#8715) | **PASS** | coordinator, 2026-10-02 |
| Live voice-channel playback | **NOT_RUN** | requires a human in a VC; left for the user |

## Notes

- The unit suite grew 181 → 204 with the Round-3 tests (ordering, command
  registration, download-leak, stop-race, startup fallback, retention, RVC
  switching). No test was removed.
- Gate R7 from-scratch proof exposed and fixed real `setup.sh` defects
  (`548b150`): missing legacy deps (audiostretchy, pydub, gTTS, pyttsx3,
  pyyaml, dotenv), `setuptools<81` pin needed for `pkg_resources` (setuptools
  84 dropped it), and the `dev` extra missing from the main-venv install.
- GPU residency context: a transient ~15.9 GiB allocator spike can appear
  during LLM load with concurrent external GPU load; it settles to the
  11.2 GiB steady state. No OOM in the recorded run.
- Persistent per-model RVC worker cache is **not implemented**; the ~5.5 s
  per-switch podcast-turn cost is documented as future work.
- Benchmarks are reported separately in `BENCHMARK_RESULTS.md`.
