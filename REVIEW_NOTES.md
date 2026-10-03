# Review notes — notamusicbot_goku rework (branch rework/qwen3-tts), Round-4 return

HEAD at evidence time: `9d370fe8f8d47fc37cd8d245f6c77269993c7591`. Round-3
review return anchor: `02d1d970b5bb3a897398d10a13e6058f9c0caf6b`. Round-4
commits (SHAs verified with `git rev-parse`):

- `f804284` — P0 playback outcomes: `PlaybackOutcome` StrEnum (NATURAL/
  SKIPPED/STOPPED/FAILED/DROPPED) in `domain/media.py`; `_play_one()` returns
  `(media, outcome)`; skip/stop record `state.interrupt_reason` **before**
  `voice_client.stop()`; post-song hook fires **only on STREAM + NATURAL**,
  awaited before the next dequeue; after-callback errors → FAILED; dequeue
  without voice client → DROPPED. P1 artifacts: per-job `.temporary` JSON
  markers (crash-orphan pruning), full file + empty-dir cleanup in the
  production path, explicit `CancelledError` handling
- `9d370fe` — P0 variant identity: `voice_conversion.variant` end-to-end
  (profile → IPC `load-model {name, variant}` → worker selects the exact
  `<variant>.yaml`; `found[0]` eliminated; cache key `(resolved_dir, variant)`;
  same-dir variant switch = in-process config hot-swap, zero restarts);
  P1 shared GPU arbitration (one `InferenceCoordinator` for the content LLM
  provider AND the SpeechPipeline; pipeline holds the slot across
  TTS→VC→postprocess; warmup arbitrates); P1 unit isolation
  (`importorskip("discord")`, `FakeSynthesizer` in `build_pipeline` tests —
  no `qwen_tts` import in the unit suite); P2: `setup.sh validate_rvc_assets`
  per enabled profile (caught the real `GOKU`/`Goku` mismatch), explicit
  `.env.example`-only packaging rule + benchmark/evidence CSV/JSON
  allowlisted, podcast `turns`→`reply_turns`, guild-scoped interstitial
  cooldowns, silent-worker hard startup timeout (select-based deadline read
  replacing blocking `readline()`)

All Round-3 gates now **PASS**: R1, R2, R3 (variant identity preserved with
real config values; migration asserts config identity, not just strings), R4
(same-model + variant switching, 0 restarts on variant switch), R5 (crash
orphans + full cleanup + cancellation + hard startup timeout), R6 (outcome
matrix NATURAL→1, SKIPPED/STOPPED/FAILED/LOCAL→0), R7 (per-profile asset
validation), R8 (shared arbitration proven non-overlapping on real GPU —
3 iterations, concurrent LLM + TTS→RVC tasks, fully serialized timeline,
zero overlap). Unit suite: **239 passed / 0 failed / 0 skipped**; ruff check
+ format clean (coordinator-verified; re-run this session). Every required
Round-4 test from review §14 is implemented and PASS — mapping in
`TEST_RESULTS.md`.

## Remaining known items (stated honestly)

1. **Live voice-channel playback NOT_RUN** — needs a human in a VC; left for
   the user.
2. **Persistent per-model RVC worker cache not implemented** — documented as
   future work. A cross-MODEL (Obama↔Goku) switch still costs ~5.5 s (worker
   restart + model load), measured by
   `notamusicbot.tools.benchmark_rvc_switching`. Variant switches **within**
   a model are cheap in-process config hot-swaps (`config_switched: true`,
   zero restarts).
3. **Transient allocator spike (historical)** — ~15.9 GiB was observed during
   LLM load with concurrent external GPU load in Round 3; with the shared
   coordinator, loading/inference (warmup included) are serialized, removing
   the concurrent-inference concern. Steady state 11.2/16 GiB.

## Gate R8 — shared GPU arbitration (real run, RTX 4070 Ti SUPER 16 GB)

`tests/integration/test_gpu_arbitration.py`: all three models resident;
concurrent LLM generation task and TTS→RVC pipeline task; 3 iterations;
mutual exclusion asserted via `perf_counter` timestamps — the timeline is
**fully serialized with zero overlap** (every stage starts exactly when the
previous ends), so the RVC subprocess can never overlap main-process
inference. Three-model residency numbers from Round 3 still hold
(11,262 / 16,375 MiB idle; peaks ≤ 9,857 MiB; no OOM).

## Gate R7 — clean install + asset validation

The Round-3 from-scratch legacy venv proof holds (`548b150`). Round-4 adds
`validate_rvc_assets` in `scripts/setup.sh`: for every enabled RVC profile it
validates the model directory (case-insensitive), the `<variant>.yaml`, the
`.pth` and the `.index`, and exits non-zero on missing assets. It caught a
real defect during development — the `GOKU`/`Goku` directory-case mismatch.
Validated by `tests/unit/test_setup_rvc_assets.py` (bash extracted and run
against temp fixtures).

## Still-open decisions (no code change required)

1. **Smaller content LLM** — optional, config-only (`content.model_id`);
   residency evidence shows the 4B model is safe on the target GPU.
2. **Persistent per-model RVC worker cache** — future work (see above).
3. **Live voice-channel playback** — needs a human in a VC.
4. **Credential rotation** — recommended for PLAY_HT/OPENAI/GEMINI-era keys
   because the first (pre-Gate-R1) review package left the dev machine. The
   leaked zip was replaced in the recipient's Downloads.

## Where to look

- `IMPLEMENTATION_REPORT.md` — Round-4 commits, Round-3 defect cross-reference, gate table
- `TEST_RESULTS.md` — per-check PASS/FAIL/SKIPPED/NOT_RUN table + §14 required-test mapping
- `BENCHMARK_RESULTS.md` — corrected RTF interpretation (RTF > 1 = slower than real time)
- `DEPENDENCY_NOTES.md` — direct-deps table, Reddit correction, legacy-venv setup deps
