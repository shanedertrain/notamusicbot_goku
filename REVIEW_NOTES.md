# Review notes — notamusicbot_goku rework (branch rework/qwen3-tts), Round-3 return

HEAD at evidence time: `f03708aad5b35c664166a65ca1c71d5397723e00`. Round-2
review return anchor: `4716fabd9963dd0153e13680edd19a4898c3de20`. Round-3
commits (SHAs verified with `git rev-parse`):

- `5cd412f` — live `users.yaml` excluded from packages; `users.example.yaml` added
- `548b150` — `setup.sh` legacy deps completed; `setuptools<81` pin; dev extra in main venv
- `96000d3` — P0 playback lifecycle: STREAM-only `on_playback_complete` hook before next dequeue, `put_front` head-insertion ordering, stop/get race recovery, download-leak ownership
- `07d16d3` — P0 podcast command fixed: `name="podcast"` → `!gpodcast`
- `f03708a` — fail-soft content load (StaticTemplateProvider fallback), commentary/content default **on** (interstitial/podcast off), lazy skip-when-disabled, `keep_artifacts` + 24 h failed-job retention, empty job-dir prune, single non-blocking warmup

All Round-2 gates now **PASS**: R1 (+users.yaml privacy fix), R2 (architecture
+ lifecycle fix), R3, R4 (+alternating-cost evidence), R5 (failure-path +
stop-race tests), R6 (recursion impossible, ordering guaranteed, command
fixed), R7 (full from-scratch legacy venv via `setup.sh`), R8 (three-model
residency + corrected semantics). Unit suite: 204 passed / 0 failed; ruff
check + format clean (coordinator-verified). Every required Round-3 test from
review §9 is implemented and PASS — mapping in `TEST_RESULTS.md`.

## Remaining known items (stated honestly)

1. **Live voice-channel playback NOT_RUN** — needs a human in a VC; left for
   the user.
2. **Persistent per-model RVC worker cache not implemented** — documented as
   future work per the review's own sequencing guidance. Until then an
   alternating-profile (Obama↔Goku) podcast turn costs ~5.5 s per switch
   (worker restart + model load), vs 0.57 s warm same-model (~10×); measured
   by `notamusicbot.tools.benchmark_rvc_switching` /
   `tests/integration/test_rvc_switching.py`.
3. **Transient allocator spike** — ~15.9 GiB during LLM load (worst case with
   concurrent external GPU load), settling to 11.2 GiB steady; no OOM.

## Gate R8 — three-model GPU residency (real run, RTX 4070 Ti SUPER 16 GB)

All three models resident (4B content LLM + Qwen3-TTS 0.6B + RVC Goku):
**11,262 / 16,375 MiB idle**. Peaks: content 9,840 MiB, TTS 9,857 MiB,
RVC 9,857 MiB, end-to-end 9,857 MiB. **No OOM.** This answers the smaller-
model question with evidence: the 4B LLM coexists — no smaller model needed;
swapping remains config-driven via `content.model_id`.

## Gate R7 — clean install

Fresh clone + `scripts/setup.sh` built the legacy RVC venv **from scratch**
(fairseq compiled, torch cu121). Setup exits 1 on missing assets. After
supplying the documented local assets (`hubert_base.pt` +
`rvc_cli/models/Goku/goku.yaml` — yamls are gitignored), the RVC integration
suite passed 2/2 using the freshly built venv, **no symlinks**. The proof
exposed real setup defects, fixed in `548b150` (missing audiostretchy/pydub/
gTTS/pyttsx3/pyyaml/dotenv; setuptools<81 for pkg_resources; missing dev
extra in the main venv).

## Still-open decisions (no code change required)

1. **Smaller content LLM** — optional, config-only (`content.model_id`);
   residency evidence shows the 4B model is safe on the target GPU.
2. **GPU arbitration (P2-19)** — separate LLM/TTS semaphores remain;
   accepted with monitoring rather than unified.
3. **Live voice-channel playback** — needs a human in a VC.
4. **Credential rotation** — recommended for PLAY_HT/OPENAI/GEMINI-era keys
   because the first (pre-Gate-R1) review package left the dev machine. The
   leaked zip was replaced in the recipient's Downloads.

## Where to look

- `IMPLEMENTATION_REPORT.md` — Round-3 commits, Round-2 defect cross-reference, gate table
- `TEST_RESULTS.md` — per-check PASS/FAIL/SKIPPED/NOT_RUN table + §9 required-test mapping
- `BENCHMARK_RESULTS.md` — corrected RTF interpretation (RTF > 1 = slower than real time)
- `DEPENDENCY_NOTES.md` — direct-deps table, Reddit correction, legacy-venv setup deps
