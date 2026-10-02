# Review notes — notamusicbot_goku rework (branch rework/qwen3-tts)

HEAD at evidence time: `6ca34bdc0f25d03f24d98926d5bd65084bbc545f` (P2-17 deps
commit — committed, not pending). Baseline: `8a6f81a8b4f8fe5c5a39639e5f1fcf1fb1c66edf`.
All review defects P0-1..P0-5 and P1-6..P1-16 are FIXED; see the
cross-reference table in `IMPLEMENTATION_REPORT.md`. All gates R1–R8 PASS
(coordinator-verified).

## Still-open decisions (no code change required)

1. **Smaller content LLM (P2-18 follow-up).** Local Qwen LLM on the 4070 Ti is
   user-approved ("We dont use Gemini anymore so we want to run the LLM parts
   on the 4070 ti"). Whether to shrink from Qwen3-4B-Instruct-2507 (~8 GB
   bf16; steady state 11.2/16 GiB, load-order peak ~15.9 GiB observed once
   with concurrent external GPU load) to e.g. Qwen3-1.7B is under review.
   Swapping is config-only: `content.model_id` in `config/app.yaml`
   (`ContentSettings`). Candidates: Qwen3-1.7B-Instruct (~3.4 GB bf16),
   Qwen3-0.6B (quality risk), or a Q4 GGUF 4B via llama-cpp-python (adds a
   second inference stack).
2. **GPU arbitration (P2-19).** Separate LLM/TTS semaphores remain; accepted
   with monitoring rather than unified.
3. **Live voice-channel playback** — needs a human in a VC; left for the user.
4. **Credential rotation** — recommended for PLAY_HT/OPENAI/GEMINI-era keys
   because the first (pre-Gate-R1) review package left the dev machine. The
   leaked zip was replaced in the recipient's Downloads.

## Clean-clone asset list (Gate R7 finding — review point confirmed)

Tracked RVC assets are incomplete: `rvc_cli/models/*/yaml` are gitignored and
`pth`/`index` files are only partially tracked. A fresh clone additionally
needs: the legacy py3.10 `.venv` runtime (fairseq 0.12.2, torch 2.5.0+cu121),
`hubert_base.pt`, per-model YAMLs (e.g. `goku.yaml`), and the untracked
`pth`/`index` files. `scripts/setup.sh` prints this list. With these supplied,
the clean clone passes import, unit (180/1 skip), and RVC smoke (2 passed).

## Where to look

- `IMPLEMENTATION_REPORT.md` — commits, defect cross-reference, gate table
- `TEST_RESULTS.md` — per-check PASS/FAIL/SKIPPED/NOT_RUN table
- `BENCHMARK_RESULTS.md` — corrected RTF interpretation (RTF > 1 = slower than
  real time) + post-fix run `benchmark/20261002T130047Z`
- `DEPENDENCY_NOTES.md` — direct-deps table incl. new torch/transformers
