# Review package — notamusicbot_goku rework (branch rework/qwen3-tts, HEAD fee1dd1)

Contents: full source tree (src/notamusicbot, tests, config, legacy/, scripts/, docs) + evidence reports + RVC model YAMLs. Excluded: venvs, git objects, GPU model weights, runtime var/, backups. Baseline & evidence live in git on the Linux side; this zip is a read-only snapshot for review.

## Open question flagged by the user: use a smaller content LLM

The current content model is Qwen/Qwen3-4B-Instruct-2507 (bf16, ~8 GB). Measured on the RTX 4070 Ti SUPER 16 GB (shared with Qwen3-TTS-0.6B and the RVC worker process):

- Combined steady-state LLM + TTS: ~11.2 GiB used (≈5.2 GiB headroom) — fine.
- Load-order peak is the problem: loading the 4B LLM peaked at ~15.9 GiB while other allocations were live → OOM risk at startup. Current mitigation: TTS loads first, LLM failure falls back to static commentary template.
- Generation quality/latency: 3.4–6.3 s per commentary line (~120 tokens), good in-persona output.

Candidate smaller models to review (all fit comfortably with TTS+RVC):
1. Qwen3-1.7B-Instruct (or Qwen2.5-1.5B-Instruct) — ~3.4 GB bf16; likely 2–3 s/line; VRAM peak pressure disappears.
2. Qwen3-0.6B — tiny, but commentary quality may degrade noticeably.
3. GGUF quantized 4B (Q4, ~2.5 GB) via llama-cpp-python — keeps quality, halves VRAM; adds a second inference stack.

## Where to look

- src/notamusicbot/content/llm.py — LocalQwenLlmProvider (model id/dtype/device all in ContentSettings; swapping the model is a config change: content.model_id)
- src/notamusicbot/config.py — ContentSettings (model_id, dtype, max_new_tokens, max_concurrent_generations)
- config/app.example.yaml — content section; content.enabled defaults false
- tests/integration/test_llm_gpu.py — real GPU smoke incl. combined-load VRAM measurement
- Evidence: IMPLEMENTATION_REPORT.md, TEST_RESULTS.md, BENCHMARK_RESULTS.md, DEPENDENCY_NOTES.md, REWORK_HANDOFF.md (original spec)

Also worth reviewing: Spotify command simplification (src/notamusicbot/media/spotify.py docstring) and the reddit.py stub.
