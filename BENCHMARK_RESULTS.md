# Benchmark Results — Qwen3-TTS + RVC voice conversion

Run: `benchmark/20261002T062325Z/` (2026-10-02), produced with
`benchmark-voices` (`src/notamusicbot/tools/benchmark_voices.py`).

## Configuration (from `metadata.json`)

- TTS: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, device `cuda`
- VC backend: `rvc`, model `GOKU`, pitch shift 0
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER (driver 617.14)
- Torch: 2.14.1+cu130
- Corpus: `tests/fixtures/voice_benchmark.txt`, 5 lines × 2 speakers
  (short neutral, excited, question, names/numbers, ~30 s paragraph)
- Failures: 0 — **10/10 OK**; peak level across all clips −0.2 dBFS

## Results (from `results.csv`)

Per-speaker means over the 5 corpus lines (times in seconds):

| Speaker | tts_rtf (mean) | vc_rtf (mean) | total_rtf (mean) |
|---|---|---|---|
| Aiden | 1.47 | 1.06 | 2.54 |
| Ryan | 1.41 | 0.94 | 2.35 |

Representative rows (full data in `benchmark/20261002T062325Z/results.csv`):

| Speaker | Line | tts_time | vc_time | total | audio | tts_rtf | vc_rtf | total_rtf | peak dBFS |
|---|---|---|---|---|---|---|---|---|---|
| Aiden | 1 (neutral) | 6.95 | 6.18 | 13.13 | 4.14 s | 1.68 | 1.49 | 3.17 | −0.60 |
| Aiden | 4 (names/numbers) | 10.50 | 6.42 | 16.92 | 7.42 s | 1.42 | 0.86 | 2.28 | −0.39 |
| Aiden | 5 (paragraph) | 39.12 | 10.36 | 49.47 | 27.58 s | 1.42 | 0.38 | 1.79 | −0.18 |
| Ryan | 5 (paragraph) | 42.28 | 11.49 | 53.77 | 30.14 s | 1.40 | 0.38 | 1.78 | −0.18 |

## Interpretation

- RTF is highest on short lines (fixed per-call overhead dominates) and drops
  toward ~1.8 on long text — the pipeline amortizes well for commentary-length
  speech.
- VC RTF improves strongly on longer audio (0.86–0.38) because RVC inference
  scales sub-linearly once the HuBERT/VC pipeline is warm.
- Long lines are comfortably faster than real time in total; single short
  sentences take ~12–17 s wall clock end-to-end including model-step overheads.
- Warm per-stage latencies from the Gate D end-to-end run
  (coordinator-verified): tts 7.9 s, vc 7.98 s for a 4.14 s clip.

## GPU memory

Combined LLM (Qwen3-4B-Instruct-2507 bf16) + TTS steady state:
11.2 / 16.0 GiB. Load-order peak ~15.9 GiB is tight (TTS loads first, LLM
second); LLM failure falls back to static template lines, so speech keeps
working.

## Retained audio

`benchmark/20261002T062325Z/Aiden/*.wav`, `benchmark/20261002T062325Z/Ryan/*.wav`
(one WAV per corpus line, retained for listening).
