# Benchmark Results — Qwen3-TTS + RVC voice conversion (corrected)

**Correction (review P1-13):** RTF = processing_time / audio_duration, so
**RTF > 1 means SLOWER than real time**. The previous version of this report
claimed long lines were "comfortably faster than real time" — that was wrong:
every total RTF in both runs exceeds 1. The system is ~1.6–2.1× slower than
real time end-to-end for full-length lines. The pipeline is non-streaming
through TTS→VC, so Discord playback starts only after the full utterance is
generated and converted; short commentary lines are proportionally faster in
wall-clock terms.

## New run (Gate R8, after the P1-8 RVC persistence fix)

Run: `benchmark/20261002T130047Z/` (2026-10-02), produced with
`benchmark-voices` (`src/notamusicbot/tools/benchmark_voices.py`).
Coordinator-verified re-run after commit `ac0c601` (resolve-first model
persistence). **10/10 lines, 0 failures.**

Per-speaker means over the 5 corpus lines (times in seconds; means verified
against `results.csv`):

| Speaker | tts time | tts RTF | vc time | vc RTF | total RTF | line-5 total RTF |
|---|---|---|---|---|---|---|
| Aiden | 13.7 | 1.51 | 3.7 | 0.58 | **2.10** | 1.72 |
| Ryan | 13.2 | 1.42 | 2.9 | 0.40 | **1.82** | 1.64 |

Representative rows (full data in `benchmark/20261002T130047Z/results.csv`):

| Speaker | Line | tts_time | vc_time | total | audio | tts_rtf | vc_rtf | total_rtf | peak dBFS |
|---|---|---|---|---|---|---|---|---|---|
| Aiden | 5 (paragraph) | 37.13 | 5.08 | 42.21 | 24.54 s | 1.51 | 0.21 | 1.72 | −0.17 |
| Ryan | 5 (paragraph) | 36.00 | 5.53 | 41.53 | 25.26 s | 1.43 | 0.22 | 1.64 | −0.64 |
| Aiden | 1 (neutral) | 8.33 | 6.64 | 14.98 | 4.70 s | 1.77 | 1.41 | 3.19 | −1.32 |
| Ryan | 1 (neutral) | 6.51 | 2.27 | 8.78 | 4.62 s | 1.41 | 0.49 | 1.90 | −0.48 |

### Effect of the P1-8 persistence fix (vs pre-fix run `20261002T062325Z`)

| Metric | Pre-fix | Post-fix | Change |
|---|---|---|---|
| Aiden vc RTF (mean) | 1.06 | 0.58 | ~2× faster |
| Aiden total RTF (mean) | 2.54 | 2.10 | improved |
| Ryan vc RTF (mean) | 0.94 | 0.40 | ~2× faster |
| Ryan total RTF (mean) | 2.35 | 1.82 | improved |

The improvement comes from warm converts no longer restarting the RVC worker
and reloading the model (one worker / one load across sequential conversions,
Gate R4). TTS RTF is unchanged (~1.4–1.5) — TTS is now the dominant stage.

## Correct interpretation

- RTF is highest on short lines (fixed per-call overhead dominates) and drops
  toward ~1.6–1.7 on long text — the pipeline amortizes, but **never crosses
  below 1.0 end-to-end** on this hardware/corpus.
- VC RTF improves strongly on longer audio (0.40–0.58 mean, 0.21–0.22 on the
  paragraph line) because RVC inference scales sub-linearly once warm.
- Time-to-first-playable-audio equals full generation time: non-streaming
  TTS→VC means Discord cannot start playback until the complete utterance is
  ready.
- Warm per-stage latencies from the Gate D end-to-end run
  (coordinator-verified): tts 7.9 s, vc 7.98 s for a 4.14 s clip.

## Configuration (from `metadata.json`)

- TTS: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, device `cuda`
- VC backend: `rvc`, model `GOKU`, pitch shift 0
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER (driver 617.14); Torch 2.14.1+cu130
- Corpus: `tests/fixtures/voice_benchmark.txt`, 5 lines × 2 speakers
- Peak level across all clips −0.2 dBFS

## GPU memory

Combined LLM (Qwen3-4B-Instruct-2507 bf16) + TTS steady state:
11.2 / 16.0 GiB. Load-order peak ~15.9 GiB observed once with concurrent
external GPU load (P2-19, accepted-with-monitoring limitation); LLM failure
falls back to static template lines, so speech keeps working.

## Retained audio

Both runs retain one WAV per corpus line for listening:
`benchmark/20261002T130047Z/{Aiden,Ryan}/*.wav` (current) and
`benchmark/20261002T062325Z/{Aiden,Ryan}/*.wav` (pre-fix, kept for
comparison).
