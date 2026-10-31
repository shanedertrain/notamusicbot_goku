# Qwen-native custom voices instead of the RVC runtime stage

Status: **decision taken 2026-10-30 — clone-only, RVC removed from the repo.**
The owner approved the cloned Obama voice (14 s reference, x-vector), added a
cloned Goku voice from the original 1.68 s clip, and directed that RVC be
removed entirely rather than kept as a fallback. The implementation now in
flight: `QwenCloneSynthesizer` (Qwen3-TTS 0.6B Base), voice profiles carrying
reference audio, `passthrough`-free pipeline (TTS -> postprocess), no legacy
venv/submodule/HuBERT/faiss. The clone backend prefers the installed
`faster-qwen3-tts` CUDA-graph wrapper (RTF ~0.3-0.5) and falls back to the
stock `qwen_tts` model. The sections below are the study that led to the
decision; the RVC measurements are historical context, not live behaviour. Current operational notes
(adding voices, reference audio quality, performance, git asset policy):
`docs/VOICE_PIPELINE_NOTES.md`. Owner question: *"RVC takes a long time to convert. I am considering
leveraging the existing qwen voice model capabilities to just build custom models
based on audio. Map that out and let's discuss."*

Everything below that is labelled **measured** was run on this box, today, with
the GPU otherwise idle. Files quoted under `var/qwen_clone_probe/` are gitignored
listening artefacts.

## 1. The short version

Qwen3-TTS already ships exactly the capability being asked about: the **Base**
checkpoint does 3-second zero-shot voice cloning from audio, with no gradient
training. A voice becomes a small prompt/embedding asset, not a trained model.
A working local probe shows:

- **clone generation costs the same as the current CustomVoice generation**
  (RTF 0.67–0.69 vs 0.67 measured back to back);
- building the clone prompt is a one-time cost (0.72 s cold, 0.02 s warm) and can
  be cached per voice;
- the output is intelligible end to end (Parakeet round-trip recovered every
  sentence exactly, including both clone modes);
- streaming clone generation works through the installed `faster-qwen3-tts`
  backend (RTF 0.51, first chunk 1.65 s cold);
- therefore **the whole RVC stage can be deleted from runtime** if the clone
  timbre is acceptable to your ears.

That last "if" is the decision. RVC and cloning fail differently: RVC keeps the
source TTS prosody and rewrites timbre; cloning generates timbre *and* prosody
from the reference. For a stylised character like Goku, the mannerisms may be
the hard part, not the timbre. The probe WAVs exist so that this is decided by
listening, not by me.

## 2. What RVC currently costs (measured today)

Current production settings from `config/app.yaml` are the quality levers:
`f0_method_override: rmvpe`, `use_index: true`, `index_rate: 0.5`,
`index_nprobe: 32`, `protect: 0.33`, `streaming: true`, Obama `pitch_shift: -3`.

**Correction (same day, second measurement pass).** The first revision of this
document quoted RTF 0.40 for whole-file RVC. That figure was the *first*
conversion after a worker start, so it included ~2.5-2.9 s of weight/index load.
Re-measured on an already-warm worker, the conversion itself is ~25x cheaper.
The expensive RVC moments are worker/weight loads after a start or a model
switch, not steady-state conversion.

| Path | Measurement | Cost |
|---|---|---|
| First whole-file convert after worker start (7.42 s speech) | 2.85 s | one-time; includes weight + index load |
| Warm whole-file, production settings (rmvpe + index 0.5 + nprobe 32 + protect), 7.42 s | 0.183 s | **RTF 0.025** |
| Warm whole-file, same settings, 29.98 s continuous speech | 0.567 s | RTF 0.019 |
| Warm whole-file, index off, 7.42 s | 0.146 s | the index adds ~0.04 s |
| Cross-model switch, this measurement | 3.74-3.93 s | worker restart + weight/index load + convert |
| Cross-model switch, benchmark-only run | 3.2 s | same |
| Cross-model switch, checked-in evidence recorded 12:57 today | 10.0-11.9 s | contended machine; re-record when idle |
| RVC worker start alone | 0.91-0.98 s | one-time |
| Streaming RVC (listen reply path), design-doc measurement | — | RTF 0.06-0.11 |

Three honest readings:

1. **Per-utterance conversion is not the bottleneck.** On a 7 s line it is
   ~0.18 s (~2.5 % of the speech stage); on a 30 s commentary ~0.6 s. TTS is the
   bottleneck (RTF 0.67-0.95 across these runs), and that is true with or
   without RVC.
2. **The real recurring RVC cost is the per-model switch**: 3.2-3.9 s here and
   10-13 s when the box is contended, because the adapter restarts the worker
   and reloads weights. Every listen/commentary/podcast profile change pays it.
   That is the most plausible source of the owner's "RVC takes a long time to
   convert", and it is exactly what cloning removes (a prompt swap).
3. **The checked-in `evidence/rvc_cross_model_switching.json` is inflated.** It
   was recorded at 12:57 today and says 10-13 s per switch; clean re-runs give
   3.2-3.9 s. The file should be re-generated when idle before it is quoted
   again.

The listen reply path is *not* the worst case: it uses streaming RVC (RTF
0.06–0.11). The expensive path is whole-file conversion — post-song commentary,
podcast turns, and any fallback when streaming is unavailable — plus the 3.2 s
(up to ~10 s) worker restart whenever the voice profile changes.

## 3. What Qwen actually provides (facts, primary sources)

Official family as released by Qwen (`github.com/QwenLM/Qwen3-TTS`,
HF collection `Qwen/qwen3-tts`):

| Checkpoint | Purpose | Streaming | Instruction control |
|---|---|---|---|
| 0.6B / 1.7B **Base** | 3-second rapid voice clone from user audio; base for fine-tuning | yes | no |
| 0.6B / 1.7B **CustomVoice** | 9 built-in speakers (Ryan, Aiden, …) | yes | 1.7B only |
| 1.7B **VoiceDesign** | voice from a natural-language description | yes | yes |

Relevant details for this project:

- The installed `qwen-tts` 0.1.1 already exposes `create_voice_clone_prompt`,
  `generate_voice_clone`, `generate_voice_design`; only the CustomVoice
  checkpoint was cached locally. `Qwen/Qwen3-TTS-12Hz-0.6B-Base` is a 2.52 GB
  one-time download (done for this probe).
- Two clone modes: **ICL** (reference audio + its transcript; higher fidelity,
  conditions on reference content/prosody) and **x-vector-only** (speaker
  embedding only, `ref_text` not needed; the docs say cloning quality may be
  reduced, but it avoids phoneme/accent bleed from the reference).
- Prompt reuse is the documented production pattern: build once with
  `create_voice_clone_prompt`, pass `voice_clone_prompt` for every generation.
  The faster backend additionally caches reference embeddings/codes to disk.
- Fine-tuning is supported in the repo (`finetuning/`): single-speaker only,
  JSONL of `{audio, text, ref_audio}` → `prepare_data.py` (audio codes) →
  `sft_12hz.py` from a Base checkpoint, one checkpoint per epoch. Plan B only.
- Voice Design → clone is a documented workflow for voices with no reference
  audio: design a voice from a description, then use that clip as the clone
  reference. Useful for new personas, not for Goku/Obama.

## 4. Probe results (measured today)

Tool: `tools/qwen_clone_probe.py`. Outputs: `var/qwen_clone_probe/`.
Reference candidates: `refA` = the accepted Goku e2e render
(`~/.cache/listen-speed/rvc-quality/samples/HeyGoku/00_REFERENCE_e2e_goku.wav`,
4.14 s), `refB` = the shipped original Goku clip `input/heyitsmegoku.mp3`
(1.68 s), `refC` = an Obama RVC render
(`~/.cache/listen-speed/rvc-quality/samples/ObamaShift/obama_shift+0.wav`,
3.5 s) used to test the "RVC output as reference" migration path.

| Run | Wall time | Audio | RTF |
|---|---|---|---|
| Base 0.6B load | 3.3 s | — | — |
| Clone prompt build, 4.14 s ref | 0.72 s | — | — |
| Clone prompt build, same ref second time | 0.02 s | — | — |
| clone refA x-vector, sentence A | 4.06 s | 5.84 s | 0.69 |
| clone refA x-vector, sentence B | 2.49 s | 3.68 s | 0.68 |
| clone refA ICL, sentence A | 2.86 s | 4.24 s | 0.67 |
| clone refB x-vector, sentence A | 2.75 s | 4.00 s | 0.69 |
| clone refC (Obama) x-vector | 2.85 s | 3.92 s | 0.73 |
| **current** CustomVoice Ryan, sentence A | 4.86 s | 7.20 s | **0.67** |
| faster-qwen3-tts clone streaming, sentence A | 3.33 s | 6.56 s | 0.51 (first chunk 1.65 s cold) |

Correctness: Parakeet transcribed every clone output back to the exact input
text, including the 1.68 s reference. Qwen's own speaker encoder gives cosine
0.984–0.987 between the clone outputs and `refA`, 0.975 for `refB`-cloned output
vs `refB`, against 0.958 / 0.931 for the un-cloned Ryan voice — the clone moves
towards the reference, though that encoder is a weak discriminator here and not
a substitute for listening.

Same-sentence A/B set, ready to play (`var/qwen_clone_probe/`):

- `clone_refA_xvec_a.wav` / `clone_refA_icl_a.wav` — Goku clone from the 4.14 s ref
- `clone_refB_xvec_a.wav` / `clone_refB_icl_a.wav` — Goku clone from the 1.68 s ref
- `clone_obama_from_rvc_ref_a.wav` — Obama clone from an RVC render
- `current_path_rvc_goku_a.wav`, `current_path_rvc_obama_a.wav` — today's
  production path (Ryan TTS → RVC with the app.yaml quality settings)
- `current_tts_ryan_a.wav` — the un-converted source for reference
- `faster_clone_stream_a.wav` — the streaming clone, for latency/jitter check

Cost delta, corrected: the warm current path is TTS 0.67-0.85 + RVC
0.02-0.03 ≈ 0.70-0.88 RTF; clone measured 0.75-0.85 RTF in the Obama probe
(0.67-0.69 in the Goku probe). Per reply that is a saving of only ~0.2 s
(3-5 %), **not 37 %**. The large win is structural, not per-reply: a voice
change drops from 3.2-3.9 s (10-13 s contended) plus a ~2.9 s weight load to a
cached prompt swap, and the RVC worker (~0.8 GB when loaded,
`evidence/listen_models_benchmark.json` records +828 MiB after the first RVC
job) plus its legacy runtime disappears. VRAM for the clone build itself is not
yet measured; the Base checkpoint replaces the CustomVoice checkpoint 1:1 on
disk and in weights.

### 4b. Obama try-out from the 2012 DNC speech (2026-10-30)

Source: the owner's link, `https://www.youtube.com/watch?v=2rl8Ou84s5U`
(official channel, 40:12, manual captions). Five reference clips were cut at
caption timestamps (6.6-29.5 s each); full listening set, transcripts, timings
and the A/B demo are in `var/obama_voice/index.md`.

- Model load 1.18 s; prompt build 0.65 s cold, 0.02-0.04 s cached; clone
  RTF 0.75-0.85 (Ryan TTS on the same lines 0.85-0.95; run-to-run GPU
  variance is real and these are single runs).
- ASR round-trip exact on every x-vector render; speaker-embedding centroid
  cosine: clones 0.967-0.987, current RVC Obama 0.944, un-cloned Ryan 0.921
  (weak proxy, listening decides).
- **ICL bleeds reference content**: one render began with "Hope.", the
  reference's final word. All x-vector renders were clean.
- **Reference length changes pacing a lot**: a 6.6 s reference produced 3.3 s
  for a line that took 7.2 s from the 14.35 s reference.
- Same-line current path: Ryan TTS 6.34 s + RVC (2.83 s first convert / 0.18 s
  warm) vs clone 6.09 s. The clone is not slower; the structural wins are the
  removed worker load/switch and the simpler runtime, not per-reply speed.

## 5. The architecture this implies

Today a voice profile pairs a Qwen **source speaker** with an RVC **target
model**; the target voice exists only as a trained RVC checkpoint. With cloning,
a profile carries a **reference clip + transcript** and the shared TTS model
does everything. That is the "custom model based on audio" the owner is after:
adding a voice is dropping a WAV and a transcript, not training.

Concretely, the minimal change set:

1. `domain/voice_profiles.py`: `TtsProfile` gains a clone form
   (`backend: qwen_clone`, `reference_audio: config/voices/refs/<id>.wav`,
   optional `reference_text`, `x_vector_only: bool`). Old profiles keep
   working unchanged (`backend: qwen_custom_voice` + `voice_conversion.rvc`).
2. New `speech/tts/qwen_clone.py` implementing `SpeechSynthesizer` (+ the
   streaming surface): persistent Base model, one cached prompt per profile,
   prompt build at startup/warmup, prompt persisted under `var/` so a restart
   does not rebuild. The `faster_qwen3-tts` path supplies CUDA-graph streaming;
   stock `qwen-tts` is the fallback (whole-sentence only).
3. `app.py`: select the synthesizer by `speech.tts.backend`; the converter
   router already has `passthrough`, so clone profiles simply declare it. RVC
   startup is skipped when no enabled profile uses `rvc`, and the legacy
   worker/venv/fairseq/HuBERT/faiss/f0 code becomes dead weight that can be
   archived rather than deleted in the first step.
4. Voice configs: `goku.yaml` / `obama.yaml` switch to the clone form;
   `goku_raw.yaml` is already passthrough and becomes the trivial case.
5. Tests/evidence: profile validation (missing/invalid reference audio),
   prompt-cache hit, offline guard (all weights from cache), streaming parity,
   and a real-GPU A/B tool that renders the same sentences through both paths
   and writes `evidence/qwen_clone_ab.json` plus listening WAVs.

What disappears from the runtime: the RVC subprocess, the legacy Python 3.10
venv, HuBERT weights, faiss indexes, f0 extraction/post-processing, SOLA stream
joins, `VoiceConversionSettings` tuning, the "worker died; restarting" path,
and the 3–13 s model switch. What remains is one TTS model and one prompt
cache.

## 6. Options, with my recommendation

| # | Option | Verdict |
|---|---|---|
| A | **Zero-shot clone replaces RVC at runtime** (Base 0.6B, then 1.7B if quality needs it) | Recommended to evaluate first: no training, same RTF as today's TTS, removes the whole RVC stage. |
| B | **Hybrid migration**: keep RVC only as an *offline* voice builder. Render a neutral corpus through each existing RVC model once, use the render as the clone reference, then retire RVC at runtime. | The pragmatic path for the ~18 existing RVC models whose source audio you may not have. Demonstrated for Obama (`refC`); caveat: the clone then inherits whatever RVC timbre you were unhappy with. |
| C | **Fine-tune Base on target audio** (official single-speaker FT) | Plan B if zero-shot fails the listening test. One training run per voice, a trained artefact to keep, better fidelity potential. Do not start here. |
| D | **VoiceDesign for new personas** (1.7B) | Useful later for voices with no audio; not for Goku/Obama. |
| E | **Keep RVC, just optimize it**: drop the index (0.40 → 0.02 RTF but lose timbre), lower `index_nprobe`, persistent per-model worker cache (fix the 3–10 s switch), GPU faiss | Cheapest partial win, but keeps the quality problems and the second runtime. Good only if the clone timbre is rejected. |

## 7. Decision gates and experiments

The expensive part of cloning is not compute or code; it is **reference audio
quality**. 3–30 s of clean, single-speaker, representative audio is now the
critical asset per voice. The current Goku reference is 4.14 s of an
already-RVC-rendered clip and the original is only 1.68 s; no original Obama
clip was found on this machine. That is the first thing to fix if the
listening test is positive.

| Gate | Question | Pass criterion |
|---|---|---|
| G1 | Does the clone sound like the character? | Owner prefers (or accepts) the clone over today's RVC for Goku and Obama, x-vector and/or ICL. Uses the A/B files above, then better references. |
| G2 | Is it stable across sentences? | 6 sentences per voice (question/exclamation/plain), same speaker identity, no hallucinated or dropped words (Parakeet round-trip), no obvious drift; RTF and inter-sentence timbre variance recorded. |
| G3 | Does streaming survive? | `faster-qwen3-tts` Base clone streaming measured on the listen path: first-chunk latency and RTF no worse than the CustomVoice stream today (~22–37 ms first block after TTS chunk, RTF 0.06–0.11). If it regresses, decide whole-sentence-only vs keep RVC for listen. |
| G4 | Long-form and VRAM | 30 s commentary stable; LLM + clone TTS + ASR ≤ 14.5 GB steady; no network at runtime (offline guard). |
| G5 | Reference audio availability | Clean clips exist (or can be sourced) for every voice that matters; rights acceptable. |

Only after G1–G2 pass is it worth writing the production backend. G3–G5 are
build-time checks, not reasons to delay the listening test.

## 8. Open questions for the owner

1. **Which pain is the driver?** Per-reply conversion time (RVC 0.4 RTF on
   whole-file, 3–13 s on a voice switch) or adding new voices (training an RVC
   model per voice)? Cloning addresses both, but the order of work differs.
2. **Which voices matter?** Goku and Obama only, or most of the 18 RVC models?
   Option B is only needed for the ones without source audio.
3. **Do you have original reference audio** (not RVC output) for Goku, Obama,
   and anyone else? Where did the RVC training sets come from?
4. **Quality bar**: must the clone beat the current RVC, or is equal-but-faster
   and simpler acceptable?
5. **0.6B or 1.7B Base?** 0.6B is measured above. 1.7B likely clones better at
   ~+2.5 GB and a higher RTF; worth one A/B once the reference audio is real.
6. **Keep RVC as a fallback** behind a config flag during migration, or cut it
   over per profile?

## 9. Reproduce

```bash
# clone probe (downloads the Base checkpoint on first run; writes var/qwen_clone_probe/)
env -u HF_HUB_OFFLINE .venv-main/bin/python tools/qwen_clone_probe.py

# RVC cost, production settings
.venv-main/bin/python -m notamusicbot.tools.benchmark_rvc_switching --models Goku Obama --turns 1
```

The probe script is deliberately outside `src/`: it renders evidence, it does
not change the pipeline.
