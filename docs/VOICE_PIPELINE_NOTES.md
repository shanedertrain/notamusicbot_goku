# Voice generation pipeline — operational notes

Status: current as of 2026-10-30. This is the "how it works / what we learned"
note for the clone-only speech pipeline. It supersedes the RVC-era design docs
(removed from the repo); `docs/QWEN_NATIVE_VOICE_PLAN.md` is the study that led
to the migration, kept as history.

## 1. Pipeline shape

```text
voice profile (config/voices/<id>.yaml: reference clip + optional transcript)
        |
        v
QwenCloneSynthesizer (src/notamusicbot/speech/tts/qwen_clone.py)
  - shared Qwen/Qwen3-TTS-12Hz-0.6B-Base checkpoint, loaded once at startup
  - clone prompt built once per (reference, mode) and cached in memory
  - xvec: speaker embedding only (default; no transcript needed)
  - icl:  reference audio + transcript in context (copies prosody, can bleed content)
  - faster-qwen3-tts CUDA-graph wrapper when installed/available, stock qwen_tts fallback
        |
        v
SpeechPipeline (src/notamusicbot/speech/pipeline.py): TTS -> postprocess
  - speed/resample/peak-normalise to 48 kHz stereo
        |
        v
Discord playback (whole-sentence synthesis; the old RVC streaming reply path is gone)
```

There is exactly one TTS model and no per-voice trained model. A "voice" is a
reference clip plus a prompt cache entry. There is no voice-conversion stage.

## 2. Adding or changing a voice

1. **Source material.** Clean, dry, single-speaker audio of the target voice,
   roughly 10-30 s (3 s is the model minimum; 1-2 s clips clone less stably).
   Studio narration, podcast/interview mics, and game voice-line rips work
   well. Avoid halls, crowds, music, applause, and mic bleed from other
   speakers.
2. **Cut it.** Mono 24 kHz, peak-normalise to ~0.95, 20 ms edge ramps. Prefer
   cuts that start/end at pauses; for very short source lines, concatenate
   several lines with per-line loudness matching and ~0.1-0.2 s gaps (the
   current Goku reference is 8 such lines).
3. **Put it in `config/voices/refs/`** (not tracked in git — see §6) with a
   transcript next to it (`<name>.txt`). Record provenance in
   `config/voices/refs/README.md`.
4. **Profile YAML** (`config/voices/<id>.yaml`):

   ```yaml
   id: <id>
   persona: { display_name: ..., description: ..., chat_prompt: ... }
   tts:
     language: English
     reference_audio: config/voices/refs/<name>.wav
     reference_text: "..."    # only required for clone_mode: icl
     clone_mode: xvec         # xvec | icl
   audio: { speed: 1.0, target_peak_dbfs: -1.0 }
   ```

5. **Audition before switching production.** Render the same lines through the
   candidate reference with `tools/qwen_clone_render.py` (manifest ->
   `var/...` WAVs + ASR round-trip + timings), listen, and compare against the
   current voice. `tools/smoke_clone_pipeline.py` runs the real production
   pipeline end to end; `speech-test --profile <id> --text ...` is the quick
   manual check.

### Echo / reverb is the #1 reference-quality failure

A reverberant reference bakes its room into the clone: the speaker embedding
carries the acoustic signature even in xvec mode. This is exactly what happened
with the first Obama reference (a 2012 convention speech):

| Reference | Reference noise floor | Clone noise floor | Envelope autocorr (echo proxy) |
|---|---|---|---|
| 2012 DNC convention clip | −49.9 dB | −58.9 dB | 0.503 |
| dry audiobook excerpt | −93.5 dB | −83 to −94 dB | 0.38-0.44 |

The convention clip also carried crowd noise and PA/slapback, which the owner
heard as "a bit of an echo". Fix: replace the reference, do not try to
de-reverb the output. Dry studio narration (audiobook read by the speaker) and
isolated game voice lines (Goku) both produced clean clones.

### Other reference notes

- **xvec vs icl.** xvec is the default and is what the shipped profiles use.
  icl reproduces more of the reference delivery but conditions on the reference
  content: an earlier Obama icl render started with the reference's last word
  ("Hope."). Use icl only when the prosody match matters and the reference
  transcript is exact.
- **Register.** There is no per-profile pitch shift anymore. Pick a reference in
  the register you want the voice to speak in.
- **Do not re-normalise the output per voice.** Postprocess already peak-
  normalises to the profile's `target_peak_dbfs`.

## 3. Performance and VRAM budget (measured on RTX 4070 Ti SUPER 16 GB)

| Stage | Measurement |
|---|---|
| Clone generation, faster-qwen3-tts CUDA-graph wrapper | **RTF 0.26-0.47** (typical 4-5 s line in 1.3-2.5 s) |
| Clone generation, stock `qwen_tts` fallback | RTF 0.7-1.3 |
| Model load | 4.6-5.9 s with graph capture; ~3.3 s stock |
| Clone prompt build | 0.02-0.7 s once per voice; cache hit ~0 s after |
| LLM (Qwen3-4B) + clone TTS resident peak | **13.8 GB of 16 GB** (2.2 GB headroom, `evidence/gpu_residency.json`) |
| Synthesizer shutdown | drops the model + CUDA cache (~2 GB freed) |

Consequences:

- The 16 GB card fits **one** LLM and **one** 0.6 B TTS model, not two TTS
  checkpoints. Do not load a second TTS instance (tests, benches) while the bot
  is running — integration tests will OOM for that reason, not because of a
  code regression.
- Listen replies are whole-sentence: the RVC-era TTS/RVC streaming reply path
  was removed with RVC. A one-sentence reply is roughly 1.3-2.9 s of TTS. If
  first-audio latency needs to improve, the faster backend already exposes
  `generate_voice_clone_streaming`; restoring a streaming reply path is a
  contained follow-up.
- The wrapper is installed by `scripts/setup.sh` with `--no-deps`; without it
  the stock path still works (slower).

## 4. Voice inventory and provenance

| Profile | Reference | Source |
|---|---|---|
| `obama` | 18.33 s dry studio clip, xvec | *A Promised Land* audiobook excerpt read by Barack Obama (official Penguin Random House Audio; YouTube `xRUU4Emx7mE`), cut 58.11-76.44 s. Replaced the 2012 DNC convention clip after the echo report. |
| `goku` | 19.84 s concatenated lines, xvec | *All Goku Voice Clips — Dragon Ball Xenoverse 2 — Voice Lines (Sean Schemmel)* (YouTube `i1o3pL454Jo`); 8 isolated dry lines, trimmed, RMS-matched, joined with 0.12 s gaps. Replaced the 1.68 s `input/heyitsmegoku.mp3` clip. |

All users in `config/users.yaml` currently map to `obama`. Adding the other
characters back means sourcing one clean clip each (the RVC-era voices are
gone), then a profile YAML; no training is involved.

## 5. Testing / evidence

```bash
PYTHONPATH=src .venv-main/bin/python -m pytest tests/unit -q
PYTHONPATH=src .venv-main/bin/python -m pytest tests/integration tests/evidence -q   # needs the GPU idle
.venv-main/bin/python tools/smoke_clone_pipeline.py                                 # real end-to-end render
.venv-main/bin/python tools/qwen_clone_render.py --manifest <manifest.json> --out <dir> --asr
.venv-main/bin/python -m notamusicbot.tools.speech_test --profile obama --text "..."
```

Known test caveat: `tests/unit/test_spark_chat.py::test_live_spark_streams_a_goku_reply`
is a live loopback-Spark test; it can fail transiently when the Spark endpoint
is busy (reran green here). The integration/evidence suite needs the bot
stopped (VRAM).

## 6. Git / asset policy

- Audio and model binaries are **not tracked** (`.gitignore` covers
  `*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.opus *.webm *.pt *.pth *.onnx
  *.safetensors *.ckpt *.bin *.npy *.npz *.gguf`).
- Voice references therefore live only on disk under `config/voices/refs/`;
  provenance is in `config/voices/refs/README.md`, and `scripts/setup.sh`
  checks that every configured profile has its clip.
- `tests/fixtures/sine_2s.wav` is untracked but regenerated by
  `tests/conftest.py` when missing, so a fresh checkout still passes.
- Historical evidence under `evidence/listen_*.json` still uses RVC-era key
  names (e.g. `rvc_startup_s`); it is a record of those runs, not live
  behaviour. `evidence/gpu_*` were regenerated for the clone pipeline.

## 7. Known limitations / possible follow-ups

1. **No TTS streaming in replies** (removed with RVC). Restore via the faster
   backend's clone streaming if latency work resumes.
2. **0.6 B Base cloning ceiling.** A 1.7 B Base checkpoint clones better at
   ~+2.5 GB and higher RTF; fine-tuning (official, single-speaker only) is the
   escalation path if zero-shot quality is ever insufficient.
3. **Reference audio is not in git** (owner request), so deployments must carry
   `config/voices/refs/` separately.
4. **ICL content bleed** (see §2) — keep xvec as the default.
5. **One TTS model at a time** — see the VRAM note in §3.
6. **Per-voice pitch/register control is gone**; choose the reference at the
   intended register instead.

## 8. Decision history (2026-10-30)

Owner decisions, in order: use the Qwen voice pipeline only (no RVC); remove RVC
from the repo entirely; keep audio/model binaries out of git. The cloned Obama
voice was auditioned and approved from the 2012 DNC clip; after an echo report
the production reference moved to the dry audiobook clip. A Goku voice was then
built from the Xenoverse 2 voice lines. The RVC removal deleted the legacy
Python 3.10/3.12 environment, the `rvc_cli` submodule, `legacy/` worker/stream/
f0 code, HuBERT weights, faiss indexes, the voice-conversion router, the
streaming reply path, and the RVC tests/tools/evidence.
