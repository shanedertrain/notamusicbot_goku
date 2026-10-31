# Dependency Notes — notamusicbot

**2026-10-30 — clone-only voice pipeline (owner decision).** The RVC voice
conversion runtime was removed from the repo: the legacy `.venv`, the
`rvc_cli` git submodule, `legacy/` (worker/stream/f0), the HuBERT weights and
the faiss/pyworld/torchcrepe dependencies are gone, and
`faster-qwen3-tts` is used only as an optional CUDA-graph accelerator for the
clone backend: `QwenCloneSynthesizer.from_settings` tries it first on CUDA and
falls back to the stock `qwen_tts` model when the import/load fails (measured on
the 4070: ~0.3-0.5 RTF with the wrapper vs ~0.7-1.3 RTF stock for the same
sentence). The stock path is always available; the wrapper is installed by
`scripts/setup.sh` with `--no-deps`. Several RVC-only dependencies were dropped
(see "Removed with the voice-conversion stage" below).

Unit-suite context: see `TEST_RESULTS.md` for the exact pass/fail counts of a
given revision (the Lead regenerates it).

Main environment: Python 3.12, uv-managed `.venv-main`. Dependencies in
`pyproject.toml` are **direct-only**: every runtime dependency is imported by
`src/notamusicbot` (or the supported entry paths), and nothing the package
imports is missing from the declaration.

## Review P2-17 — torch/transformers now direct dependencies

Application code (`content/llm.py`, `speech/`) imports `torch` and
`transformers` directly; previously they were only inherited transitively
through `qwen-tts`. Fixed in commit `6ca34bdc0f25d03f24d98926d5bd65084bbc545f`
(`fix(deps): declare torch and transformers as direct dependencies (review
P2-17)`), which adds to `[project.dependencies]`:

```toml
# Declared explicitly: application code (content/llm.py, speech/) imports
# torch/transformers directly, not only via qwen-tts's transitive graph.
"torch>=2.6",
"transformers>=4.55",
```

Floor bounds only — the CUDA variant remains the user's choice; the tested
combination (torch 2.14.1+cu130, transformers 4.57.3) is documented in
`scripts/setup.sh`.

## Runtime dependencies (`[project.dependencies]`)

| Package | Version tested | Why it is direct |
|---|---|---|
| `qwen-tts` | 0.1.1 | Qwen3-TTS clone backend (`speech/tts/qwen_clone.py`) |
| `torch` | 2.14.1+cu130 | direct import in `content/llm.py`, `speech/` (P2-17) |
| `transformers` | 4.57.3 | direct import (P2-17) |
| `soundfile` | 0.14.0 | WAV read/write and reference-clip decoding in the speech pipeline |
| `soxr` | 1.1.0 | Reference-clip resampling + preferred resampler in `speech/postprocess.py` |
| `numpy` | 2.5.3 | Waveform math |
| `httpx` | 0.28.1 | `content/news.py` HTTP client |
| `huggingface-hub` | 0.36.2 | Model weight resolution/downloads (setup script, LLM cache checks) |
| `pydantic` / `pydantic-settings` | 2.13.5 | Domain models and typed `.env` settings |
| `pyyaml` | 6.0 | Voice-profile / user config files |
| `python-dotenv` | 1.0 | `.env` loading |

## Extras

- **`discord`**: `discord.py` 2.7.1 (tested), `PyNaCl` — bot + voice. The
  unit suite runs without this extra (`pytest.importorskip("discord")` in
  `test_command_registration`).
- **`media`**: `yt-dlp` 2026.8.19, `yt-dlp-ejs` 0.8.0 (registers JS runtimes
  with yt-dlp; requires the `deno` binary — `media/youtube.py` imports it),
  `spotipy` 2.26.0 (`media/spotify.py` metadata search).
- **`dev`**: `pytest` 9.1.1, `pytest-asyncio` 1.4.0, `ruff` 0.16.10, `mypy` 2.4.0.
  Installed into the main venv by `scripts/setup.sh`.
- **`listen`**: `silero-vad>=6.0` (`speech/vad/silero.py`, ONNX on CPU, loaded
  lazily). The energy VAD (`speech/vad/energy.py`) needs nothing and is used by
  the unit tests. Also `onnx-asr>=0.12` (the version the ASR bake-off used;
  `speech/asr/parakeet_onnx.py`, Parakeet TDT 0.6B v2 int8 on CPU, local HF
  cache only). One-time weights setup:
  `hf download istupakov/parakeet-tdt-0.6b-v2-onnx`.
- **Listen speed build (2026-10-25), installed by hand, NOT in any extra**:
  `onnxruntime` was REPLACED by `onnxruntime-gpu==1.31.0` (same version, same
  module name, so the two cannot coexist; uninstall `onnxruntime` first).
  The CUDA Parakeet path (`listen.asr.device: cuda`, fp32 weights) needs it
  plus torch imported first so the pip cuDNN is on the loader path; without
  CUDA it falls back to CPU int8. Not an extra because an extra cannot express
  a replacement. Reinstall order after a rebuild:
  `uv pip uninstall onnxruntime; uv pip install onnxruntime-gpu==1.31.0`.
  `faster-qwen3-tts==0.3.1 --no-deps` is also installed by `scripts/setup.sh`
  for the optional CUDA-graph clone path (the code falls back to stock
  `qwen_tts` when it is absent). The `--no-deps` flag avoids its own pinned
  dependency set conflicting with the main environment.
- **`reddit`**: `praw` (floor `>=7.7`; 8.0.3 tested) — `content/reddit.py`
  contains a real PRAW-backed `RedditPostFetcher` used for reddit interstitials
  and podcast topics. Optional extra, needs `REDDIT_CLIENT_ID` /
  `REDDIT_CLIENT_SECRET` in `.env`.

## Removed with the voice-conversion stage (2026-10-30)

- the `rvc_cli` git submodule and the repo-root `legacy/` RVC worker
  (`rvc_worker.py`, `rvc_stream.py`, `rvc_f0.py`), plus the legacy Python 3.12
  `.venv` that ran them;
- `hubert_base.pt` / `hubert_base.state_dict.pt`, the `legacy/` faiss indexes
  and the `rvc_cli/models/<Name>/` weights (not Python dependencies, but
  required assets the old runtime loaded);
- the extra Python packages that only the legacy venv used: `pyworld`, `faiss`,
  `torchcrepe`, `praat-parselmouth`, `librosa`, `resampy`, `audiostretchy`,
  `pydub`, `gTTS`, `pyttsx3`;
- the `faster-qwen3-tts` streaming CustomVoice path (the package itself stays
  installed as an optional, non-streaming clone accelerator with a stock
  fallback);
- no cloud LLM/TTS SDKs in the main environment: Gemini (`google-genai`),
  PlayHT and the TikTok TTS integration were removed with the legacy code.
  Commentary uses a local Qwen3-4B-Instruct-2507 — an explicit user decision
  ("We dont use Gemini anymore so we want to run the LLM parts on the 4070 ti",
  review P2-18). A smaller model (e.g. Qwen3-1.7B) is optional, config-only
  via `content.model_id`.
- FFmpeg is a **system** prerequisite (installed by `scripts/setup.sh`), not a
  Python package; no `ffmpeg`-named Python package is required.

## Secrets

- `.env` is not tracked and is git-ignored; `.env.example` lists the supported
  variables. No LLM API key is needed (local LLM). Discord token and optional
  news/Spotify keys are the only secrets. Model weights and `var/` output are
  git-ignored and are not part of any report package.
- Packaging goes through `scripts/package.sh` (allowlist + secret scan); the
  live `config/users.yaml` (live Discord user metadata) is excluded — only
  `config/users.example.yaml` ships.
