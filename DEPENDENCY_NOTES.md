# Dependency Notes — notamusicbot rework (review-response round)

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
| `qwen-tts` | 0.1.1 | Qwen3-TTS backend (`speech/tts/qwen_custom_voice.py`) |
| `torch` | 2.14.1+cu130 | direct import in `content/llm.py`, `speech/` (P2-17) |
| `transformers` | 4.57.3 | direct import (P2-17) |
| `soundfile` | 0.14.0 | WAV read/write throughout the speech pipeline |
| `soxr` | 1.1.0 | Preferred resampler in `speech/postprocess.py` (linear fallback exists) |
| `numpy` | 2.5.3 | Waveform math |
| `httpx` | 0.28.1 | `content/news.py` HTTP client |
| `huggingface-hub` | 0.36.2 | Model weight resolution/downloads (setup script, LLM cache checks) |
| `pydantic` / `pydantic-settings` | 2.13.5 | Domain models and typed `.env` settings |
| `pyyaml` | 6.0 | Voice-profile / user config files |
| `python-dotenv` | 1.0 | `.env` loading |

## Extras

- **`discord`**: `discord.py` 2.7.1 (tested), `PyNaCl` — bot + voice. The
  unit suite runs without this extra (P1-14: 180 passed / 1 skipped with
  discord absent).
- **`media`**: `yt-dlp` 2026.8.19, `yt-dlp-ejs` 0.8.0 (registers JS runtimes
  with yt-dlp; requires the `deno` binary — `media/youtube.py` imports it),
  `spotipy` 2.26.0 (`media/spotify.py` metadata search).
- **`dev`**: `pytest` 9.1.1, `pytest-asyncio` 1.4.0, `ruff` 0.16.10, `mypy` 2.4.0.

Deliberately **not** dependencies of the main environment:

- `fairseq`, `pyworld`, `faiss`, `torchcrepe`, etc. stay only in the legacy
  Python 3.10 RVC venv (`.venv`) used by `legacy/rvc_worker.py` / `rvc_cli`.
  The main environment never imports them.
- No cloud LLM/TTS SDKs: Gemini (`google-genai`), PlayHT, gTTS, pyttsx3 and
  the TikTok TTS integration were removed with the legacy code. Commentary
  uses a local Qwen3-4B-Instruct-2507 — an explicit user decision ("We dont
  use Gemini anymore so we want to run the LLM parts on the 4070 ti", review
  P2-18). A smaller model (e.g. Qwen3-1.7B) is under review; swapping is
  config-only via `content.model_id`. `praw` was not ported, so
  `content/reddit.py` is a documented stub.
- FFmpeg is a **system** prerequisite (installed by `scripts/setup.sh`), not a
  Python package; no `ffmpeg`-named Python package is required.

## Secrets

- `.env` is not tracked and is git-ignored; `.env.example` lists the supported
  variables. No LLM API key is needed (local LLM). Discord token and optional
  news/Spotify keys are the only secrets. Model weights and `var/` output are
  git-ignored and are not part of any report package.
- Packaging now goes through `scripts/package.sh` (allowlist + secret scan,
  Gate R1). Because the first review package left the dev machine before that
  gate existed, rotation is recommended for the PLAY_HT/OPENAI/GEMINI-era
  keys (no key values are reproduced anywhere in these reports).
