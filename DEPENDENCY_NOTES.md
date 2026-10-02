# Dependency Notes — notamusicbot rework

Main environment: Python 3.12, uv-managed `.venv-main`. Dependencies in
`pyproject.toml` are **direct-only**: every runtime dependency is imported by
`src/notamusicbot` (or the supported entry paths), and nothing the package
imports is missing from the declaration.

## Runtime dependencies (`[project.dependencies]`)

| Package | Version tested | Why it is direct |
|---|---|---|
| `qwen-tts` | 0.1.1 | Qwen3-TTS backend (`speech/tts/qwen_custom_voice.py`) |
| `soundfile` | 0.14.0 | WAV read/write throughout the speech pipeline |
| `soxr` | 1.1.0 | Preferred resampler in `speech/postprocess.py` (linear fallback exists) |
| `numpy` | 2.5.3 | Waveform math |
| `httpx` | 0.28.1 | `content/news.py` HTTP client |
| `huggingface-hub` | 0.36.2 | Model weight resolution/downloads (setup script, LLM cache checks) |
| `pydantic` / `pydantic-settings` | 2.13.5 | Domain models and typed `.env` settings |
| `pyyaml` | 6.0 | Voice-profile / user config files |
| `python-dotenv` | 1.0 | `.env` loading |

## Extras

- **`discord`**: `discord.py` 2.7.1 (tested), `PyNaCl` — bot + voice.
- **`media`**: `yt-dlp` 2026.8.19, `yt-dlp-ejs` 0.8.0 (registers JS runtimes
  with yt-dlp; requires the `deno` binary — `media/youtube.py` imports it),
  `spotipy` 2.26.0 (`media/spotify.py` metadata search).
- **`dev`**: `pytest` 9.1.1, `pytest-asyncio` 1.4.0, `ruff` 0.16.10, `mypy` 2.4.0.

Deliberately **not** dependencies of the main environment:

- `torch`/`transformers` arrive transitively via `qwen-tts` (tested with
  torch 2.14.1+cu130, transformers 4.57.3); they are not declared separately
  to avoid pinning a CUDA variant the user must choose. `scripts/setup.sh`
  documents the tested combination.
- `fairseq`, `pyworld`, `faiss`, `torchcrepe`, etc. stay only in the legacy
  Python 3.10 RVC venv (`.venv`) used by `legacy/rvc_worker.py` / `rvc_cli`.
  The main environment never imports them.
- No cloud LLM/TTS SDKs: Gemini (`google-genai`), PlayHT, gTTS, pyttsx3 and
  the TikTok TTS integration were removed with the legacy code. Commentary uses
  a local Qwen3-4B-Instruct-2507 (user decision). `praw` was not ported, so
  `content/reddit.py` is a documented stub.
- FFmpeg is a **system** prerequisite (installed by `scripts/setup.sh`), not a
  Python package; no `ffmpeg`-named Python package is required.

## Phase G changes

- Added to `media` extra: `yt-dlp-ejs`, `spotipy` (both are real imports in
  `src/notamusicbot/media/`).
- Promoted to runtime deps: `httpx`, `huggingface-hub`; declared `soxr`
  explicitly.
- `dev` extra gained `pytest-asyncio` (required by `asyncio_mode = "auto"`).
- Added the `benchmark-voices` console script next to `speech-test`.
- Removed with legacy code: `poetry.lock` (project uses uv, not Poetry).

## Secrets

- `.env` is not tracked and is git-ignored; `.env.example` lists the supported
  variables. No LLM API key is needed (local LLM). Discord token and optional
  news/Spotify keys are the only secrets. Model weights and `var/` output are
  git-ignored and are not part of any report package.
