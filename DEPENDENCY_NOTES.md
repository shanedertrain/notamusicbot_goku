# Dependency Notes — notamusicbot rework (Final review return (implementation ACCEPTED))

**Round-8/Round-9: no dependency changes** (and none in Round-7 either). No package was added, removed, or
re-versioned in the Round-7 commits (`5de5770…`, `cceb819…`, `77ff4e7…`);
the Round-7 work is lifecycle code + tests only. The legacy RVC `.venv` was
rebuilt from scratch via `scripts/setup.sh` during Round-7 after an
accidental `uv run` overwrote it (environment incident, resolved) — same
dependency set as recorded below, re-proven buildable from scratch.

Unit-suite context (current, Final Review reconciliation): **277 passed / 0 failed / 0 skipped** (main venv); portable stripped env **269 passed / 2 justified skips / 0 failed** — collects without Discord
in the main venv. Historical counts by round: 239 (Round-3/4), 244 passed /
2 skipped (Round-4 stripped-env run), 253 (Round-5), 259 (Round-6).

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
  unit suite runs without this extra (P1-14; Round-4 `9d370fe` adds
  `pytest.importorskip("discord")` to `test_command_registration`, so the
  suite now collects cleanly with discord absent — historically 239 passed /
  0 skipped with discord present at Round-4, only a justified skip when
  absent; current count 272 passed / 0 skipped).
- **`media`**: `yt-dlp` 2026.8.19, `yt-dlp-ejs` 0.8.0 (registers JS runtimes
  with yt-dlp; requires the `deno` binary — `media/youtube.py` imports it),
  `spotipy` 2.26.0 (`media/spotify.py` metadata search).
- **`dev`**: `pytest` 9.1.1, `pytest-asyncio` 1.4.0, `ruff` 0.16.10, `mypy` 2.4.0.
  Installed into the main venv by `scripts/setup.sh` (fixed in `548b150` — the
  dev extra was previously missing from the main-venv install).
- **`reddit`**: `praw` (floor `>=7.7`; 8.0.3 tested) — **Reddit IS ported**:
  `content/reddit.py` contains a real PRAW-backed `RedditPostFetcher` used for
  reddit interstitials and podcast topics. This is an optional extra, not a
  main dependency, and needs `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET` in
  `.env`. (An earlier revision of these notes wrongly called reddit a
  documented stub — corrected in the Round-3 return.)

Deliberately **not** dependencies of the main environment:

- `pyworld`, `faiss`, `torchcrepe`, etc. stay only in the legacy Python 3.12
  RVC venv (`.venv`) used by `legacy/rvc_worker.py` / `rvc_cli`.
  `fairseq` was **removed** from that venv: 0.12.2 does not support
  Python >= 3.11, and the legacy venv is now Python 3.12. HuBERT is loaded by
  `rvc_cli/hubert_loader.py` (pure PyTorch, a reimplementation of fairseq's
  `HubertModel.extract_features`) from `hubert_base.state_dict.pt`, a plain
  tensor state dict generated once from `hubert_base.pt` with the old py3.10
  fairseq venv (see `LEGACY_VENV.md`).
  The main environment never imports any of them. Round-4 `9d370fe` also makes
  `setup.sh` validate each enabled RVC profile's assets (model dir,
  `<variant>.yaml`, `.pth`, `.index`) and exit non-zero when missing — no new
  Python packages were added for this.
- No cloud LLM/TTS SDKs in the main environment: Gemini (`google-genai`),
  PlayHT and the TikTok TTS integration were removed with the legacy code.
  Commentary uses a local Qwen3-4B-Instruct-2507 — an explicit user decision
  ("We dont use Gemini anymore so we want to run the LLM parts on the 4070 ti",
  review P2-18). A smaller model (e.g. Qwen3-1.7B) is optional, config-only
  via `content.model_id`; the Round-3 GPU-residency run shows the 4B model
  coexists with TTS + RVC on 16 GB with no OOM, so no smaller model is needed.
- FFmpeg is a **system** prerequisite (installed by `scripts/setup.sh`), not a
  Python package; no `ffmpeg`-named Python package is required.

## Legacy RVC venv dependencies (`scripts/setup.sh`, Gate R7)

The legacy RVC environment is **Python 3.12** and is built **from scratch** by
`scripts/setup.sh` (proven on a fresh clone in the Round-3 return; setup
exits non-zero on missing assets). Its dependency list was completed in
`548b15007f87eb11d85cb52523d2e460de159e3d`:

- previously missing legacy deps now installed: `audiostretchy`, `pydub`,
  `gTTS`, `pyttsx3`, `pyyaml`, `python-dotenv`;
- `fairseq` is **not** installed: 0.12.2 does not support Python >= 3.11, so
  the venv moved to Python 3.12 and the fairseq build (and the
  `cython<3`/`wheel`/`setuptools<81` pre-install that only existed to build
  it) was dropped;
- HuBERT is loaded by `rvc_cli/hubert_loader.py` (pure PyTorch) from
  `hubert_base.state_dict.pt`; that file is generated once from
  `hubert_base.pt` using the old py3.10 fairseq venv (see `LEGACY_VENV.md`);
- torch 2.5.0+cu121 is still installed there.

These stay out of the main environment; the main app never imports them.

## Secrets

- `.env` is not tracked and is git-ignored; `.env.example` lists the supported
  variables. No LLM API key is needed (local LLM). Discord token and optional
  news/Spotify keys are the only secrets. Model weights and `var/` output are
  git-ignored and are not part of any report package.
- Packaging goes through `scripts/package.sh` (allowlist + secret scan,
  Gate R1); `5cd412f` additionally excludes the live `config/users.yaml`
  (live Discord user metadata) from review packages — only
  `config/users.example.yaml` ships. Because the first review package left
  the dev machine before that gate existed, rotation is recommended for the
  PLAY_HT/OPENAI/GEMINI-era keys (no key values are reproduced anywhere in
  these reports).
