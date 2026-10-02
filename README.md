# notamusicbot (Goku)

A Discord music bot with a local AI speech pipeline: it announces songs and
talks in voice channels using **Qwen3-TTS** converted to a target voice with
**RVC**, plus the usual `!gplay`-style media playback (YouTube / Spotify
search). Everything runs locally on your GPU — **no LLM API key required**.

## Architecture

```text
Discord commands / content providers (news, local LLM commentary)
        |
        v
SpeechPipeline            src/notamusicbot/speech/pipeline.py
        |                 synthesize -> voice-convert -> post-process
        v
SpeechSynthesizer         protocol; backend:
Qwen3-TTS 0.6B            Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice (persistent, bf16/sdpa/CUDA)
        |
        v
VoiceConverter            protocol; backend:
RVC adapter               JSON IPC -> legacy/rvc_worker.py (isolated Python 3.10 env) -> rvc_cli
        |
        v
Post-processing           speed (time-stretch), resample, peak normalize (WAV/PCM throughout)
        |
        v
Discord playback          guild-scoped, event-driven asyncio.Queue player
```

Key layout:

- `src/notamusicbot/` — the application package (`app.py` composition root,
  `config.py` typed settings, `discord_bot/`, `media/`, `content/`, `speech/`,
  `domain/`, `tools/`).
- `config/` — voice profiles (`config/voices/goku.yaml`) and `users.yaml`.
- `legacy/` — the isolated legacy RVC worker (JSON over stdin/stdout).
- `rvc_cli/` — vendored legacy RVC stack (Python 3.10, fairseq), never imported
  by the application.
- `scripts/setup.sh` — one-command setup for a fresh machine.
- `benchmark/` — retained voice-conversion benchmark runs.

## Setup

Prerequisites: git, curl, ffmpeg, an NVIDIA GPU (recommended), and `deno`
(for yt-dlp's JS challenges; the setup script checks for it).

```bash
./scripts/setup.sh              # full setup + non-GPU sanity checks
./scripts/setup.sh --gpu-check  # additionally run Qwen TTS + RVC smoke tests
./scripts/setup.sh --no-rvc     # skip the legacy Python 3.10 RVC environment
```

The script is idempotent: it installs uv + Python 3.12, creates `.venv-main`,
installs the package with the `discord` and `media` extras, downloads the Qwen
TTS/LLM weights, and seeds `.env` from `.env.example`. It does not print or
store any secret.

### Configuration (`.env`)

Copy `.env.example` to `.env` and fill in what you use:

```text
DISCORD_TOKEN=            # required to run the bot
DISCORD_GUILD=            # optional: home guild
NEWS_API_KEY=             # optional: news headline commentary
REDDIT_CLIENT_ID/SECRET=  # optional (reddit content is currently a stub)
SPOTIFY_CLIENT_ID/SECRET= # optional: better Spotify metadata search
```

There is **no LLM API key**: song commentary is generated locally by
Qwen3-4B-Instruct on the GPU, falling back to static template lines if the
LLM cannot load. Non-secret settings live in `config/app.example.yaml`
(speech defaults, output dir, warmup, etc.). Old user configs are migrated
with the `migrate_users` tool (`src/notamusicbot/tools/migrate_users.py`).

## Voice profiles

A voice profile pairs a Qwen CustomVoice source speaker (e.g. `Aiden`, `Ryan`)
with an RVC target model name and post-processing defaults
(`config/voices/goku.yaml`). Speech is always generated from the configured
source speaker, then converted to the target voice — the source speaker is
per-profile, never hard-coded, and sample rates are taken from the TTS return
value rather than assumed.

## Commands

Default prefix `!g` (configurable). Commands preserve legacy semantics:

| Command | Effect |
|---|---|
| `!gplay <query>` | Search YouTube, enqueue and play |
| `!gplaynow <query>` | Enqueue and skip straight to it |
| `!gplay_spotify <query>` | Spotify metadata search → YouTube → shared queue |
| `!gjoin` | Join your voice channel |
| `!gskip` | Skip the current song |
| `!gstop` | Stop and disconnect |
| `!gdeleteallmessages` | Clean up command chatter |

Generated commentary can be queued ahead of a song; a speech failure never
kills playback. yt-dlp runs off the event loop.

## CLI tools

```bash
.venv-main/bin/speech-test --help        # synthesize one line through the full pipeline
.venv-main/bin/benchmark-voices --help   # RTF/latency benchmark over a corpus
.venv-main/bin/python -m notamusicbot    # run the bot
```

## RVC legacy worker notes

RVC (fairseq 0.12.2, torch 2.5.0+cu121) only runs on Python 3.10, so it is
isolated in the repo-root `.venv` legacy environment behind
`legacy/rvc_worker.py`. The main app speaks a line-based JSON protocol to it:
one process, one persistently loaded model, `load-model` / `convert` /
`health` actions. The root-level legacy modules `tts.py` and
`configuration.py` exist solely for that worker's imports — the modernized
bot itself lives entirely in `src/notamusicbot/`. See `LEGACY_VENV.md`.

## Testing

```bash
.venv-main/bin/python -m pytest tests/unit                                  # 105 tests
.venv-main/bin/python -m pytest tests/integration -m "rvc and integration"  # needs GPU + RVC models
.venv-main/bin/ruff check src tests && .venv-main/bin/ruff format --check src tests
```

Evidence from the rework: `IMPLEMENTATION_REPORT.md`, `TEST_RESULTS.md`,
`BENCHMARK_RESULTS.md`, `DEPENDENCY_NOTES.md`.
