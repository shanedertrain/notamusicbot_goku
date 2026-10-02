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
- `config/` — voice profiles (`config/voices/goku.yaml`) and
  `config/users.example.yaml` (the live `users.yaml` stays in the working
  deployment only and is excluded from review packages).
- `legacy/` — the isolated legacy RVC worker (JSON over stdin/stdout).
- `rvc_cli/` — vendored legacy RVC stack (Python 3.10, fairseq), never imported
  by the application.
- `scripts/setup.sh` — one-command setup for a fresh machine.
- `benchmark/` — retained voice-conversion benchmark runs.

## Setup

> **Platform requirement:** the runtime targets **Linux / WSL2 only**. Windows
> is not natively supported — the setup script and the application use POSIX
> paths (`.venv-main/bin/python`), Bash and ffmpeg. On Windows, run everything
> inside WSL2.

Prerequisites: git, curl, ffmpeg, an NVIDIA GPU (recommended), and `deno`
(for yt-dlp's JS challenges; the setup script checks for it).

```bash
./scripts/setup.sh              # full setup + non-GPU sanity checks
./scripts/setup.sh --gpu-check  # additionally run Qwen TTS + RVC smoke tests
./scripts/setup.sh --no-rvc     # skip the legacy Python 3.10 RVC environment
```

The script is idempotent: it installs uv + Python 3.12, initializes the
`rvc_cli` git submodule (`git submodule update --init --recursive`), creates
`.venv-main`, installs the package with the `discord` and `media` extras,
scaffolds `config/app.yaml` from `config/app.example.yaml` (only if missing),
downloads the Qwen TTS/LLM weights, and seeds `.env` from `.env.example`. It
does not print or store any secret, and any failed prerequisite aborts the
script with a non-zero exit code.

**Required RVC assets** (not distributed; supply locally — see the header of
`scripts/setup.sh` for details):

- `hubert_base.pt` at the repository root (HuBERT feature extractor, from the
  Retrieval-based-Voice-Conversion project releases);
- `rvc_cli/models/<Name>/*.pth` + `*.index` per voice (weights + FAISS index);
- `rvc_cli/models/<Name>/*.yaml` per voice/variant (the yaml stem is the
  legacy model name; e.g. `obama_trans.yaml` defines `OBAMA_TRANS` and shares
  the Obama weights in `rvc_cli/models/Obama/`).

### Configuration (`.env`)

Copy `.env.example` to `.env` and fill in what you use:

```text
DISCORD_TOKEN=            # required to run the bot
DISCORD_GUILD=            # optional: home guild
NEWS_API_KEY=             # optional: news headline commentary
REDDIT_CLIENT_ID/SECRET=  # optional: reddit interstitials / podcast topics
SPOTIFY_CLIENT_ID/SECRET= # optional: better Spotify metadata search
```

There is **no LLM API key**: song commentary is generated locally by
Qwen3-4B-Instruct on the GPU (`speech.commentary_enabled` and
`content.enabled` default to **true** in `config/app.example.yaml`), falling
back to static template lines if the LLM cannot load. Interstitial and
podcast generation remain opt-in (default off).

### Configuration resolution (non-secret app settings)

`load_app_settings()` resolves its YAML file in this order:

```text
explicit path argument → NOTAMUSICBOT_CONFIG env var → config/app.yaml → built-in defaults
```

A missing file is not an error (built-in defaults apply), but when
`config/app.yaml` exists its values win over the defaults. `scripts/setup.sh`
scaffolds it from `config/app.example.yaml` on first run.

Old user configs are migrated with the `migrate_users` tool
(`src/notamusicbot/tools/migrate_users.py`); it resolves each legacy model
name to a matching voice profile (e.g. `OBAMA` → `obama`) and exits non-zero
if a referenced model has no profile — it never silently falls back to `goku`.

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
| `!gpodcast start\|stop` | Toggle podcast conversation mode (see below) |

Generated commentary can be queued ahead of a song; a speech failure never
kills playback. yt-dlp runs off the event loop.

### Interstitials (news / reddit between songs)

Legacy behavior restored through the new abstractions: after a song finishes,
with a configured probability (and at most once per interval), the bot picks a
news headline or reddit text post at random, generates a commentary line with
the `ContentProvider`, synthesizes it with the configured voice profile through
the `SpeechPipeline`, and enqueues it ahead of the next track. Any fetch or
speech failure is skipped silently — playback continues.

```yaml
interstitial:
  enabled: false       # opt-in; default false
  probability: 0.25
  min_interval_s: 300
  voice_profile: goku
```

Reddit fetching needs the optional extra and credentials in `.env`
(`pip install 'notamusicbot-goku[reddit]'`); without them only news is used.

### Podcast conversation mode (`!gpodcast`)

`!gpodcast start` (stop with `!gpodcast stop`) puts the guild in podcast mode:
after each finished song the bot generates a short two-voice conversation — the
host persona from `podcast.host_profile` and the invoking user's voice profile
as guest — about a freshly fetched reddit/news topic. Every turn's text goes
through the shared `ContentProvider` and every turn is synthesized through the
`SpeechPipeline` with the speaker's voice profile, then enqueued in order. A
failure mid-segment aborts just that segment and playback continues.

```yaml
podcast:
  enabled: false       # opt-in; default false
  host_profile: obama
  turns: 2
```

Note: alternating voice profiles (e.g. Obama↔Goku) currently restart the
legacy RVC worker per model switch (~5.5 s per switch, vs ~0.6 s warm
same-model — see `TEST_RESULTS.md`); a persistent per-model worker cache is
planned future work.

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
.venv-main/bin/python -m pytest tests/unit                                  # full unit suite
.venv-main/bin/python -m pytest tests/integration -m "rvc and integration"  # needs GPU + RVC models
.venv-main/bin/ruff check src tests && .venv-main/bin/ruff format --check src tests
```

Exact pass/fail counts for a given revision are deliberately **not**
hard-coded here — see `TEST_RESULTS.md` (it also maps the review's required
tests and records the RVC-switching benchmark and GPU-residency run).

## CLI tools

Also available:

```bash
.venv-main/bin/python -m notamusicbot.tools.benchmark_rvc_switching --help  # alternating-profile RVC switch-cost benchmark
```

Evidence from the rework: `IMPLEMENTATION_REPORT.md`, `TEST_RESULTS.md`,
`BENCHMARK_RESULTS.md`, `DEPENDENCY_NOTES.md`.
