# Implementation Report — notamusicbot rework (Qwen3-TTS)

Rework of the legacy flat "notamusicbot" Discord bot onto a typed package
(`src/notamusicbot/`) with Qwen3-TTS as the speech backend and RVC voice
conversion behind an isolated worker boundary.

## Scope

- **Starting commit:** `8a6f81a` ("remove tik tok voice repo" — legacy baseline)
- **Ending commit:** see `git log 8a6f81a..HEAD`; Phase G closes at the
  `chore: quarantine legacy flat modules...` / `docs:` commits on top of
  `c2c1c6d` (Phase F).
- **Diff:** 112 files changed, ~20,300 insertions / ~10,000 deletions across
  `8a6f81a..HEAD` (including 82 new files).

## Architecture summary

```text
Discord/content code            (discord_bot/, media/, content/)
        ↓
generic SpeechPipeline          (speech/pipeline.py — synthesize → convert → postprocess)
        ↓
generic SpeechSynthesizer       (speech/tts/ — protocol; QwenCustomVoiceSynthesizer backend)
        ↓
Qwen3-TTS 0.6B                  (Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice, persistent in-process model)
        ↓
generic VoiceConverter          (speech/voice_conversion/ protocol)
        ↓
RVC adapter                     (RvcVoiceConverter → legacy/rvc_worker.py JSON IPC → rvc_cli)
        ↓
generic post-processing         (speech/postprocess.py — speed stretch, resample, peak normalize)
        ↓
queued Discord playback         (guild-scoped event-driven asyncio.Queue player)
```

- Application composition root: `notamusicbot/app.py` + `notamusicbot/__main__.py`.
- Typed settings: `notamusicbot/config.py` (pydantic-settings over `.env`).
- Domain models: `notamusicbot/domain/` (speech, media, users, voice profiles).
- No business-logic module imports `rvc_cli`; RVC is reachable only through the
  worker's stdin/stdout JSON protocol (`legacy/rvc_worker.py`, legacy venv).
- Persona data (voice profiles, users) lives in `config/` / `domain/`, separate
  from RVC model weights (`rvc_cli/models/`).

## Migration decisions

| Area | Decision |
|---|---|
| TTS | Qwen3-TTS 0.6B CustomVoice replaces pyttsx3/gTTS/TikTok/PlayHT; no cloud TTS dependency remains. |
| LLM / commentary | Local Qwen3-4B-Instruct-2507 replaces Gemini entirely (**user decision**); no `google-genai` dependency. `genai/` vendor dir removed. LLM failure falls back to static template lines. |
| Spotify | Simplified: spotipy metadata search → YouTube search → shared yt-dlp queue (no Spotify audio playback path). |
| Discord gateway | The voice-gateway monkey patch was removed; discord.py 2.7.1 handles speaking-mode intersection correctly. |
| Playback | Global `audio_player` process replaced by guild-scoped event-driven `asyncio.Queue` player; yt-dlp runs off the event loop. |
| Speed change | Implemented as interpolation time-stretch in `speech/postprocess.py` (soxr when available, linear fallback). |
| Reddit | `content/reddit.py` is a documented stub — `praw` was not ported. |
| RVC runtime | Kept in an isolated Python 3.10 legacy venv (fairseq 0.12.2, torch 2.5.0+cu121), driven by JSON IPC from the main py3.12 app. fairseq is **not** in the main environment. |
| Config | `.env` no longer tracked (`d2805a4`); `.env.example` provided; no LLM API key is required at all (local LLM). |

## Phase G cleanup inventory

Legacy root flat modules moved to `legacy_flat/` (kept for reference, unreferenced
by the new package): `main.py`, `bot_manager.py`, `audio_player.py`,
`audio_generator.py`, `discord_gateway.py`, `discord_voice_client.py`,
`manual_generator.py`, `newsapi.py`, `podcast.py`, `reddit_scraper.py`,
`spotify_handler.py`, `types_playlist_items.py`, `users.py`, root `__init__.py`,
`README_ejs.md`.

Removed (`git rm`): `genai/`, `playht_tts/`, old root `setup.sh`,
`poetry.lock`.

**Deliberately kept at the repo root:** `tts.py` and `configuration.py`.
These are imported by the isolated legacy RVC runtime (`rvc_cli/models.py`,
`rvc_cli/voice_converter.py`) via `REPO_ROOT` on `sys.path` inside
`legacy/rvc_worker.py`, so they belong to the RVC boundary, not to removable
flat app code (verified: moving them broke `tests/integration/test_rvc_worker.py`).

Untouched: `input/`, `backups/`, `rvc_cli/` (vendored legacy stack),
`legacy/`, `tools/`, `benchmark/`.

## Runtime details

| Item | Value |
|---|---|
| Qwen model ID | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| qwen-tts package | 0.1.1 |
| LLM | Qwen/Qwen3-4B-Instruct-2507, bf16 |
| Torch (main env) | 2.14.1+cu130 |
| transformers | 4.57.3 |
| attention / dtype | sdpa, bf16 |
| CUDA / GPU | RTX 4070 Ti SUPER 16 GB, driver 617.14 |
| discord.py | 2.7.1 |
| Main environment | Python 3.12 (`.venv-main`, uv-managed) |
| Legacy RVC environment | Python 3.10 (`.venv`), fairseq 0.12.2, torch 2.5.0+cu121, cuDNN SDP disabled |
| yt-dlp EJS | yt-dlp-ejs 0.8.0 + deno (JS runtime for yt-dlp) |

## Test commands and results

See `TEST_RESULTS.md` for exact counts (unit 105 passed; RVC integration
2 passed; live Discord smoke **NOT_RUN** at time of writing).

## Known limitations / follow-up

- Live Discord smoke test not yet run by the coordinator (field left NOT_RUN).
- `content/reddit.py` is a stub (praw not ported).
- `warmup_on_start` is off by default; `warmup()` now handles both loop states.
- Combined LLM+TTS steady-state GPU memory is 11.2/16.0 GiB; the load-order
  peak (~15.9 GiB, TTS first) is tight on 16 GB cards.
- Benchmark totals are cold-process numbers per line batch (see
  `BENCHMARK_RESULTS.md`); RTF improves on warm repeats within a run.
- Fresh-install verification of `scripts/setup.sh` is owned by the coordinator
  (Gate G); syntax-checked only (`bash -n`).

## Artifact paths (no weights or secrets included)

- Smoke: `var/speech/smoke/hey_it_me_goku.wav`
- End-to-end: `var/speech/e2e_goku.wav`
- Benchmark audio: `benchmark/20261002T062325Z/{Aiden,Ryan}/`
