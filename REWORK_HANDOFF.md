# notamusicbot_goku — Full Modernization / Qwen3-TTS Rework Handoff

**Repository:** `shanedertrain/notamusicbot_goku`  
**Baseline reviewed:** `main` at `8a6f81a8b4f8fe5c5a39639e5f1fcf1fb1c66edf`  
**Handoff date:** 2026-10-01  
**Primary runtime target:** Python 3.12+  
**Primary TTS target:** `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`  
**Initial voice-conversion target:** preserve existing RVC behavior behind a new adapter boundary  
**Longer-term extension:** Qwen3-TTS 1.7B VoiceDesign and/or direct Qwen voice cloning/fine-tuning

---

## 1. Objective

Renovate `notamusicbot_goku` into a maintainable, modular Discord audio bot with a clean speech pipeline:

```text
content / dialogue text
        ↓
Qwen3-TTS 0.6B CustomVoice
        ↓
source speech
        ↓
voice conversion
    initially RVC
        ↓
audio post-processing
        ↓
Discord playback queue
```

The important architectural rule is:

> **Text generation, TTS, target-voice conversion, post-processing, and Discord playback must be independent components.**

Qwen3-TTS is responsible for intelligible, natural source speech. The voice-conversion stage is responsible for the final character identity.

Do not redesign the system around a single TTS or voice-conversion implementation. We explicitly want to be able to replace either later.

---

## 2. User intent / product behavior

The bot is still fundamentally a Discord music/radio bot.

Existing behavior that should survive the rework:

- Discord voice-channel connection.
- YouTube queueing/playback.
- Spotify-related behavior where currently functional.
- Song-request commentary.
- Reddit/news commentary.
- Character/persona-driven generated dialogue.
- Generated speech before/around media playback.
- Target-character voices through voice conversion.
- Existing per-user association with a target character/voice.
- Skip, stop, join, play, play-now, and queue behavior.
- The experimental podcast/conversation capability should remain possible, but should use the same new speech pipeline rather than its own special path.

Do not couple a Discord user directly to an RVC implementation object. A user should select a **voice/persona profile**, and that profile decides how speech is synthesized and converted.

---

# 3. Immediate implementation decision

For the first working version use:

```text
Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

Official Qwen package:

```text
qwen-tts
```

Official project:

- https://github.com/QwenLM/Qwen3-TTS
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
- https://huggingface.co/collections/Qwen/qwen3-tts

The official Qwen documentation recommends a clean Python 3.12 environment and exposes `Qwen3TTSModel`.

### Important 0.6B limitation

Do **not** build the current architecture around instruction-driven voice control from the 0.6B CustomVoice checkpoint.

Qwen currently lists instruction control for the 1.7B CustomVoice model, but not for `0.6B-CustomVoice`.

For 0.6B use:

- text
- language
- built-in speaker

Keep an optional instruction field in our abstraction for future backends, but the 0.6B implementation must not depend on it.

### Built-in speakers

Current CustomVoice speakers include:

- `Aiden`
- `Ryan`
- `Vivian`
- `Serena`
- `Uncle_Fu`
- `Dylan`
- `Eric`
- `Ono_Anna`
- `Sohee`

For English target voices, start evaluation with `Aiden` and `Ryan`, but do **not** globally hard-code one source speaker.

Each target voice profile must be able to select the Qwen source speaker that converts best through its downstream VC model.

---

# 4. Current repository findings that must be addressed

The current repository already implements roughly:

```text
old TTS → RVC → speed modification → Discord
```

That is conceptually useful, but the implementation is too tightly coupled.

The rework must address the following.

## 4.1 Old TTS stack

`tts.py` currently supports or contains remnants of:

- `gTTS`
- `pyttsx3`
- TikTok voice TTS
- commented PlayHT support

Replace this with a backend abstraction.

The primary implementation becomes Qwen3-TTS.

Old TTS backends may be removed unless there is a demonstrated reason to retain one as a fallback.

---

## 4.2 RVC is coupled directly into application logic

`audio_generator.py` imports directly from `rvc_cli` and obtains:

- models
- audio processing
- voice conversion

This must stop.

No Discord, content-generation, or orchestration code should import `rvc_cli` directly.

---

## 4.3 Legacy dependency environment

The current `pyproject.toml` includes:

- Python 3.10
- `fairseq==0.12.2`
- old CUDA/Torch pins
- old RVC dependencies
- Windows-specific packages
- multiple obsolete TTS packages
- duplicate/ambiguous Discord packages

Do not force the new Python 3.12 application environment to inherit the legacy RVC dependency graph.

### Required approach

Prefer:

```text
main Python 3.12 application
    │
    ├── Qwen3-TTS
    ├── Discord
    ├── content providers
    └── speech orchestration
             │
             └── VoiceConverter interface
                        │
                        └── legacy RVC adapter/service
                            in isolated environment if necessary
```

If the existing RVC implementation works correctly inside the main Python 3.12 environment, it may run in-process.

If it requires old Python/fairseq/Torch dependencies, isolate it.

Do not compromise the new application environment merely to keep legacy RVC in-process.

---

## 4.4 Tracked `.env`

The repository currently contains a tracked `.env` blob while `.env` is also listed in `.gitignore`.

Treat this as a security defect.

Required actions:

1. Remove `.env` from the tracked working tree.
2. Add `.env.example` containing names only, never secrets.
3. Ensure `.env` remains ignored.
4. Document that historical commits may still contain prior values.
5. Do **not** rewrite published Git history without explicit user approval.
6. Report which credentials need rotation if live credentials were ever committed.
7. Run a secret scan over the resulting tree and, if practical, commit history.

Do not expose secret values in logs or implementation reports.

---

## 4.5 Current code defects to close during rework

Do not preserve known bugs merely for compatibility.

Observed examples include:

- `AudioPlayer.clear_playlist()` is async but is called without `await` in the stop path.
- User JSON writing and reading use incompatible shapes: writer emits a list while reader expects `.items()`.
- Spotify queue logic can reference `media_audio_path` before assignment.
- A Spotify path calls `generate_song_comment_audio_file()` with an argument shape inconsistent with its current signature.
- Blocking `yt-dlp` extraction occurs in an async path in `types_playlist_items.py`.
- Playback uses `bot.voice_clients[0]` rather than explicit guild-scoped state.
- The player polls continuously with `sleep(1)` instead of waiting on a queue/event.
- Generated/downloaded temporary media is not managed by a clear lifecycle.
- `setup.sh` attempts to install a `requirements.txt` that is not present in the reviewed repository.
- `GEMENI_API_KEY` is misspelled.
- The current Gemini integration uses the legacy `google-generativeai` SDK and obsolete hard-coded `gemini-pro`.
- There are `sys.path` modifications that should not exist in a packaged application.
- `main.py`, `podcast.py`, and other modules duplicate application startup/cleanup logic.
- Application behavior depends on mutable module globals such as `MODELS`, `USERS`, and globally constructed provider clients.

Fix these as part of the migration.

---

# 5. Target architecture

Use a package-oriented structure approximately like this.

Exact names may vary, but responsibilities must remain separated.

```text
src/
└── notamusicbot/
    ├── __init__.py
    ├── __main__.py
    ├── app.py
    ├── config.py
    │
    ├── domain/
    │   ├── media.py
    │   ├── users.py
    │   ├── voice_profiles.py
    │   └── speech.py
    │
    ├── speech/
    │   ├── pipeline.py
    │   ├── coordinator.py
    │   ├── postprocess.py
    │   │
    │   ├── tts/
    │   │   ├── base.py
    │   │   └── qwen_custom_voice.py
    │   │
    │   └── voice_conversion/
    │       ├── base.py
    │       ├── passthrough.py
    │       └── rvc.py
    │
    ├── content/
    │   ├── base.py
    │   ├── gemini.py
    │   ├── prompts.py
    │   ├── news.py
    │   └── reddit.py
    │
    ├── media/
    │   ├── youtube.py
    │   ├── spotify.py
    │   └── downloader.py
    │
    ├── discord_bot/
    │   ├── bot.py
    │   ├── commands.py
    │   ├── player.py
    │   └── guild_state.py
    │
    └── tools/
        ├── speech_test.py
        ├── migrate_users.py
        └── benchmark_voices.py

config/
├── voices/
│   ├── goku.yaml
│   └── ...
└── app.example.yaml

tests/
├── unit/
├── integration/
├── smoke/
└── fixtures/
```

The new package should be importable normally.

No `sys.path.append()` hacks.

---

# 6. Core interfaces

Use typed interfaces or abstract base classes.

The exact form is flexible, but callers must depend on abstractions rather than concrete engines.

## 6.1 TTS request/result

Conceptually:

```python
@dataclass(frozen=True, slots=True)
class TtsRequest:
    text: str
    language: str
    speaker: str
    instruction: str | None = None


@dataclass(frozen=True, slots=True)
class AudioArtifact:
    path: Path
    sample_rate_hz: int
    duration_s: float
```

---

## 6.2 Speech synthesizer

Conceptually:

```python
class SpeechSynthesizer(Protocol):
    async def synthesize(
        self,
        request: TtsRequest,
        output_path: Path,
    ) -> AudioArtifact:
        ...
```

The current implementation is:

```text
QwenCustomVoiceSynthesizer
```

Future implementations may be:

```text
QwenVoiceDesignSynthesizer
QwenVoiceCloneSynthesizer
RemoteTtsSynthesizer
```

No higher-level component should care which one is active.

---

## 6.3 Voice converter

Conceptually:

```python
class VoiceConverter(Protocol):
    async def convert(
        self,
        source: AudioArtifact,
        profile: VoiceConversionProfile,
        output_path: Path,
    ) -> AudioArtifact:
        ...
```

Implement at least:

```text
PassthroughVoiceConverter
RvcVoiceConverter
```

The passthrough converter is important for testing Qwen without RVC.

---

## 6.4 Speech pipeline

The speech pipeline owns:

```text
TTS
→ voice conversion
→ post-processing
→ final audio artifact
```

The Discord bot does **not** manually perform these stages.

The content generator does **not** manually perform these stages.

---

# 7. Qwen3-TTS implementation requirements

## 7.1 Model

Initial model:

```text
Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

Use the official `qwen-tts` package and official `Qwen3TTSModel`.

Do not fork Qwen code unless necessary.

---

## 7.2 Persistent loading

The model must be loaded once at application/service startup and reused.

Do not construct or reload the model for every utterance.

Do not create one full GPU model instance per Discord request.

---

## 7.3 Device and dtype

Configuration must expose at least:

```text
device
dtype
attention implementation
model ID/path
```

Preferred CUDA behavior:

```text
bfloat16 where supported
```

Attention backend:

1. use FlashAttention 2 if the actual target installation supports it;
2. otherwise use SDPA.

The bot must not fail solely because FlashAttention 2 is unavailable.

This matters particularly on Windows installations.

---

## 7.4 Sample rate

Do not assume a fixed Qwen output sample rate.

Use the sample rate returned by Qwen:

```python
wavs, sample_rate = model.generate_custom_voice(...)
```

Write the source WAV with that sample rate.

Normalize/resample only at a defined downstream boundary.

---

## 7.5 Source format

Use uncompressed PCM/WAV between inference stages unless a backend explicitly requires another format.

Do not encode intermediate TTS audio to MP3 before voice conversion.

Preferred chain:

```text
Qwen float waveform
→ PCM WAV
→ VC input
→ PCM WAV
→ post-process
→ Discord/FFmpeg
```

---

## 7.6 Text handling

Do not strip punctuation indiscriminately.

The old code removes quote characters before TTS.

The new code should preserve punctuation unless a proven Qwen-specific normalization step is required.

Implement a small explicit text-normalization stage and test it.

---

## 7.7 Concurrency

Start conservatively.

Use a bounded inference coordinator.

Do not allow arbitrary Discord tasks to invoke the same GPU model concurrently without verification.

Initial safe model:

```text
async requests
→ bounded queue / semaphore
→ persistent Qwen instance
```

Measure before increasing concurrency.

If Qwen and RVC share the same GPU, coordinate their inference so simultaneous large allocations do not cause avoidable OOMs.

---

## 7.8 Warmup

Support an optional startup warmup utterance.

Record warmup separately from normal latency metrics.

---

# 8. Voice profile model

Separate:

- persona/content behavior
- TTS source voice
- target VC voice
- output audio behavior

Example:

```yaml
id: goku

persona:
  display_name: Goku
  description: >
    Character/persona description used by the content-generation provider.

tts:
  backend: qwen_custom_voice
  model: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
  language: English
  speaker: Ryan

voice_conversion:
  backend: rvc
  model: GOKU
  pitch_shift: 0

audio:
  speed: 1.0
  target_peak_dbfs: -1.0
```

Do not keep persona descriptions inside RVC model configuration.

---

# 9. Source-speaker calibration

The source TTS speaker is a per-target-voice parameter.

Implement a simple benchmark/audition tool:

```text
python -m notamusicbot.tools.benchmark_voices ...
```

It should be able to:

1. synthesize the same short English corpus using candidate Qwen speakers;
2. pass every result through a selected VC target;
3. save outputs in a clearly labeled directory;
4. record:
   - TTS generation time
   - VC time
   - total time
   - generated audio duration
   - TTS real-time factor
   - VC real-time factor
   - total real-time factor
   - errors
   - clipping/peak level
5. emit a CSV or JSON summary.

Start by comparing at least:

```text
Aiden
Ryan
```

Allow all nine speakers.

Do not automatically declare an objective "best" source speaker from simplistic metrics. Preserve the audition files so the user can select the best perceptual match for each target VC model.

---

# 10. RVC migration strategy

## 10.1 First milestone

Preserve current target voices.

Do not simultaneously replace Qwen TTS **and** change the target voice-conversion technology unless the current RVC stack is genuinely unusable.

We want a controlled comparison:

```text
old TTS → RVC
versus
Qwen3-TTS 0.6B → same RVC
```

This allows us to determine what Qwen improves.

---

## 10.2 Submodule verification

The reviewed tree references:

```text
rvc_cli
commit 535d3643144890461f2a77cef0a9c47dc9bfa0e6
```

and `.gitmodules` points to:

```text
https://github.com/AshtonScalise/rvc-cli.git
```

Verify whether the submodule is still retrievable from the developer environment.

If not:

- inspect the existing local submodule checkout before deleting anything;
- preserve the exact currently working source if available;
- identify its provenance and commit;
- report the missing/upstream state;
- do not silently substitute a different RVC implementation and call it equivalent.

---

## 10.3 Isolated RVC runtime if needed

If legacy RVC requires Python 3.10/fairseq/old Torch, isolate it.

Acceptable options include:

```text
local subprocess service
local HTTP service bound to localhost
simple IPC worker
```

Requirements:

- explicit startup/shutdown
- health check
- timeout
- meaningful error propagation
- one persistent model/runtime where practical
- no shell-command construction from untrusted Discord text
- paths passed safely
- no model reload for every line if avoidable

Do not use the current pattern of creating an unbounded/global process pool sized to every CPU core for GPU inference.

---

# 11. Post-processing

Create an explicit audio post-processing component.

At minimum support:

- sample-rate conversion where required
- mono/stereo conversion where required
- speed adjustment
- peak protection
- optional loudness normalization
- format conversion for Discord playback

Avoid multiple unnecessary encode/decode generations.

Do not perform speed modification by an opaque side effect inside RVC.

The pipeline should make the order explicit.

Preferred default:

```text
TTS
→ VC
→ resample/channel normalization
→ speed adjustment
→ peak/loudness protection
→ final WAV
```

Document any backend-specific reason to change the order.

---

# 12. Discord playback architecture

## 12.1 Replace list polling with `asyncio.Queue`

The player should be event-driven.

Prefer:

```python
asyncio.Queue[PlayableMedia]
```

over:

```text
list + pop(0) + sleep(1)
```

---

## 12.2 Guild-scoped state

Do not rely on:

```python
bot.voice_clients[0]
```

Create explicit guild state.

Even if we currently use only one guild, correct ownership should be:

```text
guild
→ voice client
→ media queue
→ playback task
```

---

## 12.3 Preserve commands

Preserve current user-facing semantics for at least:

```text
!gplay
!gplaynow
!gplay_spotify
!gjoin
!gstop
!gskip
```

Preserve administrative cleanup functionality if still useful.

Changing command spelling is not part of this rework.

---

## 12.4 Remove the Discord voice gateway monkey patch if possible

The current code patches `DiscordVoiceWebSocket.initial_connection`.

The reviewed dependency is `discord.py==2.3.2`.

As of this handoff, current PyPI `discord.py` is `2.7.1`.

Upgrade and test with the current supported Discord voice stack.

If the workaround is no longer required, remove it.

If a workaround remains necessary:

- isolate it in one compatibility module;
- document the exact reproduced upstream problem;
- test it;
- do not leave a large monkey patch inline in the bot manager.

---

# 13. YouTube / yt-dlp

Preserve the current resilience work around YouTube extraction, cookies, and JS runtimes, but reorganize it.

Requirements:

- no blocking `yt-dlp` calls directly on the event loop;
- one media/downloader abstraction;
- extraction and download offloaded from the event loop;
- temporary download cleanup after playback;
- graceful failure if cookies are missing;
- configurable JS runtime discovery;
- no hard-coded browser state;
- logging must not print authentication cookies.

Preserve current fallback behavior where a local download can be used when a direct stream URL fails.

---

# 14. Content generation / Gemini modernization

Content generation must become its own provider abstraction.

The current code uses the legacy:

```text
google-generativeai
```

Google now recommends:

```text
google-genai
```

Migrate to the current Google GenAI SDK.

Use async API calls where available rather than wrapping every request with `asyncio.to_thread()`.

Current official migration documentation:

https://ai.google.dev/gemini-api/docs/migrate

---

## 14.1 Model configuration

Remove hard-coded:

```text
gemini-pro
```

Make the model configurable.

As of this handoff, current Gemini models include the Gemini 3.x generation; use a currently supported model verified at implementation time.

Do not make application architecture depend on a specific Gemini model name.

---

## 14.2 Environment variable

Canonical variable:

```text
GEMINI_API_KEY
```

The old code uses:

```text
GEMENI_API_KEY
```

During migration, optionally read the misspelled legacy name once as a compatibility fallback and emit a warning.

Do not preserve the misspelling as the documented configuration.

---

## 14.3 Personas

Content-generation personas belong in voice/persona profiles.

The content provider gets a persona description; it should not receive an RVC model object.

---

# 15. Users configuration migration

Current user configuration handling is inconsistent.

Create a typed user-profile representation.

Example:

```yaml
users:
  "123456789":
    screen_name: example
    real_name: Example Name
    background: Example background
    voice_profile: goku
```

JSON is also acceptable.

Provide a migration tool that can handle both plausible old shapes:

```text
dictionary keyed by user ID
list of user objects
```

Do not destroy the user's existing `input/users.json`.

Migration should:

1. make a backup;
2. parse old data;
3. validate it;
4. write the new format;
5. re-read and validate the result.

---

# 16. Dependency cleanup

Rewrite `pyproject.toml` around **direct dependencies**.

Do not manually pin the entire transitive dependency graph in the project file.

Remove dependencies that existed only for removed code.

Expected main-runtime categories include:

```text
discord.py
qwen-tts
google-genai
python-dotenv or equivalent settings loader
yt-dlp
PyNaCl / Discord voice dependencies
soundfile
audio/resampling dependency where required
spotipy if Spotify behavior remains
praw / HTTP parsing dependencies if Reddit remains
```

Keep system FFmpeg as an explicit prerequisite rather than confusing it with unrelated Python packages named `ffmpeg`.

Avoid having both:

```text
discord
discord.py
```

unless there is a demonstrated reason.

---

## 16.1 Torch

Do not blindly carry forward:

```text
torch 2.5.0+cu121
```

Determine the correct current Torch/CUDA combination for the actual NVIDIA target environment.

Record:

- GPU
- driver
- CUDA runtime/toolkit as relevant
- Torch version
- Qwen package version
- attention implementation

The lockfile or documented install procedure must reproduce the tested environment.

---

# 17. Configuration

Create one typed settings layer.

Environment variables should be used for secrets and machine-specific overrides.

Non-secret application configuration can live in YAML/TOML.

Example `.env.example`:

```text
DISCORD_TOKEN=
DISCORD_GUILD=
GEMINI_API_KEY=
NEWS_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
SPOTIFY_CLIENT_ID=
SPOTIFY_CLIENT_SECRET=
```

Do not commit actual credentials.

---

# 18. Files and temporary artifacts

Use `pathlib.Path` everywhere.

Create explicit directories such as:

```text
var/
├── cache/
│   ├── youtube/
│   └── speech/
├── runs/
└── logs/
```

or an equivalent structure.

Do not clear broad output directories destructively at startup.

Use per-run/per-job directories and explicit cleanup policies.

Generated artifacts should have deterministic metadata and unique IDs.

---

# 19. Application lifecycle

Build a clear application composition root.

At startup:

1. load settings;
2. validate configuration;
3. configure logging;
4. initialize content provider;
5. initialize Qwen TTS model;
6. initialize VC backend/service;
7. initialize speech pipeline;
8. initialize Discord bot;
9. optionally warm up speech inference;
10. enter event loop.

At shutdown:

1. stop accepting work;
2. stop player tasks;
3. drain/cancel speech jobs intentionally;
4. disconnect voice clients;
5. shut down RVC worker;
6. release GPU/runtime resources;
7. clean temporary files according to policy.

Avoid import-time construction of expensive clients/models.

---

# 20. Failure behavior

Speech-generation failure must not crash the media queue.

If commentary generation fails:

```text
log failure
skip commentary
continue media playback
```

If TTS fails:

```text
skip generated commentary
continue media playback
```

If VC fails and profile policy permits fallback:

```text
optionally play raw Qwen TTS
```

Make that fallback configurable per profile or globally.

Default behavior should be explicit, not accidental.

---

# 21. Performance instrumentation

Record timings for every generated line:

```text
content generation
TTS
VC
post-processing
total speech pipeline
```

Also record:

```text
output audio duration
TTS RTF
VC RTF
total speech RTF
queue wait time
```

For CUDA, optionally log peak allocated GPU memory during benchmark runs.

Do not flood normal logs with GPU allocator diagnostics.

---

# 22. Local speech test CLI

Replace the role of `manual_generator.py` with a proper CLI.

Example desired workflow:

```text
python -m notamusicbot.tools.speech_test \
    --profile goku \
    --text "Hey, it's me, Goku!"
```

Support:

```text
--no-vc
--speaker
--language
--output
```

This tool must initialize the same production speech pipeline components, not duplicate implementation logic.

---

# 23. Voice benchmark CLI

Provide a repeatable command for source-speaker comparison.

Example:

```text
python -m notamusicbot.tools.benchmark_voices \
    --profile goku \
    --speakers Aiden Ryan \
    --corpus tests/fixtures/voice_benchmark.txt
```

Outputs:

```text
benchmark/<timestamp>/
├── metadata.json
├── results.csv
├── Aiden/
│   ├── source/
│   └── converted/
└── Ryan/
    ├── source/
    └── converted/
```

Include exact model IDs/versions in metadata.

---

# 24. Testing requirements

Use `pytest`.

The test suite must not require Discord, Qwen model weights, an API key, or an RVC model just to run the normal unit test suite.

Use fakes at boundaries.

---

## 24.1 Unit tests

At minimum cover:

- settings validation
- voice-profile loading
- user migration
- user lookup
- speech-pipeline sequencing
- passthrough VC
- post-processing contracts
- queue ordering
- skip
- stop
- queue clearing
- failure fallback
- temporary-file cleanup
- YouTube option construction
- content-provider prompt inputs
- generated artifact naming

---

## 24.2 Integration tests

Mark hardware/external tests separately.

Suggested markers:

```text
integration
gpu
rvc
network
discord_live
```

Tests should skip cleanly when prerequisites are absent.

Never report skipped/not-run integration tests as passed.

---

## 24.3 Qwen integration test

When model/GPU prerequisites are present:

1. load `Qwen3-TTS-12Hz-0.6B-CustomVoice`;
2. generate a short English line;
3. assert output file exists;
4. assert non-zero duration;
5. assert valid sample rate;
6. assert finite audio samples;
7. record generation latency and RTF.

---

## 24.4 RVC integration test

When target model is present:

1. synthesize known Qwen source speech;
2. convert with RVC;
3. assert output exists;
4. assert valid duration;
5. verify no NaN/Inf samples;
6. verify output peak is sane;
7. record latency.

---

## 24.5 End-to-end local test

Without Discord:

```text
text
→ content bypass/static text
→ Qwen TTS
→ RVC
→ post-process
→ final WAV
```

This is the primary acceptance gate for the speech rework.

---

## 24.6 Discord smoke test

Use mocks/test doubles for the ordinary automated suite.

A live Discord test may be run separately and must be explicitly reported as:

```text
PASS
FAIL
SKIPPED
NOT RUN
```

---

# 25. Code quality requirements

Target Python 3.12+.

Use:

- type hints
- `pathlib`
- dataclasses where appropriate
- explicit interfaces/protocols
- async only where work is actually asynchronous
- no blocking network/media operations on the event loop

Python imports should be organized as:

```text
standard library

installed libraries

workspace libraries
```

Keep functions and modules focused.

No giant new `bot_manager.py`.

No new module-level mutable registries for runtime objects.

No wildcard imports.

No `sys.path` mutation.

---

## 25.1 Tooling

At minimum:

```text
ruff check
ruff format --check
pytest
```

Add static type checking (`mypy` or equivalent) and configure it to provide useful coverage of the new package.

Do not require the legacy RVC internals themselves to become perfectly typed before the adapter can be accepted.

---

# 26. Logging

Use normal Python logging.

Include useful structured context in messages:

```text
guild_id
media_id
speech_job_id
voice_profile
backend
```

Never log:

- API keys
- Discord token
- cookies
- authentication headers

Exceptions should preserve tracebacks at the error boundary.

---

# 27. Backward compatibility

Preserve the user's existing bot workflow wherever practical.

Do not require immediate manual recreation of every user or voice mapping.

Provide migration tools/adapters.

A compatibility period is acceptable, but old architecture must not remain the permanent internal API.

---

# 28. VoiceDesign future extension

Do not implement VoiceDesign as the default now, but design the TTS interface so adding it is trivial.

Future model:

```text
Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

Expected future workflow:

```text
voice description
→ VoiceDesign reference utterance
→ optionally reusable clone/reference
→ production speech
→ optional VC
```

Potential long-term configurations:

```text
Qwen 0.6B CustomVoice → RVC
Qwen 1.7B VoiceDesign → RVC
Qwen Base clone → no VC
fine-tuned Qwen voice → no VC
```

The application above the speech pipeline must not change when we switch among these.

---

# 29. Phase plan

Implement in this order.

## Phase A — Baseline and safety

- Create implementation branch.
- Record baseline commit.
- Make a basic behavior inventory.
- Add `.env.example`.
- Stop tracking `.env`.
- Run secret scan.
- Preserve user/model configuration backups.
- Verify RVC submodule availability.
- Capture at least one known-good old-pipeline output if the old environment still runs.

### Gate A

Do not start destructive cleanup until configuration and legacy voice assets are preserved.

---

## Phase B — Modern project skeleton

- Python 3.12 package layout.
- modern `pyproject.toml`.
- typed configuration.
- logging.
- domain models.
- test framework.
- user migration.
- no import-time external clients.

### Gate B

```text
ruff PASS
type checker PASS or documented bounded findings
unit tests PASS
```

---

## Phase C — Qwen TTS

- add `SpeechSynthesizer`.
- implement persistent Qwen 0.6B CustomVoice backend.
- add passthrough VC.
- add post-processing.
- add local `speech_test`.
- add timing/RTF metrics.

### Gate C

Generate a valid local WAV from Qwen with no RVC.

---

## Phase D — RVC adapter

- isolate legacy RVC dependencies if necessary.
- implement `VoiceConverter`.
- eliminate RVC imports from business/Discord code.
- preserve existing voice models/profiles.
- integrate Qwen → RVC.
- benchmark Aiden and Ryan source speakers for at least one existing target voice.

### Gate D

Generate a valid final target-voice WAV:

```text
text → Qwen → RVC → post-process
```

---

## Phase E — Discord/media refactor

- guild state.
- `asyncio.Queue`.
- modern discord.py.
- remove or isolate gateway patch.
- reorganize yt-dlp.
- temp-file lifecycle.
- preserve commands.
- graceful speech failure.

### Gate E

Mocked Discord suite passes and live smoke result is explicitly reported.

---

## Phase F — Content provider modernization

- `ContentProvider` abstraction.
- migrate Gemini SDK to `google-genai`.
- make model configurable.
- move persona data out of RVC.
- update Reddit/news/song commentary.
- migrate podcast/conversation path to shared speech pipeline.

### Gate F

All content paths return text through the same abstraction and can feed the same speech pipeline.

---

## Phase G — Cleanup and final acceptance

Remove dead code/dependencies only after replacements are working.

Candidates likely include:

- old `tts.py` implementation
- pyttsx3
- gTTS
- TikTok TTS integration
- PlayHT remnants
- duplicate startup paths
- obsolete GenAI SDK
- direct RVC imports
- stale setup script behavior
- unused dependencies

Update README from scratch where necessary.

### Gate G

Clean install from documented instructions succeeds in a fresh environment.

---

# 30. Acceptance criteria

The rework is accepted only when all applicable items below are demonstrated.

## Architecture

- [ ] Python 3.12+ main environment.
- [ ] Qwen3-TTS 0.6B CustomVoice is the default TTS.
- [ ] Qwen model persists across utterances.
- [ ] TTS is behind an interface.
- [ ] VC is behind an interface.
- [ ] RVC is not imported by Discord/content business logic.
- [ ] Persona data is separate from VC model data.
- [ ] Future VoiceDesign can be added without changing Discord/content APIs.

## Speech

- [ ] Raw Qwen TTS works.
- [ ] Qwen → existing target VC works.
- [ ] Intermediate audio uses lossless PCM/WAV.
- [ ] Sample rate is not hard-coded incorrectly.
- [ ] Speed/post-processing is explicit.
- [ ] Timings and RTF are recorded.
- [ ] Source speaker is configurable per target voice.

## Discord/media

- [ ] Queue uses event-driven async behavior.
- [ ] State is guild-scoped.
- [ ] play works.
- [ ] play-now works.
- [ ] skip works.
- [ ] stop works.
- [ ] join works.
- [ ] generated commentary can precede media.
- [ ] speech failure does not kill playback.
- [ ] yt-dlp does not block the event loop.

## Configuration/security

- [ ] `.env` is not tracked.
- [ ] `.env.example` exists.
- [ ] secrets are absent from logs.
- [ ] old user config is migrated safely.
- [ ] old voice assets are preserved.
- [ ] secret-scan result is documented.

## Quality

- [ ] unit tests pass.
- [ ] Qwen integration result is documented.
- [ ] RVC integration result is documented.
- [ ] local end-to-end speech result is documented.
- [ ] Discord live test is explicitly PASS/FAIL/SKIPPED/NOT_RUN.
- [ ] fresh install procedure is verified.
- [ ] README reflects actual setup.
- [ ] dead direct dependencies are removed.

---

# 31. Required implementation evidence

Return evidence, not just a statement that the migration is complete.

Provide:

```text
IMPLEMENTATION_REPORT.md
TEST_RESULTS.md
BENCHMARK_RESULTS.md
DEPENDENCY_NOTES.md
```

The report should include:

- starting commit
- ending commit
- files added/removed
- architecture summary
- migration decisions
- RVC runtime decision
- Qwen model ID
- Qwen package version
- Torch version
- CUDA/device information
- Discord.py version
- content-provider SDK/model
- test commands
- exact pass/fail/skip counts
- benchmarks
- known limitations
- remaining follow-up work

Include representative generated audio artifact paths/names if they are part of the returned workspace.

Do not place model weights or secret files into the report package.

---

# 32. Required test-result semantics

For every requested verification, report one of:

```text
PASS
FAIL
SKIPPED
NOT_RUN
```

Do not convert:

```text
not available
could not run
no GPU
no Discord token
missing model
```

into a pass.

If a test cannot be run, state the blocker precisely.

---

# 33. Commit discipline

Use logically separated commits.

Suggested sequence:

```text
chore: secure config and capture migration baseline
refactor: create typed application architecture
feat: add qwen3 tts backend
feat: add speech pipeline and postprocessing
refactor: isolate rvc voice conversion
refactor: modernize discord playback queue
refactor: modernize media retrieval
refactor: migrate content provider
test: add speech and bot integration coverage
docs: document setup migration and benchmarks
```

Do not squash everything into one opaque commit during development.

---

# 34. Do not do these things

- Do not rewrite Git history without user approval.
- Do not delete existing RVC models/configuration until migration is verified.
- Do not expose `.env` values.
- Do not replace RVC and Qwen simultaneously without preserving an A/B baseline.
- Do not use MP3 between TTS and VC.
- Do not reload Qwen per utterance.
- Do not create one GPU model per Discord request.
- Do not keep blocking `yt-dlp` or network operations on the event loop.
- Do not keep the old global process pool sized to CPU count for GPU inference.
- Do not make the new main environment Python 3.10 merely because legacy RVC prefers it.
- Do not keep `fairseq` in the main environment if only legacy RVC needs it.
- Do not hard-code the Qwen output sample rate.
- Do not hard-code a single source speaker for every target voice.
- Do not hard-code a specific Gemini model throughout application code.
- Do not claim integration success for tests that were not run.
- Do not build VoiceDesign now in a way that delays the 0.6B baseline.

---

# 35. Recommended first performance baseline

After Phase D, benchmark one representative target voice with:

```text
Qwen 0.6B / Aiden → RVC
Qwen 0.6B / Ryan  → RVC
```

Corpus:

1. short neutral line
2. excited line
3. question
4. sentence with names/numbers
5. 20–30 second paragraph

Capture:

```text
TTS latency
TTS RTF
VC latency
VC RTF
end-to-end latency
end-to-end RTF
GPU memory
audio duration
peak level
```

Retain the WAVs for listening.

This becomes the baseline against which later:

```text
1.7B CustomVoice
VoiceDesign
new VC engines
direct Qwen voice cloning
```

can be judged.

---

# 36. Completion definition

The project is not considered reworked merely because a Qwen call was added to `tts.py`.

The rework is complete when:

```text
Discord/content code
        ↓
generic SpeechPipeline
        ↓
generic SpeechSynthesizer
        ↓
Qwen3-TTS 0.6B
        ↓
generic VoiceConverter
        ↓
RVC adapter
        ↓
generic post-processing
        ↓
queued Discord playback
```

is the actual runtime architecture, the old dependency coupling is removed, the current bot behavior is preserved, the known defects above are closed, and the result can be installed and validated reproducibly.

---

# 37. Reference facts used for this handoff

As of 2026-10-01:

- Qwen's official repository recommends a fresh Python 3.12 environment for Qwen3-TTS.
- The official package is `qwen-tts`.
- `Qwen3-TTS-12Hz-0.6B-CustomVoice` supports the nine built-in CustomVoice speakers and streaming generation.
- Qwen lists natural-language instruction control for the 1.7B CustomVoice checkpoint, not the 0.6B CustomVoice checkpoint.
- The official generation API returns both waveform data and the sample rate; the implementation should use the returned rate.
- Qwen's current architecture also supports Base/voice cloning and a 1.7B VoiceDesign model, which is why the new TTS abstraction must not be CustomVoice-specific above the backend layer.
- Google recommends migrating Python applications from `google-generativeai` to the GA `google-genai` SDK.
- Current PyPI `discord.py` at the time of this handoff is 2.7.1; the reviewed project is pinned to 2.3.2.

---

## Final instruction to implementation agent

Perform the rework rather than only producing a proposal.

Prefer a working, tested migration in phases over a broad rewrite that leaves the repository non-runnable.

Preserve user data and existing voice assets.

When tradeoffs are required, prioritize:

1. a clean Python 3.12 application boundary;
2. a working Qwen3-TTS 0.6B → existing voice-conversion pipeline;
3. correct async Discord/media behavior;
4. reproducible configuration and tests;
5. future replaceability of TTS and VC backends.

Return the implementation plus evidence described above.
