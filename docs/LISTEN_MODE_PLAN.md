# Listen mode plan — notamusicbot Goku

Written 2026-10-25 by a Fable planning agent (read-only) from the code in this repo. Status: **owner decisions recorded 2026-10-25 (section 10); nothing built yet.**
Execution model after approval: an Opus orchestrator dispatches Sonnet builders per task, reviews the diffs, merges.
Paths below are relative to this repo unless absolute. Branch `listen` from `rework/qwen3-tts` (HEAD `b0022fa`).

> **2026-10-30 update — the opt-in consent system was removed** on owner instruction. Listen mode no longer has per-user opt-in: every human in the session channel is decoded and transcribed, and only `None` and the bot's own id are dropped (self-hearing, before opus decode). `ConsentState`, `!glisten on`/`!glisten off`, the opt-out clear hooks and `privacy.persist_opt_ins` are gone; the endpointer polls `ListenIngress.known_user_ids()`, and a member leaving only has their buffers and half-finished utterance dropped. Section 6 and the task table below reflect the current state. The review and measurement paragraphs further down (for example "**Consent: holds**") are dated records of the reviews that were actually run at the time and are preserved as history.

## 0. What the code says vs. what we assumed

| Assumption | Reality | Consequence |
|---|---|---|
| LLM is the Spark server on 127.0.0.1:8000 | `content/llm.py` loads **Qwen3-4B-Instruct-2507 in-process** via transformers (about 7.7 GB peak). No HTTP client exists. :8000 is the SSH tunnel to the Spark (reasoning models, `content: null` trap) | Decision 1.1: it sets VRAM and so the ASR choice |
| RVC is the latency bottleneck | **TTS is.** Qwen3-TTS 0.6B RTF 1.4-1.5, no streaming generation in `qwen_tts` 0.1.1. RVC warm 0.57-0.62 s; `f0_method_override: pm` already set | Latency plan targets TTS; skipping RVC would lose the Goku voice, so it is not a lever |
| RVC worker is Python 3.10 | `.venv` is Python 3.12, no fairseq; `legacy/README.md` is stale | Do not let a builder "fix" the venv |
| Stock discord.py cannot receive | discord.py 2.7.1 requires `davey` for voice and exposes `add_socket_listener`, `davey.DaveSession.decrypt`, `discord.opus.Decoder`, and a voice-WS `hook`. It lacks SSRC to user-id mapping (no op 5 SPEAKING handling) | A receive library only adds SSRC mapping, RTP parse, transport decrypt, DAVE decrypt, opus decode, per-user sinks |

Environment notes:
- `.venv-main` was rebuilt 2026-10-25 17:28 (davey 0.1.6 to 0.1.7). Record `uv pip list -p .venv-main` before Phase 0.
- Working tree is dirty (`evidence/*.json`, `rvc_cli` submodule, `tests/fixtures/converted_goku.wav`, `tests/fixtures/rvc_worker_last_latency.txt`; `stash@{0}` exists). Never sweep those into a listen commit.
- **Worktree trap:** `speech/voice_conversion/rvc.py:38` computes `REPO_ROOT = parents[4]` for `.venv/bin/python`, `legacy/rvc_worker.py`, `rvc_cli/models`; `hubert_base.state_dict.pt` is at repo root, all untracked. A worktree has none of them, and `.venv-main` is an editable install of the main checkout. Unit tests in a worktree: `PYTHONPATH=<worktree>/src .venv-main/bin/python -m pytest tests/unit`. GPU/live checks run from the main checkout on the listen branch, or symlink `.venv`, `.venv-main`, `rvc_cli/models`, `hubert_base*.pt`.
- Model downloads are a network step. Builders must not do them silently; put them in `scripts/setup.sh` or a documented tool.

## 1. Architecture

### 1.1 First decision: where the LLM runs (owner picks; interfaces are identical)

Steady state with LLM + TTS + RVC resident is **11.3 GB of 16 GB** (`evidence/gpu_residency.json`). Qwen3-ASR-1.7B needs about 4-6 GB (unmeasured here) and does not fit beside the in-process 4B LLM.

- **Path A: LLM stays in-process.** ASR is small: Whisper `small.en` or `base.en` via `transformers` (about 0.3-1 GB, 0.2-0.6 s per 5 s clip). No new deps. Lower accuracy on names and slang.
- **Path B: LLM moves to the tunnel server in listen mode.** New `OpenAiCompatProvider` (httpx already present) implementing `ContentProvider`, with `chat_text`-style null handling, `reasoning_effort: low`, optional streaming. Frees about 7 GB, so Qwen3-ASR-1.7B fits (already working in `~/projects/software/voiceover/.venv-asr`). Needs the tunnel up whenever the bot runs, and "local" read as "owner's own hardware". Latency unmeasured.

Recommendation: build A first, build B's provider as an optional task, and make the choice a config flip (`listen.llm.provider`). Both ASR backends implement one `Transcriber` protocol.

### 1.2 Data flow

```
Discord UDP -> receive library (SSRC->user, transport decrypt, DAVE decrypt)   [reader thread]
   | opus packets + user id
ListenIngress: drop None and the bot's own id (self-hearing)                  [reader thread]
   | decode every human in the channel; downmix + resample to 16 kHz mono
   | per-user bounded ring (<= 30 s)
VadWorker: silero ONNX (CPU) or energy VAD; endpointing; pre-roll              [1 CPU thread]
   | UtteranceEvent(user_id, pcm16k, t_start, t_eos) via call_soon_threadsafe
TurnManager: IDLE / LISTENING / THINKING / SPEAKING; barge-in                  [asyncio]
Transcriber (through InferenceCoordinator.run_blocking)                        [GPU, 1 slot]
Conversation: history, Goku persona prompt, reply cap
ContentProvider: in-process Qwen3-4B | OpenAiCompatProvider
ReplySynth: sentence chunks -> SpeechPipeline.run (TTS -> RVC -> postprocess)
ReplyPlayer: discord.PCMAudio(BytesIO) on voice_client.play; backchannel clip first
```

Threading rules:
- Receive callbacks never await, allocate GPU, or log per packet. They filter, decode, resample and append to a bounded `deque` under a `threading.Lock`.
- VAD runs on one dedicated CPU thread, never on GPU and never through the coordinator, so barge-in detection stays responsive while TTS holds the GPU slot.
- All GPU work goes through the existing single-slot `InferenceCoordinator` (`speech/coordinator.py`). An utterance's ASR queues behind an in-flight TTS chunk; accept in v1, measure, and only then consider a dedicated ASR lease.
- `SpeechPipeline.run` cancellation detaches but does not stop `generate_custom_voice`. Barge-in can only stop playback, cancel the awaiting reply task, and discard the in-flight chunk's artifact. Keep chunks to one sentence.
- Back-pressure: per-user ring 30 s (drop oldest); per-guild utterance queue depth 1 (newest replaces, older text merged into the prompt).
- Nothing new loads at import time. ASR loads in `Application.startup()` only when `listen.enabled`. `speech_features_enabled` must OR in `listen.enabled` (`app.py:118-130`).

### 1.3 New files

```
src/notamusicbot/
  config.py                     + ListenSettings (Vad/Asr/Reply/Llm/Privacy), AppSettings.listen
  domain/listen.py              AudioChunk, UtteranceEvent, SpeechStarted, Transcript, ListenTurnMetrics
  speech/asr/base.py            Transcriber protocol
  speech/asr/whisper_hf.py      Path A
  speech/asr/qwen_worker.py     Path B (subprocess JSON IPC, mirrors voice_conversion/rvc.py)
  legacy/asr_worker.py          Path B worker, runs in .venv-asr
  speech/vad/{base,energy,silero}.py
  content/openai_compat.py      Path B provider (optional)
  listen/{ingress,endpointer,conversation,reply,turn_taking,session,metrics}.py
  discord_bot/listen_glue.py    recv client, sink adapter, PCM playback, announce (the only discord-importing new file)
  discord_bot/commands.py       + !glisten, music guards
  discord_bot/bot.py            + ListenCoordinator, close() ordering
  app.py                        + load_asr_model(), startup/shutdown hooks
tests/unit/conftest.py          + FakeTranscriber, FakeVad, FakePacketSource, FakeContentProvider, FakeClock
tests/unit/test_no_discord_imports.py   + every new discord-free module
config/app.example.yaml         + listen: block (enabled: false)
```

```python
class Transcriber(Protocol):
    async def transcribe(self, audio: AudioChunk) -> Transcript: ...   # in-memory float32 mono 16 kHz, never a path
    async def shutdown(self) -> None: ...

class VoiceActivityDetector(Protocol):
    def process(self, frame: np.ndarray) -> float: ...   # 32 ms @ 16 kHz -> speech probability
    def reset(self) -> None: ...
```

## 2. Receive library decision

| Option | Verified | Must verify live | Verdict |
|---|---|---|---|
| **discord-ext-voice-recv** | Not on disk. discord.py hooks it relies on exist in 2.7.1. `VoiceRecvClient` subclasses `VoiceClient`, so `GuildPlayer` keeps working | Installed version; DAVE decrypt support with davey 0.1.7 + discord.py 2.7.1; PyPI reachability | First choice |
| In-house receiver on discord.py 2.7.1 | All primitives verified. Needs RTP parse, `aead_xchacha20_poly1305_rtpsize` decrypt (PyNaCl present), `dave_session.decrypt`, SSRC map from op 5 via `VoiceConnectionState(hook=...)`, opus decode. About 300 lines | Same criterion | Fallback |
| py-cord | Replaces the `discord` package; cannot coexist; batch-oriented sink | n/a | Rejected |

**Phase-0 spike P0a:** on a throwaway branch with its own venv copy, join a private test channel with **two humans**, request opus (`wants_opus()`), decode in our code, log per packet for 30 s: `(ssrc, user_id, dave_version, decrypt_ok, opus_decode_ok, pcm_rms)`, plus `voice_client.voice_privacy_code` and `dave_session.get_decryption_stats()`.
- **PASS:** DAVE session active (`dave_protocol_version > 0`), both users map to the correct `user_id`, at least 95% of packets decrypt and decode, RMS above 0 while speaking and about 0 in silence, and a 5 s capture (scratchpad only, deleted after) plays back intelligibly.
- **FAIL:** missing user ids, decrypt failures above 5%, white-noise PCM, or DAVE downgraded to passthrough (`dave_downgraded`), which would mean it only works because E2EE is off. If voice-recv fails, repeat with the in-house receiver before changing anything else.

## 3. ASR and VAD

| ASR | VRAM | Latency (5 s clip, est.) | Deps | Fit |
|---|---|---|---|---|
| Whisper `small.en` (transformers) | 0.6-1 GB | 0.3-0.6 s | none new | Path A |
| Whisper `base.en` | about 0.3 GB | 0.2-0.4 s | none new | Path A, lower accuracy |
| Qwen3-ASR-1.7B (`qwen-asr` 0.0.6) | 4-6 GB (measure) | 0.5-1.0 s | separate venv at voiceover; subprocess worker | Path B |
| Streaming Qwen3-ASR | as above | lower perceived | chunked decode every 1-2 s | later; not needed for utterances up to 15 s |
| faster-whisper | about 0.5 GB | fastest | CTranslate2 + CUDA 12 libs beside torch cu130: mismatch risk | not recommended |

VAD: **silero-vad** (ONNX on CPU; `onnxruntime` 1.30 is already in `.venv-main`; about 1 ms per 32 ms frame) with an **energy VAD** fallback (no deps; used by unit tests and degraded mode). webrtcvad rejected (C extension, Python 3.12 wheel uncertain).

## 4. Latency budget

| Stage | Target | Evidence |
|---|---|---|
| Endpoint silence | 600 ms (configurable 400-800) | config |
| VAD + handoff | under 20 ms | CPU thread |
| ASR (Whisper small.en, 5 s) | 0.3-0.6 s | estimate; measure in P0b |
| LLM first sentence | 1-2 s in-process with `max_new_tokens` 40-60 (today 3.9-11.3 s for 120 tokens) | `evidence/commentary_length_benchmark.json` |
| TTS, one sentence (about 12 words, 2 s audio) | 2.5-3.5 s | RTF 1.4-1.5 |
| RVC warm, `pm` f0 | 0.4-0.6 s | `evidence/rvc_same_model_warm.json` |
| Postprocess + buffer | under 0.15 s | soxr, in-memory |
| **End of speech to first real reply audio** | **P50 at most 6 s, P90 at most 9 s** | honest v1 figure |
| **End of speech to first bot sound (backchannel clip)** | **at most 0.8 s** | pre-rendered Goku clips, no GPU |

Levers: cap replies at 2 sentences; synthesize sentence N+1 while N plays; pre-render 4-6 backchannel clips ("Hm?", "Oh yeah?", "Let me think.") and the join announcement into `var/listen/`; warm RVC on session start (a cross-model switch costs 6.3 s); record TTS RTF in metrics. A faster TTS engine is out of scope here.

Measurement: `ListenTurnMetrics` logs one INFO line per turn with `eos→asr`, `asr→llm`, `llm→tts1`, `tts1→audio`, total. `tools/listen_bench.py` feeds a WAV through the harness and prints P50/P90; with fakes it checks plumbing overhead (under 50 ms), with `--gpu` it gives real numbers. The orchestrator must read the real GPU numbers against this table, not trust a green test.

## 5. Turn-taking

States per guild: `IDLE`, `LISTENING(user)`, `THINKING(user)`, `SPEAKING(reply_id)`.

- **Endpointing (per user):** speech starts after at least 250 ms above threshold; 300 ms pre-roll; ends after 600 ms silence or the 15 s cap; emits `UtteranceEvent` and a `SpeechStarted` at onset.
- **Addressing:** `respond_to: all | wake_word`. With `wake_word`, reply only if the transcript contains it or the bot spoke within `follow_up_window_s` (8 s).
- **Multiple speakers:** one reply pipeline per guild. An utterance during THINKING replaces the pending slot (depth 1); from a different user it is merged as `"<name A>: ...\n<name B>: ..."`.
- **Barge-in:** sustained speech (at least 400 ms) from a user who is not the bot while SPEAKING stops playback, cancels the reply task, marks history `"[interrupted after sentence k]"`, goes to LISTENING. Not triggered in the backchannel clip's first 300 ms.
- **Self-hearing:** (a) drop `user_id == bot.user.id`; (b) while SPEAKING raise the VAD threshold by 0.15 and the min speech to 400 ms; (c) echo-text guard: drop a transcript if its `difflib` ratio against the bot's last sentence exceeds 0.6 or it is a substring; (d) drop empty or under-2-word transcripts and the Whisper hallucinations `""`, `"."`, `"Thank you."`, `"Thanks for watching."`.
- **Limits:** per-user cooldown 2 s; at most 12 replies per minute per guild; LLM timeout 20 s drops the turn; ASR timeout 10 s.
- **Transcripts:** in-memory rolling history (8 turns), cleared on `!glisten stop`, session end or disconnect. Never written to disk. DEBUG logging only if `log_transcripts: true`.

## 6. Commands and config

Commands (prefix `!g`; register as `listen`, mind the double-prefix trap in `tests/integration/test_command_registration.py`):

| Command | Who | Effect |
|---|---|---|
| `!glisten start` | operator (`operator_user_ids`, else guild admins) | Requires the operator's voice channel in `allowed_channel_ids` (empty list refuses). Connects the recv client, plays the pre-rendered announcement, posts a text notice |
| `!glisten stop` | operator | Stops the session, clears buffers and history, disconnects |
| `!glisten status` | anyone | Channel, state |

A user leaving the channel has their buffers and half-finished utterance dropped; a user joining is heard automatically (there is no opt-in step). Music commands while listening reply "listen mode is on; `!glisten stop` first" (v1 mutual exclusion).

**Hard invariants (orchestrator reviews these):**
1. Packets from `None` or `bot.user.id` are dropped **before opus decode**; every other user in the session channel is decoded, VAD'd and transcribed (no per-user permission state exists).
2. User audio never touches disk. `AudioChunk` is in-memory; the Path B worker gets base64 PCM inside the JSON line, never a path.
3. `listen.enabled: false` and `allowed_channel_ids: []` by default, so listen cannot start.
4. No consent/approval state is persisted or carried between sessions; the only gate is the channel the session was started in.

```yaml
listen:
  enabled: false
  allowed_channel_ids: []          # empty => listen refuses to start
  operator_user_ids: []            # empty => guild administrators only
  voice_profile: goku
  announce_on_join: true
  announcement_text: "Heads up, I'm listening now. Everyone in this channel is transcribed, and nothing is saved."
  respond_to: all                  # all | wake_word
  wake_word: goku
  follow_up_window_s: 8.0
  vad: {backend: silero, threshold: 0.5, min_speech_ms: 250, end_silence_ms: 600, max_utterance_s: 15.0, pre_roll_ms: 300, ring_buffer_s: 30.0}
  asr: {backend: whisper_hf, model_id: openai/whisper-small.en, language: en, timeout_s: 10.0}   # whisper_hf | qwen_worker
  reply: {max_sentences: 2, max_new_tokens: 60, backchannel_enabled: true, backchannel_texts: ["Hm?", "Oh yeah?", "Let me think."], barge_in_min_speech_ms: 400, per_user_cooldown_s: 2.0, max_replies_per_minute: 12, history_turns: 8, llm_timeout_s: 20.0}
  llm: {provider: shared, base_url: "http://127.0.0.1:8000/v1", model: GLM-5.3-Flash-EXL3, reasoning_effort: low, timeout_s: 20.0}   # shared | openai_compat
  privacy: {log_transcripts: false}
```

Validators a test must check: `end_silence_ms >= min_speech_ms`, `max_sentences >= 1`, `respond_to in {all, wake_word}`, `llm.provider in {shared, openai_compat}`. `enabled` with an empty channel list loads, but `ListenCoordinator.start` refuses.

## 7. Task list

Legend: **[O]** orchestrator does it; **[S]** Sonnet builder; GPU / LIVE = needs the 4070 / a private Discord server with two humans. Every [S] task touches only its listed files, adds each new discord-free module to `NON_DISCORD_MODULES`, runs `ruff` and `pytest tests/unit`, and adds no dependency without an entry under `[project.optional-dependencies] listen` and `DEPENDENCY_NOTES.md`.

| ID | Who | Task | Accept (decided by a test or number) |
|---|---|---|---|
| P0a | S, LIVE | Receive + DAVE spike, scratch script and separate venv | Section 2 PASS; log versions and per-packet stats; a human confirms audio. Parallel with P0b |
| P0b | S, GPU | `tools/listen_bench_models.py`: one-sentence TTS+RVC timing x5; Whisper small.en and base.en on 3 clips; Qwen3-ASR if Path B; free VRAM with everything loaded | JSON `evidence/listen_models_benchmark.json`. **O reads the values** and fixes the A/B choice and budget |
| P1 | O | Interfaces, `ListenSettings`, `domain/listen.py`, protocols, fakes in `conftest.py`, example yaml, `speech_features_enabled` | Unit suite green; `AppSettings().listen.enabled is False`; example yaml validates. Blocks the rest |
| P2 | S | `listen/ingress.py`, `known_user_ids` | Users `{A, B, bot}` x 100 packets: decoder called exactly 200 times, all for A and B; `user_id=None` and the bot id never reach the decoder; `remove_user(A)` leaves `pending_samples(A) == 0`; ring caps at `ring_buffer_s*16000`. **O reviews invariant 1** |
| P3 | S | Energy and silero VAD, `listen/endpointer.py` | 0.3 s silence / 1.0 s tone / 0.8 s silence gives one event with duration in [1.0, 1.35] s and audio length `(0.3+1.0)*16000 +/- 1 frame`; 20 s tone is cut at 15 s |
| P4a | S | Whisper HF transcriber | Fake-model unit tests for normalization values; GPU test transcribes a fixture with non-empty text and `elapsed_s < 2.0`. **O reads GPU result** |
| P4b | S, optional | Qwen3-ASR worker (`speech/asr/qwen_worker.py`, `legacy/asr_worker.py`) | Protocol test with a stdlib fake worker; GPU test with the real one. Do not touch `.venv`, `rvc_cli` or the RVC worker |
| P5 | S | `listen/conversation.py`, `ContentKind.CHAT`, `build_chat_prompt` (additive only) | Echo guard true at ratio 0.7 and for identical text, false at 0.3; prompt has the speaker name and the last 2 turns only; reply trimmed to 2 sentences |
| P5b | S, optional | `content/openai_compat.py` | `httpx.MockTransport`: normal reply; `content: null, finish_reason: length` raises `EmptyModelText`; timeout raises. Not wired into music commentary |
| P6 | S | `listen/reply.py`: sentence synthesizer, `BackchannelCache`, announcement clip | Bytes length `== duration_s*48000*2*2 +/- 4`; artifact dir empty after; second call hits the cache (pipeline not called again) |
| P7 | S | `listen/turn_taking.py` with `ReplyPlayerPort` | With fakes and `FakeClock`: one transcribe + one generate per utterance, backchannel before reply; barge-in at 0.5 s calls `player.stop()` once and cancels the task, history has `"[interrupted"`; two utterances during THINKING give one generate with both texts; 13 utterances in a minute give 12 replies; LLM timeout returns to IDLE; the bot never barges in |
| P8 | S | `listen/metrics.py`, `tools/listen_bench.py` | Log line matches a regex with all five deltas; harness prints P50/P90 |
| P9 | S, after P0a | `discord_bot/listen_glue.py`, `!glisten`, `bot.py`, `app.py` lifecycle | `"listen"` in command registration test; guard tests with a fake ctx; lifecycle tests: listen off means no ASR load, listen on loads ASR after the LLM and before RVC start. **O reviews the drop-before-decode rule and mutual exclusion** |
| P10 | S | WAV harness: FakePacketSource through ingress, endpointer, TurnManager, ReplySynthesizer, FakeReplyPlayer | One reply PCM; plumbing overhead under 50 ms excluding fake sleeps; metrics line emitted; `--gpu` swaps in the real transcriber and pipeline |
| P11 | O + human, GPU + LIVE | Live checklist below | Results recorded in `TEST_RESULTS.md` as PASS / FAIL / NOT RUN |
| P12 | O | Squash per task onto `listen`; README, `DEPENDENCY_NOTES.md`, example yaml | n/a |

Order: P0a and P0b in parallel, then P1, then P2, P3, P4a, P4b, P5, P5b, P6, P8 in parallel, then P7, then P9 (needs P0a, P2, P7), then P10, P11, P12.

## 8. Test strategy

- **Unit:** fakes beside `FakeSynthesizer`; `FakeClock` for all timing; every `TurnManager` transition pinned with exact counts; the ingress bot/`None` drop pinned with exact decoder call counts; defaults pinned with exact values.
- **Integration without Discord:** the P10 harness, with a `--gpu` variant (markers `gpu`/`integration`, skipping cleanly when weights are absent).
- **GPU evidence:** `evidence/listen_models_benchmark.json` (P0b) and `evidence/listen_turn_latency.json` (P10 `--gpu`, 10 turns, P50/P90 per stage).
- **Live checklist** (private server, two humans, owner present):
  1. `!glisten start` outside an allowed channel is refused.
  2. Start in the allowed channel: announcement audible within 2 s, text notice posted.
  3. The bot's own voice is never transcribed: 20 s of bot speech triggers no self-reply (self-hearing guard).
  4. Human A talks with no opt-in step: backchannel within 1 s, Goku reply within 9 s, metrics line present.
  5. A interrupts mid-reply: playback stops within 500 ms and the next reply references the interruption.
  6. A and B talk over each other: two transcripts, one merged reply.
  7. A stops talking mid-sentence (or leaves the channel): the half-finished utterance is dropped, not replied to.
  8. `!gplay` during listen is refused.
  9. `!glisten stop`: bot disconnects, and `var/` has no user audio (`find var -newer <marker> -name '*.wav'` shows only bot clips).
  10. Ctrl+C during a reply: clean exit, RVC worker reaped, no orphan ASR worker.

## 9. Risks (ranked) and stop conditions

1. **DAVE receive not decodable.** Stop if P0a fails with both voice-recv and the in-house receiver; report, do not start Phase 2.
2. **Latency unacceptable.** If P0b shows one-sentence TTS+RVC above 5 s, raise a different TTS engine before polishing P6/P7. The backchannel masks the wait but does not fix it.
3. **VRAM.** ASR + LLM + TTS + RVC must stay under 14.5 GB steady. OOM in P0b stage 4 means Path B or a smaller ASR.
4. **Coordinator head-of-line blocking** (ASR waits for TTS). If `eos→asr` P90 exceeds 3 s in P10 `--gpu`, add a dedicated ASR lease as a follow-up, not before.
5. **Acoustic echo and self-trigger loops.** Mitigated in section 5. If loops persist, default to `respond_to: wake_word`.
6. **Receive thread starvation.** Decode only users that are not the bot; move resampling to the VAD thread if P10 shows more than 5 ms per packet.
7. **Environment drift.** Pin versions before P0; apply the PYTHONPATH rule for worktrees.
8. **Self-hearing guard regression by a builder** (moving the `None`/bot drop after decode, or dropping a human). Orchestrator-owned review of P2 and P9; the exact-count tests are the tripwire.

## 10. Owner decisions (2026-10-25)

| # | Decision |
|---|---|
| 1 | **Path A**: LLM stays in-process on the 4070. ~~Path B (Spark tunnel) optional later task~~ **Removed 2026-10-25: local only, see section 13** |
| 1b | **ASR: NVIDIA Parakeet TDT 0.6B** (see section 11), replacing the Whisper `small.en` default. Whisper stays as the baseline and fallback |
| 2 | Not answered; checked by the assistant (see below) |
| 3 | ~~Opt-ins are **per session**, not persisted~~ **Removed 2026-10-30: there are no opt-ins; everyone in the session channel is heard** |
| 4 | Music and listen are **mutually exclusive** per guild in v1 |
| 5 | If RVC is unavailable mid-session: **drop the reply**, log it, say nothing |

## 10a. Original open questions

1. Is the DGX Spark tunnel (127.0.0.1:8000, owner's own hardware, reasoning models) acceptable as "local" for listen mode, and is it up whenever the bot runs? This decides Path A or B.
2. Is PyPI reachable from this box for `discord-ext-voice-recv` and `silero-vad`, or must wheels be fetched elsewhere?
3. ~~Should opt-ins persist across restarts (a file under `var/`) or stay per session (proposed)?~~ **Moot since 2026-10-30: there are no opt-ins.**
4. Listen mode and music: mutually exclusive per guild (proposed), or should listen pause music on speech?
5. If RVC is unavailable mid-session: drop the reply (proposed, consistent with `rvc_unavailable_policy: error`) or speak in the source voice?

## Critical files

- `src/notamusicbot/speech/coordinator.py`: single-slot GPU lease; all new GPU work goes through it.
- `src/notamusicbot/speech/pipeline.py`: `SpeechPipeline.run`, the only sanctioned TTS to RVC to postprocess entry.
- `src/notamusicbot/app.py`: composition root; startup ordering where ASR load and listen lifecycle slot in.
- `src/notamusicbot/discord_bot/commands.py`: `_join_and_start`, and the home for `!glisten` and the music guards.
- `src/notamusicbot/config.py`: `AppSettings` / `ListenSettings` and the `enabled: false` flag.
- `.venv-main/lib/python3.12/site-packages/discord/voice_state.py`: verified receive primitives (`add_socket_listener`, DAVE session).

## 11. ASR decision update: Parakeet TDT 0.6B (2026-10-25, owner)

Owner chose NVIDIA Parakeet TDT 0.6B as the Path A ASR. Whisper `small.en` / `base.en` remain the benchmark baseline and the fallback backend (`asr.backend: whisper_hf`). The P0b run already in flight measures Whisper only.

Verified 2026-10-25:
- Models on Hugging Face, CC-BY-4.0: `nvidia/parakeet-tdt-0.6b-v2` (English), `nvidia/parakeet-tdt-0.6b-v3` (multilingual), and ONNX exports `istupakov/parakeet-tdt-0.6b-v2-onnx`, `istupakov/parakeet-tdt-0.6b-v3-onnx`.
- PyPI: `onnx-asr` 0.12.0 (lightweight ONNX ASR runner), `sherpa-onnx` 2.0.0, `onnxruntime-gpu` 1.31.0.
- **`.venv-main` has `onnxruntime` 1.31.0 CPU only** (providers: Azure, CPU). So the ONNX route runs on CPU unless the GPU build is swapped in.

Design consequences:
- **Run Parakeet on CPU first.** That removes ASR from the single-slot GPU `InferenceCoordinator` and from the head-of-line blocking risk (plan risk 4): ASR no longer waits behind a TTS chunk, and it frees about 1-2 GB of VRAM. CPU latency is unmeasured and may be too slow for a 15 s utterance; the bake-off decides.
- GPU option: `onnxruntime-gpu` conflicts with the CPU `onnxruntime` package in one venv, and whether its CUDA build matches the torch cu130 stack is unverified. Treat as a P0c experiment, not an assumption. NeMo (full toolkit) is the heavy fallback and would live in its own venv with a subprocess worker, like Path B.
- Backend choice: `onnx-asr` first (thin, one dependency), `sherpa-onnx` as the alternative.
- v2 (English only) vs v3 (multilingual): bake-off decides; default v2 for a Texas English-speaking channel.
- Parakeet does not emit Whisper's silence hallucinations ("Thank you."), but P4a still keeps the empty / under-2-word / known-phrase filter as a guard.

### P0c: ASR bake-off (new, before P4a)

Same 3+ clips and the same harness as P0b. Candidates: Whisper small.en, Whisper base.en, Parakeet TDT 0.6B v2 via `onnx-asr` on CPU, the same on GPU (if `onnxruntime-gpu` installs cleanly in a scratch venv, not `.venv-main`), Qwen3-ASR-1.7B via `voiceover/.venv-asr`. Report per model: load time, per-clip latency (warm), WER, VRAM delta, CPU time and cores used, silence behavior (feed 5 s of silence and low noise: does it emit text?), install steps. Clips: TTS-synthesized (optimistic); add a real recording only with the owner's explicit OK. Accept: JSON in `evidence/listen_asr_bakeoff.json`; the orchestrator reads the values and fixes the backend. Changes to P4a: it implements `ParakeetOnnxTranscriber` (and keeps `WhisperHfTranscriber`), config `listen.asr.backend: parakeet_onnx | whisper_hf | qwen_worker`, `listen.asr.model_id` and `listen.asr.device: cpu | cuda`.

## 12. Round 2 results (2026-10-25, orchestrator)

Landed (uncommitted, discord-free, registered in `NON_DISCORD_MODULES`): `listen/ingress.py` (P2), `speech/vad/{energy,silero}.py` + `listen/endpointer.py` (P3), `listen/conversation.py` + `ContentKind.CHAT` / `build_chat_prompt` (P5), `listen/reply.py` (P6), `listen/metrics.py` + `tools/listen_bench.py` (P8, log-line reader only). P0c: `tools/listen_bench_asr.py`, `evidence/listen_asr_bakeoff.json`. Unit suite after integration: `386 passed, 1 skipped` (skip = silero not installed). `ruff check src tests` reports 3 errors, all in committed files no round touched (`test_failure_fallback.py` I001, `test_hubert_loader_parity.py` 2x E501); every listen file is clean.

**Consent invariant 1 (P2): holds.** `allowed()` runs before `decode()`; consent is re-checked under the per-user buffer lock after decode, before the append and `on_pcm`. `opt_out` removes the user before running clear hooks, so a late packet either re-checks False or appended before the hook clears it. Nuance: a packet that passed the first check just before `opt_out` is decoded (consent existed when it arrived) but never reaches the buffer, `on_pcm`, VAD or ASR. Tripwire: the threaded race test passed 50/50 with the re-check removed (it rarely hits the window), so the orchestrator added a deterministic test (`opt_out` inside `decode`) that fails 5/5 without the re-check. Downstream gap: the endpointer drops a user only on its next poll, so P7 must re-check `consent.allowed(user_id)` on each `UtteranceEvent` before ASR and again before the LLM call. P9: member leave must call `consent.opt_out`, not only `ingress.remove_user`.

**ASR decision (P0c): Parakeet TDT 0.6B v2, int8 ONNX, CPU, `onnx-asr`, 4 intra-op threads.**

| Candidate | ~5 s clip latency | 15 s clip (RTF) | WER, 4 clips | Silence / -50 dBFS noise | VRAM delta |
|---|---|---|---|---|---|
| **Parakeet v2 int8 CPU** | 0.26-0.29 s | 0.616 s (0.042) | 0 | no text | 0 |
| Parakeet v2 fp32 CPU, 4 threads | 0.23-0.26 s | 0.597 s (0.041) | 1 error ("to morrow") | no text | 0 |
| Parakeet v2 fp32 GPU (onnxruntime-gpu, scratch venv) | 0.02-0.03 s | 0.066 s (0.005) | 1 error | no text | +3562 MiB |
| Whisper small.en GPU | 0.06 s | 0.159 s (0.011) | 0 | "you" on both | +817 MiB |
| Whisper base.en GPU | 0.04 s | 0.095 s (0.007) | 0 | "you" on both | +173 MiB |
| Qwen3-ASR-1.7B GPU | 0.21-0.24 s | 0.66 s (0.045) | 0 | "Okay." on both | +4810 MiB |

Why: CPU Parakeet beats the 0.3-0.6 s per 5 s target, needs no GPU slot (removes risk 4, head-of-line blocking behind TTS) and no VRAM, and emits nothing on silence. GPU Parakeet would put steady state at about 14.9 GB (over the 14.5 GB ceiling) and needs `onnxruntime-gpu` + pip cuDNN, which conflicts with the CPU `onnxruntime` in `.venv-main`. int8 over fp32: loads straight from the HF cache (fp32 needs the hard-link workaround) and had no errors. Threads: ORT default spins about 22 of 24 cores for no gain (4 threads: same 15 s latency at about 8.7 effective cores), so pin 4 or ASR starves the VAD thread. Whisper `small.en` stays the fallback backend.

Bake-off trust: same 4 clips for all, 1 warm-up then median of 3, silence tests on every candidate, GPU providers verified per session and the 2-attempt cuDNN workaround recorded. Caveats: clips are synthetic clean TTS (WER saturated near 0), loads were from a warm disk cache, CPU-row VRAM deltas are noise. **Unmeasured:** real microphones, accents and names; audio after P2's boxcar 48->16 kHz resampler; int8 with 4 threads; latency while the LLM and TTS are busy; cold load.

Other round-2 findings: P3 assumed `ingress.drain(user_id)`, which matches P2. `SileroVad` calls are unverified until `silero-vad` is installed. P6 needs `rvc_unavailable_policy: error` for listen and `artifact_root` passed in, or "drop the reply" and "no files left" do not hold. P5's "ratio 0.7" echo case computes 0.667 (above the 0.6 threshold; accepted).

## 13. Local-only constraint (owner, 2026-10-25)

**The listen system makes no API calls to any AI or speech service.** Everything that understands, thinks or speaks runs on this desktop: VAD (CPU), ASR (Parakeet on CPU, Whisper small.en fallback on GPU), the LLM (Qwen3-4B in-process on the 4070), TTS (Qwen3-TTS) and RVC. The Discord connection itself is the only network traffic the feature needs, because that is the channel the bot lives in.

Consequences:
- **Path B is dropped.** No `content/openai_compat.py` (task P5b), no Qwen3-ASR worker against the Spark (P4b), no use of the DGX Spark tunnel at 127.0.0.1:8000. The Spark and its models are not part of this feature. Cancel P4b and P5b.
- **Config:** `listen.llm` (base_url, model, reasoning_effort) is dead. Shrink `llm.provider` to `shared` only and remove `openai_compat` from the validator; round 1 left both values in `ListenSettings`. Follow-up task for the orchestrator.
- **No phoning home at runtime:** set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for the bot process once the models are cached, plus `HF_HUB_DISABLE_TELEMETRY=1`. Model downloads are a one-time setup step done by hand (`scripts/setup.sh` or a documented tool), never at bot start. A test must fail the build if the listen path imports `httpx`, `requests`, `openai` or `huggingface_hub` downloads at runtime (allow the Discord library).
- **Models to cache before offline:** `openai/whisper-small.en` (already cached), `istupakov/parakeet-tdt-0.6b-v2-onnx` int8 (already cached), `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` and `Qwen/Qwen3-4B-Instruct-2507` (cached), silero VAD ONNX (needs `pip install silero-vad` once; it ships its model in the wheel).
- **Existing music features are unchanged** (yt-dlp, YouTube, Spotify search). They are separate from listen mode and not covered by this constraint.
- Transcripts and audio stay on this machine, in memory, per plan section 6.

## 14. Round 3 results (2026-10-25, orchestrator)

Landed (uncommitted, discord-free, all registered in `NON_DISCORD_MODULES`; every one sits under an N1 scan target, so the scan list needed no change):

| Task | Files | Tests |
|---|---|---|
| P4a + config | `speech/asr/{_common,parakeet_onnx,whisper_hf,factory}.py`; `config.py` (`ListenAsrSettings`: `parakeet_onnx` default, `whisper_model_id`, `device`, `threads=4`, `quantization=int8`; `ListenLlmSettings` and `listen.llm` **removed**); example yaml; `listen` extra (`silero-vad>=6.0`, `onnx-asr>=0.12`); `DEPENDENCY_NOTES.md` | `test_listen_asr.py` 15, `test_settings.py` 9 |
| P7 | `listen/turn_taking.py` (`TurnManager`, `TurnState`, `ReplyPlayerPort`, `ClockPort`, `MonotonicClock`) | `test_listen_turn_taking.py` 31 (28 builder + 3 orchestrator), fakes in `_listen_turn_fakes.py` |
| V1 | `speech/vad/silero.py` verified against silero-vad 6.2.3; restores the torch thread count that `import silero_vad` sets to 1 | `test_vad_silero.py` 8 |
| N1 | `listen/offline.py` (`enforce_offline_models`, `assert_offline`) | `test_listen_local_only.py` 17 |

Installs into `.venv-main` (dry-run first, one package each): `onnx-asr==0.12.0`, `silero-vad==6.2.3`.

**Suite after integration:** `pytest tests/unit -q` = **458 passed**, 0 skipped (silero and the Parakeet integration test now run). `ruff check src tests` = the same 3 pre-existing errors as round 2 (`test_failure_fallback.py` I001, `test_hubert_loader_parity.py` 2x E501); `ruff format --check` flags only `test_hubert_loader_parity.py` (committed, untouched). Every listen file is clean. No round-3 builder edited `conftest.py`, `README.md`, ingress/endpointer/conversation/reply/metrics, `.venv`, `rvc_cli` or `legacy/`.

**SileroVad measured (6.2.3, CPU, 1 thread, 32 ms frames):** silence max 0.009 (mean 0.005); synthetic voiced signal (formant-filtered 120 Hz sawtooth) min 0.159, mean 0.617, max 0.894; white noise max 0.042; latency mean 0.10-0.13 ms per frame. Endpointer with the silero factory on 0.3 s silence / 1.0 s voiced / 1.0 s silence: one `SpeechStarted` at 0.288 s, one utterance of 0.992 s. Threshold 0.5 is **not yet confirmed on real speech**.

**Parakeet by hand (orchestrator):** default settings, `HF_HUB_OFFLINE=1`, run inside `unshare -rn` (no network interface at all), CPU, 4 threads: load 1.46 s; warm median of 3 per bake-off TTS clip: 4.1 s clip 0.18 s, 4.2 s 0.38 s, 5.0 s 0.21 s, 14.6 s 0.41 s; all four transcripts exact. Still synthetic clips only.

**Consent verdict (P7): holds.** Checks: (1) before ASR in `on_utterance`; (2) after ASR in `_accept`, before the transcript is logged or queued; (3) before the LLM call in `_run_turn`, applied to **every** merged line, so a merged second user is checked too; (4) barge-in on onset and again after the sustain timer. Mutation results (each reverted at once): removing (1) fails `test_never_opted_in_user_is_not_transcribed`; removing (3) fails the merge-revocation test; removing the first-sentence `ReplySynthesisError` drop fails `test_synthesis_error_drops_reply`; removing the onset barge-in check fails both `test_unknown_or_bot_speech_never_barges_in` cases. Removing (2) alone **killed no builder test**, because (3) still blocked the LLM. But without (2), a revoked user's transcript is still logged at DEBUG when `log_transcripts: true`, and its pending entry can push out a consenting speaker. The orchestrator added a tripwire that fails without (2). It also added a mid-reply synthesis-failure test (sentence 2 fails: only sentence 1 plays). `TurnManager` stored `bot_user_id` and never used it; it now drops the bot id itself (`_allowed`) even if `ConsentState` was built without it, and a test covers that. Nuance: a user who opts out after the turn has reached the LLM still gets that reply (the line was consented when committed); "off mid-sentence" (live item 7) is covered by ingress + the pre-ASR check.

**Local-only verdict: holds for the code written so far.** No listen, asr, vad or domain module imports a network library (the N1 AST scan enforces this). Loader imports are allowed only in `speech/asr/` next to `local_files_only=True` (Parakeet uses `snapshot_download(local_files_only=True)`, Whisper `from_pretrained(local_files_only=True)`). The config has no `llm`, `base_url` or `api_key` field: the N1 config test is now a hard test, no longer xfail. Nothing references the Spark tunnel, `127.0.0.1:8000` or any API. Open: nothing calls `enforce_offline_models()` yet; P9 must call it before any listen model loads.

Orchestrator edits this round: registered the 6 modules; turned the N1 `base_url` xfail into a hard guard that also rejects `api_key`/`api_base` and any `llm` field; added `TurnManager._allowed` (bot id); added 3 tripwire tests to `test_listen_turn_taking.py`. The silero torch-thread test passed even with the restore removed, because an earlier test had already imported `silero_vad`. It now re-imports `silero_vad` fresh and fails without the restore.

Known gaps: Whisper's real `load()`/`transcribe` has never run (the fake overrides `_recognize`). The Parakeet `device: cuda` and `quantization: fp32` paths are untested (the CPU onnxruntime has no CUDA provider; the fp32 external-data file may hit the symlink rule). `parakeet_onnx.load()` is not locked, so P9 must call `load()` once at startup (two first calls at the same moment could both load). `speech/asr/_common.py` imports `listen.conversation` (speech -> listen, a layering inversion, harmless today). The Parakeet integration test depends on `~/.cache/listen-asr-bakeoff/clips/clip1.wav` and skips on a fresh checkout. A user's opt-out after the LLM call does not cancel that reply.

| ID | Status | Note |
|---|---|---|
| P0a | **BLOCKED** (owner) | Live receive + DAVE with two humans; gates P9 |
| P0b, P0c, P1-P3, P5, P6, P8 | DONE | rounds 1-2 |
| P4a | DONE | Parakeet default + Whisper fallback; GPU Whisper path unverified |
| P4b, P5b | CANCELLED | section 13 |
| P7 | DONE | consent reviewed, tripwires added |
| V1 (silero verify) | DONE | real-voice threshold check outstanding |
| N1 (local-only guard) | DONE | P9 must call `enforce_offline_models()` |
| P9 | **BLOCKED** | on P0a, the `content.enabled` gap, and the member-leave rule (`consent.opt_out`, not only `ingress.remove_user`) |
| P10 | READY | WAV harness over ingress -> endpointer -> TurnManager -> ReplySynthesizer -> fake player |
| P11 | BLOCKED | after P9/P10, GPU + LIVE |
| P12 | BLOCKED | after P11 |

## 15. P10 results (2026-10-25, orchestrator)

Landed (uncommitted): `tests/integration/test_listen_wav_harness.py` (3 tests, fakes, real `time.monotonic` clock), `tools/listen_pipeline_bench.py` (real chain, not pytest), `evidence/listen_turn_latency.json`.

**Harness (fakes):** FakePacketSource -> ListenIngress -> Endpointer(EnergyVad) -> TurnManager -> real ReplySynthesizer over FakeSynthesizer/FakeConverter -> recording player. One reply PCM, one parseable metrics line, no wav left. Plumbing overhead 0.6 ms (budget 50 ms), measured endpointer emit -> first `synthesize_sentence`, minus the fakes' 30 + 30 ms sleeps. Stranger: 0 decodes, 0 events, 0 transcribe calls; a forged stranger `UtteranceEvent` handed to the manager: 0 transcribe calls.

**Real run:** SileroVad, Parakeet v2 int8 CPU (4 threads), Qwen3-4B in-process (`max_new_tokens` 60), Qwen3-TTS + RVC goku (`rvc_unavailable_policy: error`, `artifact_root` set), 3 pre-rendered backchannel clips. `enforce_offline_models()` first; models from the local cache only. Input: 5 synthetic plain-TTS questions (4.88-5.60 s), cycled to 10 turns, fed in real time as 20 ms packets. Warm-ups (RVC via `warm_voice_converter`, TTS x2, ASR x1, LLM x1) ran before timing. 10/10 turns produced a metrics line.

**Measurement review: honest; n=10 is too small for P90.** Checked in code: `t_eos` is the endpointer's last voiced frame (endpoint wait 0.60-0.62 s, so not the feed end); the backchannel is timed at the player and identified by PCM; VRAM sampled every 250 ms (1395 samples, per phase); the not-opted-in turn (317 packets) gave 0 decodes, 0 buffered samples, 0 ASR, 0 LLM, 0 metrics, 0 plays; spoken sentences counted from plays (all <= 2); input audio synthetic and in memory, backchannel dir deleted, no new files under `var/`. Two differences from production: the tool sets `provider._max_new_tokens = 60` (Application wiring would use `content.max_new_tokens` 120), and it hands both loaders a local snapshot directory instead of the hub id (same weights; needed because of the defect below). Replies are "played" by sleeping their duration.

| Line (section 4) | Target | Measured | Verdict |
|---|---|---|---|
| Total, end of speech -> first reply audio, P50 | <= 6 s | 4.63 s | PASS |
| Total P90 | <= 9 s | 5.11 s (nearest rank, n=10); max 9.10 s | PASS, provisional: 1/10 over 9 s; rerun with n >= 30 |
| End of speech -> backchannel | <= 0.8 s | P50 0.788, P90 0.803, max 0.815 s; 5/10 over | MARGINAL / FAIL per turn: 0.60-0.62 s endpoint silence + 0.15-0.20 s ASR before the clip starts |
| ASR per ~5 s clip | 0.3-0.6 s | 0.154-0.200 s | PASS (better than target; LLM/TTS idle while ASR ran) |
| LLM (whole reply, 60 tokens) | 1-2 s to first sentence | 0.73-1.13 s, P50 0.83 | PASS |
| TTS + RVC first sentence (`llm->tts1`) | 2.5-3.5 s | P50 2.21, P90 3.55, range 1.30-7.37 s | PASS P50, at limit P90, max over |
| RVC warm | 0.4-0.6 s | not isolated in this run | NOT MEASURED here; `rvc_same_model_warm.json` has one warm convert at 0.573 s |
| Backchannel wait (`tts1->audio`) | (not in budget) | P50 0.00, P90 1.16, max 1.63 s | 4/10 turns wait for the clip to finish |
| VRAM steady (device total) | <= 14.5 GB | 12119 MiB after load, 13053 MiB idle after warm-up | PASS |
| VRAM PEAK (device total, new) | none set | 14863 MiB (14.51 GiB) during turns, 1513 MiB free of 16376 | AT CEILING; includes 1993 MiB baseline (Discord desktop client, not the bot), so bot-attributable about 12.9 GiB |

**`llm->tts1` is explained, not noise:** `llm->tts1 = 0.27 s + 0.69 x first-sentence audio seconds`, r = 0.9996 over all 10 turns. Turn 10 (9.10 s total) is on the line: its first sentence was 23 words joined by an em dash (10.3 s of audio). `content/trim.py` splits only on `.!?`. The plan's "12 words, 2 s audio" assumption was wrong: the measured rate is about 1.6-2.7 words per second.

**Findings beyond the budget table:**
1. **Silence between sentences.** Sentence 2 synthesis starts when sentence 1 starts playing, so a long sentence 2 leaves dead air: 11.83 s (turn 3), 8.63, 8.11, 7.54, 5.90, 3.87, 3.29, 1.65, 0.57, 0.00 s. This is the worst user-facing defect in the run. When the backchannel blocks, sentence 2 also starts late.
2. **Suspected TTS over-generation.** Turn 3 sentence 2: 22 words in 20.7 s (1.06 words/s); turn 8 sentence 2: 12 words in 13.34 s (0.90 words/s); others 1.3-2.7 words/s. Audio was deleted, so it is unverified (silence or babble).
3. **Backchannel clips are long:** 1.50 / 2.30 / 3.98 s for "Hm?", "Oh yeah?", "Let me think." (probably padded). They cause the 1.16-1.63 s `tts1->audio` waits.
4. **Reply text:** the LLM wrote 3 sentences on 4/10 turns (trimmed to 2). `trim_reply` counted "Hey!" as sentence 1 on turn 8, so only one content sentence was spoken. Turn 9 contained an emoji that reached TTS unstripped. The system prompt is still the radio-host prompt.
5. **src defect (blocks P9, not fixed):** with `HF_HUB_OFFLINE=1`, `AutoTokenizer.from_pretrained(<hub id>)` reaches transformers' `_patch_mistral_regex` -> `huggingface_hub.model_info()` and raises `OfflineModeIsEnabled` (verified in `tokenization_utils_base.py`). `content/llm.py` load then fails and `Application.load_content_model` silently installs the static template provider (no LLM). `speech/tts/qwen_custom_voice.py` fails the same way (fatal in `build_pipeline`). Fix: resolve the id with `snapshot_download(id, local_files_only=True)` in both loaders.
6. **Acceptance gap:** P10 asked for a `--gpu` pytest variant with `gpu`/`integration` markers; the real run is a standalone tool instead. Evidence is equivalent; no marked GPU test exists.

**Levers, ranked by measured saving (biggest stage: `llm->tts1`, first-sentence TTS + RVC):**

| # | Lever | Expected saving |
|---|---|---|
| 1 | Shorter first sentence: split on em dash / semicolon / comma clause, cap the first chunk at about 8-10 words (3.5-4 s audio) | `llm->tts1` P90 3.55 -> about 2.8 s, max 7.37 -> about 2.8 s; the same cap on later chunks shrinks the dead air in finding 1 |
| 2 | Trim / re-render the backchannel clips to about 1 s or less | `tts1->audio` P90 1.16 -> about 0 s on 4/10 turns |
| 3 | Start the next sentence's synthesis as soon as the previous one is synthesized (not when it starts playing); stream the LLM and start TTS at the first sentence boundary | gaps shrink; streaming saves about half of `asr->llm`, about 0.4 s P50 (needs a streaming check of `LocalQwenLlmProvider`) |
| 4 | Endpoint silence 600 -> 400 ms | 0.2 s off every turn and the backchannel (0.788 -> about 0.59 s, passes); risk of cutting real speech, unverified |
| 5 | Start the backchannel at endpoint time, before ASR | 0.15-0.20 s off the backchannel only (passes at 600 ms silence) |
| 6 | `max_new_tokens` 60 -> 40 | about 0.25-0.3 s off `asr->llm`; more truncation risk |
| 7 | RVC settings | unranked: no per-stage RVC figure here; P0b's warm TTS+RVC RTF 0.74-0.78 matches the 0.69 slope, so RVC is not the dominant term |

**Still unmeasured:** real microphones, accents and names (input is clean synthetic TTS); Silero threshold 0.5 on real voice; Discord Opus decode, DAVE receive and real playback latency (P0a); concurrent music (not applicable in v1: owner decision 4, mutually exclusive); ASR while the LLM or TTS is busy (turns never overlapped); cold load; any run without the Discord desktop client holding VRAM; P90 at n >= 30.

**Suite:** `pytest tests/unit tests/integration/test_listen_wav_harness.py -q` = **461 passed** (458 unit + 3 harness) in 56.5 s. `ruff check src tests` = the same 3 pre-existing errors (`test_failure_fallback.py` I001, `test_hubert_loader_parity.py` 2x E501); `ruff format --check` flags only `test_hubert_loader_parity.py`. Both new files are clean.

| ID | Status | Note |
|---|---|---|
| P0a | **BLOCKED** (owner) | live receive + DAVE with two humans; gates P9 |
| P0b, P0c, P1-P3, P4a, P5-P8, V1, N1 | DONE | rounds 1-3 |
| P4b, P5b | CANCELLED | section 13 |
| P10 | DONE, caveats | n=10, synthetic input, no marked `--gpu` pytest |
| P10b | READY | levers 1-3 + emoji strip + interjection rule in `trim_reply`, then rerun the bench at n >= 30 |
| P9 | **BLOCKED** | on P0a. Must also: resolve model ids to local snapshots in `content/llm.py` and `qwen_custom_voice.py` (finding 5); wire `listen.reply.max_new_tokens`; call `enforce_offline_models()` before any load; chat system prompt instead of the radio prompt; member leave calls `consent.opt_out`; `content.enabled` gap |
| P11 | BLOCKED | after P9 + P10b, GPU + LIVE |
| P12 | BLOCKED | after P11 |

## 16. P10b results (2026-10-25, orchestrator)

**Headline: the code changes landed and work, but the end-to-end latency got worse, not better.** Lever 1 predicted `llm->tts1` P90 3.55 -> about 2.8 s. Measured: P50 2.21 -> 3.27, P90 3.55 -> 4.92 s. Two causes are visible in the data: (a) the reply text changed. The chat system prompt replaced the radio prompt, and interjections now merge. At n=10 the first sentence was often a short opener ("Oh, hey there!"); at n=30 the first chunk has 5-10 content words. Median first-chunk audio rose from 2.94 to 4.42 s despite the 10-word cap. (b) **TTS over-generation is now the dominant cause of the tail.** 7 of 73 chunks ran under 1.3 words/s (0.65-0.94 w/s; for example 8 words gave 12.38 s and 16 words gave 16.94 s). The `llm->tts1` fit fell from r = 0.9996 to r = 0.72 (`-0.08 + 0.81 x audio`). This is confirmed by count but still not confirmed by listening: the audio was deleted again, so silence and babble cannot be told apart.

Landed (uncommitted; no new modules, so `NON_DISCORD_MODULES` needed no change):

| Task | Files | Tests |
|---|---|---|
| B1 chunking + cleanup | `listen/conversation.py`: `clean_for_speech` (emoji, markdown, `*stage*` / `(paren)` directions), `split_reply_sentences` (a short "Hey!" merges into the next sentence and does not count toward the cap), `split_spoken_chunks` (first chunk <= 10 words, later <= 18; cuts at a dash, then `;`, `:`, `,`, then a word boundary; never ends a chunk on a stop word; leaves no 1-2 word tail) | `test_listen_conversation.py` 20 |
| B2 pipelining + backchannel trim | `listen/turn_taking.py`: `_chunk_plan`, a producer task into `asyncio.Queue(maxsize=2)`, barge-in also cancels the producer, history holds the spoken text. `listen/reply.py`: `trim_backchannel` (-45 dBFS, 80 ms fade, 1.2 s cap; the trim version and parameters are part of the cache key). `_listen_turn_fakes.py`: `TimelineClock`, `TimedSynth`, `TimedPlayer` | turn_taking 36, reply 14 |
| B3 offline loaders + chat prompt | `content/llm.py` `resolve_local_model_path` (`snapshot_download(local_files_only=True)`, a clear `RuntimeError` naming `huggingface-cli download <id>`), used by `speech/tts/qwen_custom_voice.py`; `prompts.CHAT_SYSTEM_PROMPT` for `ContentKind.CHAT` | `test_offline_loaders.py` 8 |
| B4 bench n=30 | `tools/listen_pipeline_bench.py`, `evidence/listen_turn_latency_n30.json` | n/a |

**Review.** Pinned values were checked against the code (for example the 22-word input gives `["I really think that the best food in the whole", "world is ramen — especially the spicy kind from the little shop downtown."]`, and timed playback starts at 1.0, 4.0 and 7.0 s with synthesis starts at 0.0, 1.0 and 2.0 s). The builder test files are untracked, so there is no git baseline. Instead, a read-through confirmed that every P7 acceptance test and every round-3 tripwire is still present. The one disclosed change, 24000 -> 23999 frames in the cache test, is correct: sample 0 of the fake tone is sin(0) = 0, which is below -45 dBFS and gets trimmed. No forbidden file was touched. `config.py`, `conftest.py`, `test_no_discord_imports.py` and `README.md` all have modification times earlier than the round. No user audio is on disk: the bench input is synthetic and in memory, the backchannel directory was deleted, and no `.wav` is newer than the P10 evidence. **Consent: holds.** Pipelining kept all four checks: pre-ASR (`on_utterance`), post-ASR (`_accept`), pre-LLM on every merged line (`_run_turn`), and barge-in on onset and again after the sustain timer. **Local-only: holds.** The listen code adds no network import. Caveat: `content/llm.py` and `speech/tts/` are outside the N1 `SCAN_TARGETS`. Their new `huggingface_hub` import sits next to `local_files_only=True`, but the scan does not cover it.

**Mutation tripwires** (each reverted at once and checked with `cmp`):

| Mutation | Result |
|---|---|
| (a) `split_spoken_chunks` uses `split_sentences` (no interjection merge) | KILLED: `test_chunks_interjection_merges` (`'Hey!' != 'Hey! I love ramen.'`) |
| (b) chunk k+1 synthesis waits until chunk k starts playing | KILLED: `test_pipelined_playback_has_no_gap_between_chunks` (`[0.0, 1.0, 4.0] != [0.0, 1.0, 2.0]`). Only chunk 3 can expose it: chunk 2's synthesis starts at play 1 in both designs |
| (c) LLM loader / TTS loader passes the hub id | KILLED: `test_llm_loader_passes_snapshot_path_not_hub_id` and `test_tts_loader_passes_snapshot_path_not_hub_id` |
| (d) post-ASR consent re-check removed | KILLED: `test_post_asr_recheck_drops_revoked_user_before_logging_or_queueing` |
| extra: pre-ASR check removed | KILLED: `test_never_opted_in_user_is_not_transcribed`, `test_bot_id_dropped_even_if_consent_state_lacks_it` |
| extra: pre-LLM filter removed | KILLED: `test_revoked_during_other_users_asr_is_filtered_from_merge` |
| extra: onset barge-in consent check removed | KILLED: `test_unknown_or_bot_speech_never_barges_in[50]` and `[999]` |

**Bench honesty: acceptable.** The snapshot workaround is gone and the loaders resolve hub ids under `HF_HUB_OFFLINE=1`. Production `Application` wiring is used for load, pipeline and RVC, with the production `TurnManager` and the trimmed `BackchannelCache`. Warm-ups ran before timing (RVC, TTS x2, ASR, LLM). 30/30 turns produced metrics. The not-opted-in turn (365 packets) gave 0 decodes, 0 buffered samples, 0 ASR, 0 LLM, 0 metrics and 0 plays. Remaining differences from production: the tool still sets `provider._max_new_tokens = 60` (disclosed); 3 of 8 clips are over the 6 s spec (6.08, 6.48, 6.0 s); the player sleeps for the PCM duration (faithful to audio length, not to Discord jitter); the input is synthetic TTS. The prompt change makes n=10 vs n=30 a comparison of different reply texts, not a like-for-like rerun.

| Line (section 4) | Target | n=10 (P10) | n=30 (P10b) | Verdict |
|---|---|---|---|---|
| Total P50 | <= 6 s | 4.63 | 4.75 | PASS (slightly worse) |
| Total P90 | <= 9 s | 5.11 | 6.40 | PASS (worse) |
| Total max | none | 9.10 | 11.54 (turn 9; 1/30 over 9 s) | tail |
| `eos->asr` P50 / P90 | n/a | 0.79 / 0.80 (max 0.81) | 0.79 / 0.82 (max 1.60, turn 9: ASR 1.005 s) | ok; turn 9 outlier |
| `asr->llm` (whole reply) P50 / P90 | 1-2 s | 0.83 P50 | 0.70 / 0.97 | PASS |
| `llm->tts1` P50 / P90 / max | 2.5-3.5 s | 2.21 / 3.55 / 7.37 | 3.27 / 4.92 / 8.80 | P50 PASS, **P90 FAIL** |
| `tts1->audio` P90 | 0 | 1.16 | 0.00 (all 30) | PASS: the trim fixed it |
| First reply audio (player) | = total | n/a | P50 4.747, P90 6.401 | matches the metrics to the ms |
| Backchannel P50 / P90 / max | <= 0.8 s | 0.788 / 0.803 / 0.815 | 0.793 / 0.823 / 1.604 | **FAIL**: 10/30 (33%) over vs 5/10 |
| Backchannel clip lengths | about 1 s | 1.50 / 2.30 / 3.98 | 0.367 / 0.878 / 1.200 | PASS. "Let me think." hit the 1.2 s cap: possibly cut, check by ear |
| Dead air between chunks P50 / P90 / max | ~0 | max 11.83 (n=9 gaps) | 0.0 / 2.715 / 8.703 (n=43) | **FAIL**: improved; median 0 |
| Emoji / markdown reaching TTS | 0 | 1 turn | 0 / 30 | PASS (the LLM emitted none either) |
| VRAM peak, device | <= 16.4 GB with Discord about 2 GB | 14863 (baseline 1993) | 13942 MiB (baseline 1676) | PASS: bot about 12.27 GB (was 12.87); + 2 GB client = about 14.3 of 16.4 GB, about 2 GB headroom |
| VRAM steady after load | <= 14.5 GB | 12119 | 11802 MiB | PASS |
| Sentences spoken <= 2 | yes | yes | yes (2 on 25 turns, 1 on 5) | PASS |

**Remaining levers, ranked by saving computed from the n=30 rows (none implemented):**

| # | Lever | Expected saving (from measured rows) |
|---|---|---|
| 1 | **New: bound TTS generation length per chunk** (for example max audio tokens proportional to words, or trim trailing non-speech) after confirming by ear what the slow chunks contain | removes the 7 chunks under 1.3 w/s; total max 11.54 -> about 6.5 s, `llm->tts1` P90 4.92 -> about 4 s, dead-air max 8.70 / 5.87 / 4.85 s down to the gap structure in lever 3 |
| 2 | Start the backchannel at endpoint time, before ASR | backchannel P50 0.793 -> 0.601, **0/30 over** (turn 9: 1.604 - 1.005 = 0.599). Fixes the line alone; no speech-cut risk; the clip then plays before the noise and echo filters (for example a cough gets "Hm?") |
| 3 | **New: shorter caps.** First chunk 10 -> about 6 words; later 18 -> about 10-12 | the first chunk runs at 1.72 w/s median, so 8 -> 6 words cuts about 1.2 s of audio, x 0.81 = about 0.9 s off `llm->tts1` P50. Chunk k+1 plays without a gap only if its audio is <= about 1.2 x chunk k's; 13 chunks had 14+ words. Dead-air P90 2.7 -> about 1 s (estimate) |
| 4 | Endpoint silence 600 -> 400 ms | -0.2 s on every turn: total P50 4.55 / P90 6.20; backchannel P50 0.593, 1/30 over. Risk of cutting real speech at pauses; unverified on real voices |
| 5 | Stream the LLM and start TTS at the first sentence boundary | `asr->llm` P50 0.70 / P90 0.97 is the whole reply; about 0.35 s P50 / 0.5 s P90 |
| 6 | `max_new_tokens` 60 -> 40 | about 0.2 s; raw replies reach 216 chars (about 50 tokens), so the top third would cut sentence 2. Last |

Levers 2 + 4 together put the backchannel P50 at 0.40 s.

**Code-review notes (none block):** (1) The N1 scan does not cover `content/llm.py` or `speech/tts/`, and the loader rule is asr-only. Extend the rule and the targets deliberately, not blindly. (2) `_chat_system_prompt`: a custom `content.system_prompt` (a commentary persona) overrides `CHAT_SYSTEM_PROMPT` in voice chat. P9 needs a `listen` setting for it (`config.py`). (3) A missing snapshot now raises a clear `RuntimeError`, but `Application.load_content_model` still silently installs the static provider: finding 5's second half, open for P9. (4) `trim_reply` is dead in `src` (tests only). (5) `speech.tts` -> `content.llm` is a lazy import, and `content` already imports `speech.coordinator`: a harmless cycle. (6) Chunk caps are hard-coded (10/18) in `conversation.py`, not in `ListenSettings`. (7) Chunk 2's synthesis still waits for the backchannel to finish. It could start right after chunk 1 is synthesized, but this is moot while `tts1->audio` is 0. (8) A dash glued inside a word (`myself—no`) is never a cut point.

**Suite:** `.venv-main/bin/python -m pytest tests/unit tests/integration/test_listen_wav_harness.py -q` = **488 passed** in 61.2 s (461 + 27 new). With `CUDA_VISIBLE_DEVICES=` (GPU hidden), 3 GPU-dependent tests outside listen fail: `test_rvc_variant::test_real_worker_obama_variants_differ` (`No CUDA GPUs are available`) and `test_youtube_artist_parity` x2. `ruff check src tests` = the same 3 pre-existing errors (`test_failure_fallback.py` I001, `test_hubert_loader_parity.py` 2x E501). `ruff format --check` flags only `test_hubert_loader_parity.py`. Orchestrator edits this round: none in code; this section only.

| ID | Status | Note |
|---|---|---|
| P0a | **BLOCKED** (owner) | live receive + DAVE with two humans; gates P9 |
| P0b, P0c, P1-P3, P4a, P5-P8, V1, N1 | DONE | rounds 1-3 |
| P4b, P5b | CANCELLED | section 13 |
| P10 | DONE, caveats | section 15 |
| P10b | DONE, caveats | code landed and tripwires pass. Latency regressed (prompt change + TTS over-generation); backchannel and dead air still FAIL |
| P10c | READY | (1) keep the bot audio of slow chunks (bot audio only, scratchpad) and check silence vs babble by ear; (2) bound TTS length per chunk; (3) backchannel at endpoint time; (4) caps 6/12; rerun n=30 with the same prompt |
| P9 | **BLOCKED** | on P0a. Also: `enforce_offline_models()` before any load; wire `listen.reply.max_new_tokens`; a listen chat-prompt setting so `content.system_prompt` does not leak in; no silent static-provider fallback for listen; member leave calls `consent.opt_out`; `content.enabled` gap |
| P11 | BLOCKED | after P9 + P10c, GPU + LIVE |
| P12 | BLOCKED | after P11 |

## 17. Reply model: the owner's Spark system (owner, 2026-10-25; amends section 13)

Owner direction: responses come from the **local Spark system, whatever model it is serving**. This amends section 13: "local only" means the owner's own hardware (this desktop and the Spark ring over the loopback SSH tunnel at `127.0.0.1:8000`), never a third-party cloud or paid API. Section 13's ban on the Spark tunnel and its cancellation of P5b are **reversed**; the ban on any other network service stands.

Measured live 2026-10-25 against `127.0.0.1:8000` (model `deepseek-v4.1-flash`, read from `/v1/models`; the ring may change models, so the bot must not hardcode an id):

| Request (Goku voice-chat system prompt, 120 max tokens) | Time | Result |
|---|---|---|
| default (reasoning on) | 3.11 s | **`content: null`**, 523 chars of reasoning, `finish_reason: length` (the reasoning-model trap) |
| `chat_template_kwargs: {"thinking": false}` | 1.12 s | 37 tokens, real reply |
| `chat_template_kwargs: {"enable_thinking": false}` | 0.98 s | 29 tokens, real reply |
| `reasoning_effort: "low"` | 2.94 s | still thinks, truncated |

So the provider **must disable thinking per request** (try both kwarg spellings; verify the reply has non-empty `content`, never read `content` blindly) and treat an empty `content` as failure. About 1 s for a full short reply is on par with the in-process Qwen3-4B (0.7-1.0 s), from a much larger model, and it frees about 7.6 GB of VRAM on the 4070.

Design:
- New `content/spark_chat.py`: `SparkChatProvider(ContentProvider)`, httpx (already a dependency), **loopback-only**: reject any base URL whose host is not `127.0.0.1`, `::1` or `localhost` at config validation and again at call time (same doctrine as `spark_models`). Model id from config, default `auto` = first id from `GET /v1/models` (cached, refreshed on 404). Stream the response (SSE) so the first sentence reaches TTS before the reply finishes; `asr->llm` becomes time-to-first-sentence.
- Config: `listen.llm.provider: spark | local` (default `spark`), `listen.llm.base_url` (default `http://127.0.0.1:8000/v1`), `listen.llm.model` (default `auto`), `listen.llm.timeout_s`, `listen.llm.fallback: local | none`. The earlier removal of `listen.llm` in round 3 is reversed in this narrower form.
- Failure handling: tunnel down or timeout -> if `fallback: local`, lazy-load the in-process Qwen3-4B (costs a cold load, so say so in the log) else drop the reply and say nothing (same rule as RVC failure). The bot must not start a tunnel itself.
- The N1 local-only scan gets one named exception: `content/spark_chat.py` may import httpx, and only with the loopback guard present (test asserts it); no other listen module may.
- The system prompt is the chat prompt (`CHAT_SYSTEM_PROMPT`); a custom persona must not leak the radio prompt into voice chat.
- Not a dependency of the bot's other features: music commentary keeps its in-process path unless changed separately.

Open items: verify behavior under concurrent load from the other Spark users; confirm the two kwarg spellings across a ring model change; measure time to first sentence with streaming; decide whether to keep the in-process Qwen3-4B resident when `provider: spark` (frees VRAM only if it is not loaded).

## 18. P10c results (2026-10-25, orchestrator)

**Headline: listen replies now come from the Spark, the in-process LLM stays unloaded, and every budget line passes except the TTS stage target.** Over 30 turns: total P50 4.46 s, P90 7.17 s, backchannel 0/30 over 0.8 s, Spark used 30/30 with 0 fallbacks, resident VRAM down 7.7 GB. Two things are not fixed: total P90 is worse than P10b (6.40 -> 7.17 s, still inside 9 s), and TTS over-generation is bounded but not gone. The by-ear check (P10c item 1) was never run.

Landed (uncommitted):

| Task | Files | Tests |
|---|---|---|
| S1 Spark provider + fallback + config + app | `content/spark_chat.py` (`SparkChatProvider`, SSE stream, thinking off under both spellings, `EmptyModelText`, loopback guard in the config validator, the constructor and every call, model `auto` from `/models` with one re-resolve on 404), `content/fallback.py` (`FallbackChatProvider`, `LazyLocalProvider`, `ensure_spark_preferred`, a 30 s skip window after a Spark failure), `config.py` (`ListenLlmSettings`, `assert_loopback_url`, reply caps 8/12, TTS cap settings), example yaml, `app.py` (`listen_content_provider`, eager load skipped when `provider: spark`) | `test_spark_chat.py`, `test_content_fallback.py`, plus additions to `test_settings.py`, `test_app_lifecycle.py` and `test_listen_local_only.py`; live smoke passed |
| S2 streaming turn manager | `listen/conversation.py` (`StreamChunker`, `extend_bot`), `listen/turn_taking.py` (LLM task -> chunk queue -> synth task -> player; early backchannel at end of speech, before ASR) | `test_listen_conversation.py`, `test_listen_turn_taking.py`, `_listen_turn_fakes.py` |
| S3 TTS length cap | `domain/speech.py` (`TtsRequest.max_audio_seconds`), `speech/tts/qwen_custom_voice.py` (`max_new_tokens` bound), `listen/reply.py` (trim at `max(2.0, words x 1.0)` s with an 80 ms fade, `truncated` flag, WARNING) | `test_tts_length_cap.py` |
| S4 bench n=30 on the Spark | `tools/listen_pipeline_bench.py`, `evidence/listen_turn_latency_n30_spark.json` | n/a |

**Review.** Pinned values were checked against the code: chunk timeline synth starts at +2.0/3.5/4.5 s and plays at +3.0/6.0/9.0 s; request body `chat_template_kwargs == {"thinking": False, "enable_thinking": False}`; cap 8 words gives 8.0 s and `truncated`; non-loopback `10.0.0.5` is refused with 0 requests reaching the transport. No forbidden file was touched: `conftest.py` (17:49) and `README.md` (Oct 2) predate the round, and `test_no_discord_imports.py` (18:35) was edited only by the orchestrator (below). Read-through of the untracked builder tests: every P7 acceptance test and every round-3/P10b tripwire is still present with unchanged expectations. The two disclosed test-setup changes are legitimate: `_timed_rig` pins caps 10/18 because its text expectations are 10/18, and `utterance()` now defaults to 2.0 s of audio so the 1.2 s early-clip rule does not skip the three older backchannel tests. **User audio on disk: none.** Bench input is synthetic and in memory. But a cancelled synthesis leaves **bot** TTS output behind: `var/speech/f971788cda44/final.wav` (20:28) and `var/speech/75c6875e91e9/final.wav` (20:32), written during the bench run that had 2 cancelled syntheses. `ReplySynthesizer` removes only job directories that carry the FAILED marker. Invariant 2 holds; "a dropped reply leaves no files" does not hold on cancel.

**Consent: holds.** All four checks are intact under streaming: pre-ASR (`on_utterance`), post-ASR (`_accept`), pre-LLM on every merged line (`_run_turn`), and barge-in at onset and again after the sustain timer. The early clip starts only after the `_allowed` check. It is also skipped for the bot id, while SPEAKING, inside the cooldown, over the rate limit, and in wake-word mode outside the follow-up window. Because the clip now plays before ASR, a consenting user's cough or "Thank you." gets a clip and then no reply (a test pins this). That is a UX cost, not a consent issue.

**Local-only: holds.** `httpx` is imported only by `content/spark_chat.py`, and the N1 scan asserts it is the sole importer and calls the guard. The guard runs at config validation, in the constructor and before every call (`stream`, `health`); `follow_redirects=False`. **Orchestrator fix:** the client also had `trust_env` on, so an `HTTP_PROXY` / `ALL_PROXY` variable would have carried loopback requests to the proxy host (httpx does not exempt 127.0.0.1). It is now `trust_env=False`, and a test fails without it. `grep -rn 'deepseek|GLM-5|EXL3' src tests tools config` returns 0 hits. The resolved id appears only in the evidence JSON. No other host strings appear in the listen, Spark or config code. Nothing reads `.env` or writes tokens: the bench uses `SecretsSettings(_env_file=None)`. Note: the pytest `addopts` in `pyproject.toml` (`-ra`) do not deselect `integration`, so the default suite makes one live loopback call to the Spark (`test_live_spark_streams_a_goku_reply`). It is skipped if the tunnel is down.

**Mutation tripwires** (each mutation applied to a scratchpad copy, reverted at once, and checked identical with `cmp`):

| Mutation | Result |
|---|---|
| (a) drop `chat_template_kwargs` | KILLED: `test_request_body_shape_and_model_from_models_endpoint` |
| (a2) `enable_thinking: True` | KILLED by the request-shape test only. The live Spark test failed in the first run (exception truncated in the output) but **passed** when rerun alone, so `thinking: False` by itself is enough for `deepseek-v4.1-flash`. The first live failure was not reproduced; its cause is unknown |
| (b) call-time guard returns the raw URL | KILLED: `test_non_loopback_url_rejected_by_config_constructor_and_call[10.0.0.5, example.com]` (mutated-settings `generate`/`health`) |
| (c) config validator returns the value unchecked (only the validator; `assert_loopback_url` intact) | KILLED: `test_listen_llm_rejects_non_loopback_hosts`, `test_listen_settings_llm_is_loopback_only...`, `test_non_loopback_url_rejected...` x2 |
| (d) empty stream returns instead of raising `EmptyModelText` | KILLED: `test_empty_content_stream_raises_empty_model_text` x4 |
| (e) early clip started before the consent check | KILLED: `test_no_early_backchannel_for_a_user_who_is_not_allowed` |
| (f1) StreamChunker without the interjection merge | KILLED: `test_stream_chunker_sentence_is_final...`, `test_stream_chunker_matches_batch_on_tricky_texts` x8 |
| (f2) interjection appended after the sentence (reorder) | KILLED: the same 9 tests |
| (f3) open-sentence cut allowed on the still-growing last token | KILLED: `test_stream_chunker_emits_long_sentence_chunk_before_the_text_ends` |
| (g1) post-generation trim removed | KILLED: `test_over_generation_is_trimmed_and_warned` |
| (g2) `max_new_tokens` not forwarded to the model | KILLED: `test_qwen_passes_max_new_tokens[5.0-...]` |
| (g3) `max_audio_seconds` not set on the request | KILLED: `test_over_generation_is_trimmed_and_warned`, `test_within_cap_untouched` |
| (h1) local provider built eagerly | KILLED: `test_healthy_spark_never_touches_local`, `test_non_chat_kind_never_falls_back`, `test_listen_spark_builds_fallback_provider_without_loading_llm` |
| (h2) local model also called when the Spark answered | KILLED: `test_healthy_spark_never_touches_local` |
| (h3) `load_content_model` loses the `listen_uses_spark` skip | KILLED: `test_listen_spark_builds_fallback_provider_without_loading_llm` |
| new: `trust_env=False` removed | KILLED: `test_client_ignores_environment_proxies` (new) |
| new: fallback stream without `aclosing` | KILLED: `test_early_stop_closes_the_spark_stream_at_once` (new) |
| round-3 re-run: pre-ASR check off | KILLED: `test_never_opted_in_user_is_not_transcribed`, `test_bot_id_dropped_even_if_consent_state_lacks_it`, `test_no_early_backchannel_for_a_user_who_is_not_allowed` |
| round-3 re-run: post-ASR `_accept` check off | KILLED: `test_post_asr_recheck_drops_revoked_user_before_logging_or_queueing` |
| round-3 re-run: pre-LLM merge filter off | KILLED: `test_revoked_during_other_users_asr_is_filtered_from_merge` |
| round-3 re-run: onset barge-in consent check off | KILLED: `test_unknown_or_bot_speech_never_barges_in[50]` and `[999]` |

**Bench honesty: production wiring, Spark really used, but the bench's recording had a defect that corrupted two summary rows.** Good: `Application.load_content_model()` then `app.listen_content_provider` (a `FallbackChatProvider` over `SparkChatProvider`, defaults, model `auto` resolved to `deepseek-v4.1-flash`). The production `TurnManager` streams it (the recording wrapper exposes `stream`). Per-turn Spark first delta is recorded (P50 0.24, P90 0.332, max 0.987 s). The in-process LLM was loaded neither at startup nor after 30 turns. Warm-ups ran (RVC, TTS x2, ASR, one Spark reply). 30/30 turns produced metrics. The not-opted-in turn (313 packets) gave 0 decodes, 0 buffered samples, 0 ASR, 0 LLM, 0 metrics and 0 plays. The fallback drill is real: base URL `127.0.0.1:9`, then `ConnectError`, 1 WARNING, local loaded on call 1 (1.435 s including a 0.861 s load from the page cache), then 0.378 / 0.598 s. `ensure_spark_preferred` with the real Spark unloaded it in 1.356 s, from 13059 to 5395 MiB. One difference from production: `app.py` does not pass `listen.reply.tts_*` to `ReplySynthesizer`, so the bench passes them itself.

Defect: `RecordingPlayer.ends` and `RecordingSynth.results` were appended only on completion. One cancelled play or synthesis (turns 8 and 16) shifted both lists for the rest of the run. From turn 8 on, `end_at_eos_plus_s` is off by 4-24 s, and the `tts_results` labels are off by one or two (for example, "Man, that was intense!" is listed as 11 words). As a result, the JSON's **dead-air row (P50 0 / P90 0 / max 4.463, n=72) is wrong**, and its **truncation row (6/126, with texts)** is mislabelled. Both rows below were recomputed from `plays` (start + `audio_s`), which the defect does not touch. The orchestrator fixed the tool (both lists now append in `finally`; the backchannel anchor skips a dropped fragment's clip; a stale "LLM returns the whole reply" sentence was removed from the note). It was **not rerun** (no GPU this round), and the evidence JSON is left as produced.

| Line (section 4) | Target | P10b n=30 (in-process Qwen3-4B) | P10c n=30 (Spark) | Verdict |
|---|---|---|---|---|
| Total P50 | <= 6 s | 4.75 | 4.46 | PASS (better) |
| Total P90 | <= 9 s | 6.40 | 7.17 | PASS (**worse**) |
| Total max | none | 11.54 (1/30 over 9 s) | 8.24 (0/30) | better |
| `eos->asr` P50 / P90 | n/a | 0.79 / 0.82 | 0.78 / 0.80 (max 1.32) | same |
| Time to first chunk (`asr->llm`) P50 / P90 | 1-2 s | 0.70 / 0.97 (whole reply) | 0.47 / 0.76 (max 1.15) | PASS (streaming) |
| `llm->tts1` P50 / P90 / max | 2.5-3.5 s | 3.27 / 4.92 / 8.80 | 3.02 / 5.77 / 6.45 | P50 PASS, **P90 FAIL** |
| `tts1->audio` | 0 | 0 | 0 (all 30) | PASS |
| Backchannel P50 / P90 / max | <= 0.8 s | 0.793 / 0.823 / 1.604, 10/30 over | 0.619 / 0.623 / 0.626, **0/30 over** | PASS |
| Dead air between chunks P50 / P90 / max | ~0 | 0.0 / 2.715 / 8.703 (n=43) | **recomputed** 0.003 / 0.92 / 4.82 (n=94; 17 gaps > 0.3 s, 8 > 1 s) | IMPROVED, not met |
| Chunks hit by the TTS cap | 0 | no cap (7/73 under 1.3 w/s) | **recomputed** 7 / 124 played, plus 10 more at 1.01-1.28 w/s just under the cap | bounded, not eliminated |
| VRAM after load (device) | <= 14.5 GB | 11802 (baseline 1676) | 4567 (baseline 2170); bot resident 10126 -> 2397 MiB | PASS (-7729 MiB) |
| VRAM peak during turns | <= 16.4 GB with ~2 GB client | 13942 | 6047 (13059 only in the fallback drill, local loaded) | PASS |
| Turns that fell back to local | 0 when the Spark is up | n/a | 0 / 30 | PASS |
| Emoji / markdown to TTS; sentences <= 2 | 0; yes | 0; yes | 0; yes | PASS |

Why P90 got worse: `llm->tts1 = 0.48 + 0.707 x first-chunk audio s` (r = 0.75). The P90 tail is cap-hit first chunks: turns 3 and 4 ran 8 words to the full 8.0 s cap, giving 6.25 and 6.22 s. Turns 5 and 23 (6.45 and 5.77 s) had only 3.6-3.8 s of audio but are slow at about 1.05 w/s, which looks like generation running to the token bound with the tail trimmed off. Median first chunk: 7 words, 3.9 s of audio.

**Other findings:**
1. **Two clips on a split utterance.** The Master Roshi clip has an internal pause and endpointed twice on turns 8, 16 and 24. Each fragment got its own clip (at -3.6 s and +0.6 s), and on turns 8 and 16 the fragment's turn was cancelled after one synthesis (a wasted Spark call). The claim (`_bc_claimed`) is released when the fragment's turn ends. Real speech pauses more than TTS does, so this will be more common live.
2. **Over-generation remains**: 17 of 124 chunks are at or near the cap at 1.0 s per word, while natural Goku speech is about 1.6-2.7 words/s (0.37-0.63 s per word). It is still unconfirmed by ear whether these chunks contain silence or babble.
3. **Orchestrator fix:** `CODEC_TOKENS_PER_SECOND` was 12, but the codec runs at 12.5 Hz (`speech_tokenizer/config.json`: `_frame_rate` 12.5 = 24000/1920). The bound was safe only for caps up to 24 s. It is now 12.5 with 13 tokens of slack; tests re-pinned (5 s -> 76, 2 s -> 38, 24 s -> 313 = 25.04 s).
4. **Orchestrator fix:** `FallbackChatProvider.stream` did not `aclose` the Spark generator when the turn manager stopped early, so the HTTP response closed only on garbage collection. It now uses `aclosing`.
5. Open wiring for P9: `ensure_spark_preferred()` at listen start; `listen.reply.tts_*` into `ReplySynthesizer`; in fallback mode the local model uses `content.max_new_tokens` (120), not `listen.reply.max_new_tokens` (60); remove the job directory on a cancelled synthesis; `tools/listen_pipeline_bench.py` no longer exercises `_chunk_plan`, which is still kept for it.

**Remaining levers, ranked by saving computed from the n=30 rows (none implemented):**

| # | Lever | Expected saving |
|---|---|---|
| 1 | Tighten the TTS cap from 1.0 to about 0.6 s per word, floor 2.0 -> 1.5 s, **after** the by-ear check confirms the slow chunks are padding or babble | turns 3/4 first chunk 8.0 -> 4.8 s of audio = about -2.3 s `llm->tts1` each (slope 0.707); `llm->tts1` P90 5.77 -> about 4.0-4.4, total P90 7.17 -> about 5.8-6.1 (estimate); also shortens the 17 slow chunks, so fewer dead-air gaps |
| 2 | First chunk 8 -> 6 words | median 7 words / 3.9 s -> about -1.1 s of audio = about -0.8 s `llm->tts1` P50; total P50 4.46 -> about 3.7 |
| 3 | Later chunks: cap 12 -> 10, or limit chunk k+1 to about 1.3 x chunk k's audio | a gap appears when synthesizing k+1 (0.48 + 0.707 x audio) takes longer than k plays; dead-air P90 0.92 -> about 0.3 (estimate) |
| 4 | Hold the clip claim for a few seconds after a dropped or cancelled fragment | removes the second clip on 3/30 turns; no latency effect |
| 5 | Endpoint silence 600 -> 400 ms | -0.2 s on every turn and on the clip (0.62 -> 0.42); **more split utterances** (3/30 already at 600 ms), so do lever 4 first |
| 6 | `max_new_tokens` 60 -> 40 | about 0 now: streaming already ends `asr->llm` at the first chunk. Drop |

**Suite:** `.venv-main/bin/python -m pytest tests/unit tests/integration/test_listen_wav_harness.py -q` = **600 passed** in 52.2 s (488 before the round; includes the live loopback Spark test). `ruff check src tests` = the same 3 pre-existing errors (`test_failure_fallback.py` I001, `test_hubert_loader_parity.py` 2x E501). `ruff format --check src tests` flags only `test_hubert_loader_parity.py`. Every file touched this round is clean, including `tools/listen_pipeline_bench.py`.

Orchestrator edits this round: `test_no_discord_imports.py` (registered `content.spark_chat` and `content.fallback`); `content/spark_chat.py` (`trust_env=False`) + `test_spark_chat.py` (proxy test); `content/fallback.py` (`aclosing`) + `test_content_fallback.py` (early-stop test); `speech/tts/qwen_custom_voice.py` (12.5 Hz) + `test_tts_length_cap.py` (re-pinned); `tools/listen_pipeline_bench.py` (recording alignment, backchannel anchor, note); this section.

| ID | Status | Note |
|---|---|---|
| P0a | **BLOCKED** (owner) | live receive + DAVE with two humans; gates P9 |
| P0b, P0c, P1-P3, P4a, P5-P8, V1, N1 | DONE | rounds 1-3 |
| P4b | CANCELLED | section 13 |
| P5b | SUPERSEDED | by `content/spark_chat.py` (section 17) |
| P10, P10b | DONE, caveats | sections 15-16 |
| P10c | DONE, caveats | Spark + fallback + streaming + early clip + TTS cap landed, all tripwires killed; by-ear check not done; dead-air and truncation rows recomputed (bench defect, fixed, not rerun); `llm->tts1` P90 FAIL, total P90 worse but in budget |
| P10d | READY | (1) by-ear check: keep the bot audio of chunks under 1.3 w/s (bot audio only, scratchpad, delete after); (2) cap 0.6 s/word if confirmed, first chunk 6 words, claim hold; (3) clean up cancelled synthesis output; rerun n=30 with the fixed bench |
| P9 | **BLOCKED** | on P0a. Also: `ensure_spark_preferred()` at listen start; `tts_*` into `ReplySynthesizer`; `enforce_offline_models()` before any load; fallback `max_new_tokens`; member leave calls `consent.opt_out`; no silent static-provider fallback for listen |
| P11 | BLOCKED | after P9 + P10d, GPU + LIVE |
| P12 | BLOCKED | after P11; README and section 13 still describe the round-3 `listen.llm` removal |

## 19. P9 results (2026-10-25, orchestrator)

**What landed (not committed).** G1: `discord_bot/listen_glue.py` (`dave_decrypt`, `ListenDaveSink`, `VoicePlayer`, `ListenSession`, `ListenSessionRegistry`), `test_listen_glue.py`, `test_listen_dave_sink.py`. G2: `!glisten start|on|off|stop|status` in `commands.py`, music commands refuse while a session is active, `bot.py` registry + `on_voice_state_update` + `close()` stops sessions, `app.py` fallback provider capped at `listen.reply.max_new_tokens` and `build_reply_synthesizer`, `test_listen_commands.py`, additions to `test_app_lifecycle.py` and `test_command_registration.py`.

**P0a finding (DAVE).** discord-ext-voice-recv 0.5.2a179 has no DAVE support. With `wants_opus()` the frames are transport-decrypted but still DAVE-encrypted, so the sink runs `davey` `DaveSession.decrypt` itself. davey raises `ValueError("UnencryptedWhenPassthroughDisabled")` for 3-9% of frames (arrived unencrypted): those are used as plain opus. Any other decrypt error drops the frame. The `None`/bot self-filter runs BEFORE the decrypt.

**Interface agreement G1/G2: OK.** Names, sync/async and argument order match (`make_listen_session(guild, channel)`, `registry.add/get/remove/active`, `session.start/stop/opt_in/opt_out/on_member_left/status/channel_id`). `handle_voice_state_update` (G1) is unused; the bot's own hook does that job. No fixes needed.

**Tests.** `pytest tests/unit tests/integration/test_listen_wav_harness.py` = **665 passed** (67 s; includes one live loopback Spark call). `ruff check` and `ruff format --check` clean on every touched file. The orchestrator added `test_stop_clears_a_populated_conversation`.

**Mutation table** (suites run: glue, dave sink, commands, app lifecycle):

| Mutation | Result |
|---|---|
| (a) sink consent check AFTER decrypt | KILLED |
| (b) `UnencryptedWhenPassthroughDisabled` = drop | KILLED |
| (c) other decrypt errors = raw passthrough | KILLED |
| (d) member leave only `ingress.remove_user` (glue and bot.py) | KILLED (both) |
| (e) `start` allowed with empty `allowed_channel_ids` | KILLED |
| (f) `!gplay` / `!gjoin` allowed while listening | KILLED (both) |
| (g) operator check removed (start, stop) | KILLED (both) |
| (h) `ensure_spark_preferred` not called | KILLED |
| (i) `enforce_offline_models` not called / called after ASR load | KILLED (both) |
| (j) reply persisted to disk | covered by `test_full_fake_turn_replies_and_writes_nothing` (var/ tree identical after a full turn); not mutation-run |
| extra: music-active check ignored | KILLED |
| extra: `stop()` does not clear consent | KILLED |
| extra: `stop()` does not clear the conversation | SURVIVED, equivalent: `TurnManager.shutdown()` clears it too, so the glue clear is defence in depth. Test added anyway |

**Greps.** `httpx` is imported only by `content/spark_chat.py` in listen code (`content/news.py` is old music code). No `requests`/`aiohttp`/`socket`/`websockets` in `src`. The only disk write in listen code is `listen/reply.py` backchannel clips (bot audio, `var/listen/backchannel`). No user audio or transcript is written; `log_transcripts` defaults false and gates a DEBUG line only. No token or secret is logged.

**Open issues.**
1. The "listening" text notice posts via `channel.send` on the voice channel's own text chat; `ctx.channel` is not passed. Decide if that is acceptable.
2. `!gstop` and `!gskip` have no listen guard; `!gstop` disconnects, the voice-state hook then stops the session.
3. `start` only refuses on `guild.voice_client`; `!glisten start` also checks queue and player task.
4. Fallback provider loads a second local Qwen only if music also runs local while the Spark is down.
5. Endpointer and Silero run inline on the bot loop, so a slow poll delays the Discord heartbeat. Watch in P11.
6. Cancelled-synthesis job-directory leak (P10c finding 5) is still open.
7. One harmless unawaited-coroutine `RuntimeWarning` in the glue tests.
8. Not run against a real `VoiceRecvClient`.
9. `tests/integration/test_gpu_arbitration.py` failed for G2 when run (loads a model on the GPU); not investigated.

**Owner config for a live run** (set in the real config, not edited here): `listen.enabled: true`, `listen.allowed_channel_ids: [<test voice channel id>]`, `listen.operator_user_ids: [<owner Discord user id>]` (empty falls back to guild administrators). Both lists empty means refusal.

**Live checklist for P11 (GPU + LIVE, 'ppc' channel, owner plus one more human).** Items 1-10 of section 8 apply; adapted:
1. From a channel other than `ppc`, `!glisten start` is refused ("not allowed"). A non-operator in `ppc` is refused.
2. In `ppc`: announcement audible within 2 s, text notice posted.
3. Human B (who never types a listen command) talks 20 s: B is transcribed and replied to like anyone else, `dave_dropped` low.
4. A talks: backchannel within 1 s, reply within 9 s, metrics line present.
5. A interrupts mid-reply: playback stops within 500 ms, listening continues (the `stop_playing` fix), next reply references the interruption.
6. A and B overlapping: two transcripts, one merged reply.
7. A stops mid-sentence or leaves the channel: the half-finished utterance is not replied to.
8. `!gplay` and `!gjoin` during listen: refused with the "!glisten stop first" message.
9. `!glisten stop`: bot disconnects, `find var -newer <marker> -name '*.wav'` shows only bot clips.
10. Everyone leaves: session stops by itself. Ctrl+C during a reply: clean exit, RVC worker reaped.
11. Check the DAVE drop rate over 60 s (expect 3-9% passthrough, near 0% dropped).

### Status (after P9)

| ID | Status | Note |
|---|---|---|
| P0a | DONE | DAVE finding: voice-recv has no DAVE; sink decrypts with `davey`; passthrough error = plain opus |
| P9 | **DONE (code), caveats** | 665 tests, mutations killed; not run against live Discord. Open issues above |
| P10d | READY | unchanged |
| P11 | READY for owner | needs the two config values above, GPU and a second human |
| P12 | BLOCKED | after P11 |


## 20. Speed build (2026-10-25, orchestrator)

**Why.** Live test: 9.4 s median from end of speech to first reply audio. Owner kept the Spark (loopback) as the LLM with thinking low and a tuned prompt, whole-sentence replies, no RVC (`goku_raw`), and asked for much faster.

**What landed (not committed).**
- B1 `speech/tts/qwen_custom_voice.py`: faster-qwen3-tts 0.3.1 (CUDA graphs) behind `speech.tts_fast` (default true, falls back to stock Qwen3-TTS on any import/load/capture failure) and `speech.tts_stream_chunk_size` (8). New `stream_pcm(request)`: mono float32 24 kHz chunks, GPU lease held for the whole generation (`InferenceCoordinator.slot()`, private lock without one), worker thread with 4-chunk backpressure, stop at chunk boundary, lease released only after the worker ends, cap = `max_audio_seconds`.
- B2 `listen/reply.py`, `listen/turn_taking.py`, `discord_bot/listen_glue.py`: streaming reply path chosen per turn by `ReplySynthesizer.can_stream` (passthrough voice, speed 1.0, 48k stereo, TTS supports streaming). Stateful 2x linear upsampler, fixed gain, same cap with a position-based 80 ms fade, one generation at a time, one `VoicePlayer.play_stream` per reply (120 ms prebuffer, silence on underrun), barge-in closes the generator, `stop_playing()` not `stop()`.
- B3 `content/prompts.py` (one conversational sentence of at most 18 words, think briefly), `speech/asr/parakeet_onnx.py` (CUDA fp32, torch imported before the ONNX session, self-check, CPU int8 fallback; hard-link dir because onnxruntime rejects symlinked external data).
- B4 `tools/listen_pipeline_bench.py`, `evidence/listen_turn_latency_n20_fast.json`.
- Review fixes (this section): `app.py` now passes `fast`, `stream_chunk_size` and the shared `coordinator` to both `from_settings` sites (streaming was on a private lock); stream gain re-measured live and moved (below), with a soft knee instead of a hard clip; `spark_chat` retries ONCE with thinking off when thinking low returns no text (B4 turn 13 `EmptyModelText` had dropped to the in-process model: 6.7 s cold load, +8 GB VRAM); `test_listen_conversation.py:85` updated to the new prompt wording; `DEPENDENCY_NOTES.md` and a `pyproject.toml` comment record the manual installs.

**Gain check (real fast TTS, flock).** 8 sentences, streamed PCM vs the postprocessed WAV (peak-normalised to -1 dBFS), same text, run twice (sampling differs per run):

| Gain | Median RMS vs WAV | RMS range | Median peak vs WAV |
|---|---|---|---|
| +3 dB (B2's constant) | -3.7 dB | -6.8 .. +0.8 | -1.5 dB |
| +5 dB | -0.6 dB | -2.4 .. +4.2 | 0.0 dB |
| **+4 dB (shipped)** | about -2 dB (interpolated) | | |

+3 dB was over the 3 dB threshold, so the constant is now `STREAM_GAIN_DB = 4.0` with a tanh knee above 0.8 (linear below, never past full scale). A fixed gain cannot match per-sentence normalisation; +-4 dB spread per sentence is inherent. Re-measure by ear.

**Bench (end of speech -> first reply audio, seconds, P50 / P90 / max).**

| Run | n | eos->asr | asr->llm (first sentence) | llm->first audio | Total |
|---|---|---|---|---|---|
| Live test (RVC, in-process LLM era) | - | - | - | - | 9.4 median |
| n=30 Spark (before this build) | 30 | 0.78 / 0.80 / 1.32 | 0.47 / 0.76 / 1.15 | 3.02 / 5.77 / 6.45 | 4.46 / 7.17 / 8.24 |
| n=20 fast (this build, all) | 20 | 0.46 / 0.47 / 0.49 | 1.47 / 4.31 / 6.72 | 0.20 / 0.20 / 0.21 | **2.14 / 4.95 / 7.38** |
| n=20 fast, Spark-only 17 | 17 | 0.46 / 0.49 / 0.49 | 1.57 / 4.31 / 5.65 | 0.20 / 0.20 / 0.21 | 2.21 / 4.95 / 6.30 |

TTS first audio is flat at 0.2 s (was 3.0 s). ASR is faster (400 ms end silence plus CUDA fp32). The LLM is now the bottleneck: asr->llm regressed from 0.47 to 1.47 s P50 (thinking low, and the stage now waits for a whole sentence). Stream quality: 0 underruns in 22 streams, chunk gaps P50 0.175 s, max 0.343 s. Turns 13-15 used the in-process fallback (the retry above should stop that); turns 5 and 10 ran 17-19 s for 16 words (cap 1.6 s/word did not clip). VRAM steady about 8.5 GB, peak 15.9 GB only while the fallback LLM was loaded. The n=20 run predates the gain change and the Spark retry.

**Mutation table** (suites: streaming, qwen fast tts, asr, chat prompt, glue, conversation, turn taking):

| Mutation | Result |
|---|---|
| (a) streaming used when voice conversion is not passthrough | KILLED |
| (b) barge-in does not cancel synthesis / TTS generator not closed (two variants) | KILLED (both) |
| (c) sentence k+1 generation starts before k ended | KILLED |
| (d) TTS cap not applied on the stream path | KILLED |
| (e) `VoicePlayer.stop` uses `stop()` | KILLED |
| (f) consent filter removed before the turn | KILLED by `test_revoked_during_other_users_asr_is_filtered_from_merge`; survived the streaming suites alone, so `test_consent_revoked_before_turn_never_streams_synthesis` was added (KILLED) |
| (g) fast-TTS failure not falling back to stock | KILLED |
| (h) Parakeet cuda + int8 not switched to fp32 | KILLED |
| (i) torch not imported before the ONNX session | KILLED |
| (j) prompt one-sentence line removed (both prompts) | KILLED (both) |

**Tests.** `pytest tests/unit tests/integration/test_listen_wav_harness.py` = **723 passed** (61 s). ruff check and format clean on every touched file (three pre-existing lint errors remain in untouched `test_failure_fallback.py`, `test_hubert_loader_parity.py`).

**Greps.** Network library: only `content/spark_chat.py` in listen code (`news.py` is old music code; `config.py` only parses URLs). `faster_qwen3_tts` imported only in `speech/tts`. Disk writes in listen code: backchannel clips only (bot audio). No secret logged.

**Owner config to apply** (`config/app.yaml`, gitignored, not edited here):

```yaml
speech:
  tts_fast: true                   # default; false = stock Qwen3-TTS
  tts_stream_chunk_size: 8         # default
listen:
  asr:
    device: cuda
    quantization: fp32             # int8 on cuda is overridden to fp32 with a warning
  vad:
    end_silence_ms: 400            # was 600; must be >= min_speech_ms
  voice_profile: goku_raw          # already set: stream path needs passthrough, speed 1.0
  llm:
    thinking: low                  # already set
  reply:
    backchannel_enabled: false     # already set
```

**Open issues.**
1. LLM is the latency floor: Spark first delta P90 3 s with thinking low. Options: thinking off (about 1 s), or keep low and accept.
2. TTS can ramble (16 words -> 17-19 s). The 1.6 s/word cap is too loose for streaming; consider `tts_max_seconds_per_word: 1.2` (the old 1.0 clipped real speech).
3. Streaming bypasses `synthesize`'s pipeline lease; it takes the coordinator slot itself (wired now). With the ASR on CUDA outside the coordinator, ASR can overlap TTS on the GPU (small, 0.02 s).
4. Stop requests land at a chunk boundary (about 0.2 s). `_playing_sentence` can lead what is heard by up to 0.5 s.
5. CUDA ASR first load not cold-measured; hard-link dir adds directory entries (about 2.4 GB apparent) under the HF cache.
6. `VoicePlayer.play_stream` only tested against a fake voice client; real `VoiceRecvClient` untested.
7. `turn_taking.py` module docstring does not describe streaming yet.
8. After rebuilding the venv: reinstall `onnxruntime-gpu==1.31.0` (replacing `onnxruntime`) and `faster-qwen3-tts==0.3.1 --no-deps` (DEPENDENCY_NOTES.md).

**What to listen to (P11 live).** Reply start feels about 2 s after you stop talking (not 9). Sentence joins sound continuous (one stream, no gap) and inflection is whole-sentence. Loudness is steady across replies (fixed gain; flag any reply much quieter or crunchy). Barge-in cuts audio within about half a second and the bot keeps listening. No clicks at chunk joins (linear 2x upsampler). A long ramble still gets cut with a short fade. Voice is stock Ryan (`goku_raw`), the same as the fast-vs-stock A/B you already judged equal.

### Status (after speed build)

| ID | Status | Note |
|---|---|---|
| B1-B4 | DONE (code), reviewed | 723 tests, mutations killed, gain re-measured |
| P10d | READY | unchanged |
| P11 | READY for owner | apply the config lines above; GPU and a second human |
| P12 | BLOCKED | after P11 |

## 21. Latency round 2 (2026-10-25, orchestrator)

**Live data (80 turns, speed build).** Total P50 2.93 s, P90 7.15 s. Spark first whole sentence (asr->llm) P50 2.03 s, P90 6.4 s, max 10.8 s: 10 of 80 turns at 6 s or more, 14 between 3 and 6 s. Thinking low (`reasoning_effort: low`, max_tokens 70+300) is on. Earlier measurement: thinking off gives a first sentence in about 0.6 s; thinking low about 1.5-2.5 s with outliers to 4.7 s. eos->asr P50 0.76 s although `end_silence_ms` is 400 and CUDA ASR takes 0.02-0.07 s.

**What landed (not committed).**
- H1 hedged Spark request (`content/spark_chat.py` `_stream_hedged`, `config.py` `ListenLlmSettings.hedge_after_s`, default `None` = off; `0` = off). With `thinking: low` and `hedge_after_s > 0`, if the primary (thinking low) has produced no CONTENT delta after `hedge_after_s` (reasoning deltas do not count), one thinking-off request is started. The first request to deliver a non-empty content delta wins; the loser is cancelled, awaited and closed at once. Text from two streams is never mixed. Both go through `_stream_once` (loopback guard, timeouts, cached model id). Primary error before the hedge starts: raised as today. Primary error after: hedge continues. Both fail: primary's error. Both empty: private `EmptyModelText` subclass, so the existing "retry once with thinking off" does not add a third request.
- H2 timing split (`domain/listen.py` `UtteranceEvent.t_emit`, `listen/endpointer.py` stamps it, `listen/turn_taking.py` `_log_asr_timing`). One INFO line per utterance that reaches ASR and returns a transcript. No transcript text.

**Load bound (reviewed).** At most 2 concurrent Spark requests per turn: one primary, one hedge, launched once (a single `if not done` branch, `timeout` set to `None` afterwards). The no-text retry is skipped after a hedge, so no third request. Consumer cancellation (barge-in closes the reply generator) closes both, and a close after a winner closes the winner; tests assert no leftover tasks and both response streams closed. Cost: one extra Spark request on turns slower than `hedge_after_s` (about 10-30% of turns at 2.0 s on the live distribution).

**Mutation table** (suites: `test_spark_hedge`, `test_spark_chat`, `test_listen_asr_timing`, `test_listen_turn_taking`):

| Mutation | Result |
|---|---|
| (a) hedge started even when the primary already produced content | KILLED (suite hangs on endless hedges; a second variant, hedge also when a hedge exists, fails 3 tests) |
| (b) streams spliced (loser's remaining deltas appended) | alone EQUIVALENT (the loser is closed first, so it yields nothing); combined with (c) KILLED by `test_simultaneous_first_deltas_yield_one_stream_and_close_the_other` (added) |
| (c) loser not closed | survived the first suite (cancellation closes in-flight responses); KILLED after the same added test |
| (d) hedge uses thinking low instead of off | KILLED |
| (e) hedge ignores the loopback guard | KILLED (`test_hedge_is_refused_by_the_loopback_guard_when_base_url_turns_non_loopback`, added: the primary is in flight, the base URL turns non-loopback, the hedge makes zero requests) |
| (f) hedge on when `hedge_after_s` is 0/None | `0` variant survived; `test_zero_hedge_after_means_off` added (asserts `_hedging()` False), KILLED |
| (g) timing line logged for a non-consenting user | KILLED |
| (h) timing line contains transcript text | KILLED |

**Tests.** `pytest tests/unit tests/integration/test_listen_wav_harness.py` = **746 passed** (57 s; includes one live loopback Spark call). ruff check and format clean on touched files.

**Owner config** (`config/app.yaml`):

```yaml
listen:
  llm:
    thinking: low
    hedge_after_s: 2.0     # 1.5-2.0; lower = faster tail, more extra Spark requests
```

**Reading `listen asr timing` lines** (`user=<id> eos->emit emit->asr_start asr_wait asr_compute asr_done->accepted audio_s`):

| Field | Meaning | Expected |
|---|---|---|
| eos->emit | last voiced frame to utterance emit | 0.416 s floor (13 x 32 ms silence); excess = VAD/poll/loop lag |
| emit->asr_start | loop hop before ASR starts | under 0.01 s |
| asr_compute | transcriber call as the loop sees it | 0.03-0.1 s |
| asr_wait | asr_compute minus the transcriber's own timer (thread hop, loop lag) | near 0 |
| asr_done->accepted | filters (noise / echo / wake word / cooldown) | near 0 |

By code reading the 0.76 s eos->asr is the 0.416 s silence rule plus 0.05-0.3 s of event-loop lag (inline Silero per 32 ms frame, Discord decode, TTS/LLM tasks on the same loop). Not yet measured live: collect the new lines from the next session.

**Open issues.**
1. Hedge untested against the live Spark (fake transport only). Measure first-sentence P90 with `hedge_after_s: 2.0`.
2. Hedge cuts the tail but the thinking-off reply can be lower quality; check by ear.
3. If `eos->emit` minus 0.416 is large, move the endpointer off the loop (`EndpointerThread` exists, unused in production). Lever: `end_silence_ms` 400 -> 300 saves about 0.1 s, costs mid-sentence cuts.
4. Parakeet `asr_wait` does not separate a default-executor queue wait; Whisper's coordinator wait is not separated either.
5. Full `tests/unit` includes `test_live_spark_streams_a_goku_reply`, a real loopback call.

### Status (after latency round 2)

| ID | Status | Note |
|---|---|---|
| H1 hedge | DONE (code), reviewed | default off; owner enables `hedge_after_s: 2.0` |
| H2 asr timing | DONE (code), reviewed | collect lines live |
| P11 | READY for owner | config lines above |
| P12 | BLOCKED | after P11 |

## 22. Streaming RVC (2026-10-25, orchestrator)

**What landed (opt-in, default OFF; the whole-file path is unchanged, its tests untouched).** Design: `docs/RVC_STREAMING_DESIGN.md`.

| Piece | File |
|---|---|
| Block engine (HuBERT + pm f0 on a context window, drop context, SOLA join, exact 2x samples) | `legacy/rvc_stream.py`, eval CLI `legacy/rvc_stream_eval.py` |
| Worker actions `stream_open` / `stream_push` / `stream_close` (one stream per worker, base64 f32le, 24 kHz in, 48 kHz out) | `legacy/rvc_worker.py` |
| Bot adapter `convert_stream`, `warm_stream`, router passthrough | `speech/voice_conversion/{base,rvc,router}.py` |
| Reply path: `can_stream` accepts rvc when the converter streams; `RVC_STREAM_GAIN_DB` | `listen/reply.py`, `app.py` |
| Config `speech.voice_conversion.streaming` + six `stream_*_ms` | `config.py`, `config/app.example.yaml` |

**Measured (RTX 4070 Ti SUPER shared with the live bot; 3 Qwen sentences, f0 pm, 250 ms chunks; about 25 s GPU in total under the lock).**

| Metric | Result | Bar |
|---|---|---|
| out_ratio | 1.0 on every run | exactly 1.0 |
| First converted audio, warm (compute of the unlocking push) | 22-37 ms (cold first stream 146 ms) | within 250 ms of the first TTS chunk (needs 240 ms of input, so first TTS chunk size dominates) |
| RTF | 0.06-0.11 | at most 0.3 |
| Step | 19-33 ms | at most 100 ms |
| f0 corr vs whole | 0.991-0.9996 | at least 0.98 |
| f0 median cents | 6.5-9.7 typical, one run 19.4 | at most 10 (missed once of 6) |
| log-mel L1 vs whole | 3.7-4.8 dB, whole-vs-whole floor 1.6-1.75 | floor + 2.5 (MISSED on 2 of 3 clips: +2.6 to +3.2) |
| Join metric (max second difference at joins / p95) | 2.23-4.32, whole at same positions 2.42-2.47 | max(2.5, 1.3 x whole) (MISSED in 1 of 6 runs: 4.32; the rest pass) |
| Join metric (max first difference within 1 ms of a join / p99.9) | stream 0.64-1.65, whole 0.80-1.27 | at most 1 for no click (above 1 in 4 of 6 stream files; same order as the whole audio at those positions) |

Plain reading: latency, RTF and pitch hold. Timbre is close but not identical (mel distance 2-3 dB above the model's own noise). Joins are not provably click-free by the numeric metric; the metric is noisy because RVC injects random noise. Tuning probe (2 clips, noisy): `stream_context_ms: 2000` gave a small gain (mel 3.43 vs 3.89 and 4.86 vs 5.45, f0 cents 7.6/9.4) for rtf 0.09; `lookahead 200` and `block 480` showed no gain. Defaults left unchanged; try 2000 by ear. Whole-vs-stream WAVs for listening: see below.

**Gain calibrated.** `RVC_STREAM_GAIN_DB` 3.0 -> -0.8. RVC output is already near full scale; at unity the stream sat a median +0.9 dB above (RMS, range +0.3 .. +1.4) and +0.6 dB (peak, +0.4 .. +0.8) the peak-normalised (-1 dBFS) whole WAV on 7 sentences. The old +3 dB would have driven the tanh knee on nearly every sentence.

**Fixes in review.** (1) `convert_stream` now records the stream id before `stream_open` is sent, so a cancel during the open still aborts the worker-side stream. (2) Gain constant above. (3) Added `test_two_streams_share_no_state`.

**Mutations (tests: engine, worker, adapter, config, reply streaming).**

| Mutation | Result |
|---|---|
| flag default ON in config | KILLED (`test_defaults_off_and_numeric_defaults`) |
| `supports_streaming` ignores `enabled` | KILLED (`test_supports_streaming_gating`) |
| `from_settings` forces enabled | KILLED (`test_from_settings_mapping`) |
| adapter does not abort the worker stream on cancel / early close | KILLED (`test_aclose_mid_stream_aborts_and_releases_lock`) |
| reply path does not close the converter generator (barge-in / cap leaves worker stream open) | KILLED (`test_cap_truncates_fades_and_closes_both`) |
| no crossfade at joins | KILLED (`test_sola_join_continuity_with_phase_breaks`) |
| SOLA search offset forced to 0 | KILLED (same test) |
| stale tail carried into the next stream (module state) | KILLED (`test_chunk_invariance_bit_identical`, `test_two_streams_share_no_state`) |
| worker keeps the stream after close (push after close accepted) | KILLED (`test_happy_path`) |
| worker accepts a push with a wrong stream id | KILLED (`test_replaced_stream`) |
| worker does not forget the stream when dropping | KILLED (`test_happy_path`) |

Survivors: none. Suite: `pytest tests/unit tests/integration/test_listen_wav_harness.py` 876 passed, 1 skipped (GPU engine test, `RVC_STREAM_GPU_TESTS=1`). Three pre-existing ruff findings in other files remain.

**Enable.** In the app YAML, under `speech.voice_conversion`:

```yaml
speech:
  voice_conversion:
    streaming: true
    # optional: stream_context_ms: 2000
```

Restart the bot (the worker loads the new code on restart). `warm_stream` runs at warm-up. f0 is forced to pm for streams.

**Listen to** (`/home/cbash23/.cache/listen-speed/rvc-stream/samples/`): `fast_nonstream_s{0..5}_whole.wav` vs `_cfg0.wav`, plus `base_bf16_sdpa_s0_*`. Check joins at multiples of 320 ms, breathy or low-pitched stretches (pm weakness), and loudness against the WAV path.

**Open issues.**
1. Not run end to end through the real `ReplySynthesizer` + real TTS + real worker (GPU shared with the live bot). First-chunk latency there is TTS chunk size plus about 40 ms.
2. Join metric and mel distance miss the stated bars on some clips; decide by ear.
3. Streams use pm only; the `f0_method_override` config is ignored for streams.
4. One stream per worker; a backchannel convert in flight delays one block (up to about 0.5 s).
5. A stale `git stash@{0}` (WIP on main at 9be572f) exists in the repo; origin unknown.

### Status (after streaming RVC)

| ID | Status | Note |
|---|---|---|
| S-RVC streaming | DONE (code), reviewed | default off; owner enables `streaming: true` and listens |
| P11 | READY for owner | config lines above |
| P12 | BLOCKED | after P11 |


## 23. RVC output quality (2026-10-26, orchestrator)

Owner feedback: converted speech is robotic/metallic, pitch wobbles / is off-key, and does not sound like the target. Plan: `docs/RVC_QUALITY_PLAN.md`. Lab: `tools/rvc_quality_lab.py` (+ `tools/rvc_quality_sources.py`). Work dir: `/home/cbash23/.cache/listen-speed/rvc-quality/` (`index.md` = tables with wav links, `results.json`, `samples/<model>/<variant>/s0..5.wav`). Nothing committed; the bot was not started. The numbers below are PROXIES computed from the renders; nobody has listened yet.

### Diagnosis

1. The faiss index never loaded (relative path vs worker cwd): the model's own timbre retrieval was off, so output kept the Ryan source timbre ("wrong timbre").
2. f0 came from `pm` (octave errors, ~6 jumps/s, track corr 0.74 vs RMVPE 0.87): the wobble / off-key / robotic pitch.
3. No pitch shift: Ryan's median is about 157 Hz, Obama's target 108 Hz, so Obama's register error was about 3.7 semitones (Goku about 3.6).
4. protect, rms_mix, index_nprobe were not reachable.

### What landed (all new settings default to the old behaviour)

- `legacy/rvc_f0.py`: RMVPE (weights vetted, see below), f0 post-processing (octave repair, clamp, median, gap fill), `auto_pitch_shift`.
- Worker/stream/rvc_cli: index loads by absolute path (`use_index`, `index_rate`, `index_nprobe`), `protect`, `rms_mix_rate` (whole path only), `f0_post`, `auto_pitch`, `f0_method=rmvpe`; stream gets `f0_algo=rmvpe`, index, protect, f0_post; `RVC_WORKER_FP32=1`. Streams keep `f0_method` pm-only.
- Bot: `speech.voice_conversion` gains `use_index, index_rate, index_nprobe, protect, rms_mix_rate, pitch_shift_offset, auto_pitch, auto_pitch_strength, auto_pitch_max, target_f0_median_hz, f0_post_median, f0_post_octave_repair, f0_post_clamp, stream_f0_algo`; `f0_method_override` accepts `rmvpe`. `config/app.yaml` is NOT changed.
- Review fixes: lab `c9_fp32` was skipped (it looked for the switch in the wrong file); fixed and run. Added tripwire test for rmvpe-unavailable. `ruff format` on `legacy/rvc_f0.py`.

### Proxy table: Obama (target 108 Hz; 6 clips; whole-path unless named e*)

| variant | octave_err | jump/s | smooth cents | f0 corr | register err st | voiced ratio | hiss | flags |
|---|---|---|---|---|---|---|---|---|
| ref_harvest_idx05 | 0.013 | 3.70 | 104 | 0.861 | 1.12 | 1.17 | 0.001 |  |
| v00_baseline | 0.020 | 5.87 | 138 | 0.737 | 3.67 | 1.06 | 0.001 |  |
| a1_pm_idx05 | 0.024 | 4.71 | 129 | 0.697 | 3.18 | 1.02 | 0.001 |  |
| a2_rmvpe | 0.011 | 3.37 | 104 | 0.869 | 3.88 | 1.12 | 0.002 |  |
| a3_rmvpe_post | 0.010 | 4.42 | 125 | 0.852 | 3.46 | 1.15 | 0.004 |  |
| a4_rmvpe_post_idx05 | 0.011 | 4.31 | 127 | 0.808 | 3.27 | 1.14 | 0.002 |  |
| a5_harvest_idx05 | 0.017 | 3.70 | 111 | 0.867 | 3.51 | 1.15 | 0.002 |  |
| b1_shift_fixed_m3 | 0.009 | 4.04 | 113 | 0.785 | 3.52 | 1.10 | 0.001 |  |
| b2_shift_fixed_m4 | 0.015 | 3.20 | 104 | 0.782 | 3.53 | 1.09 | 0.002 |  |
| b3_shift_fixed_m5 | 0.015 | 2.84 | 99 | 0.866 | 4.22 | 1.03 | 0.001 |  |
| b4_autoshift | 0.010 | 4.20 | 110 | 0.887 | 1.01 | 1.14 | 0.003 |  |
| b5_autoshift_half | 0.009 | 3.44 | 101 | 0.894 | 1.78 | 1.14 | 0.002 |  |
| c1_idx03 | 0.010 | 4.29 | 116 | 0.844 | 1.70 | 1.14 | 0.002 |  |
| c2_idx075 | 0.012 | 3.87 | 113 | 0.840 | 1.64 | 1.13 | 0.002 |  |
| c3_nprobe1 | 0.014 | 4.03 | 113 | 0.850 | 1.79 | 1.14 | 0.002 |  |
| c4_nprobe32 | 0.006 | 3.33 | 99 | 0.884 | 1.99 | 1.12 | 0.002 | BEST_OVERALL |
| c5_protect033 | 0.007 | 3.37 | 98 | 0.905 | 2.12 | 1.05 | 0.003 |  |
| c6_protect05 | 0.009 | 3.85 | 112 | 0.874 | 1.85 | 1.13 | 0.002 |  |
| c7_rms025 | 0.012 | 3.75 | 104 | 0.842 | 1.99 | 1.15 | 0.002 |  |
| c8_rms10 | 0.008 | 3.91 | 120 | 0.856 | 1.76 | 1.14 | 0.002 |  |
| c9_fp32 | 0.009 | 3.40 | 101 | 0.850 | 1.89 | 1.13 | 0.003 |  |
| d1_aiden | 0.012 | 3.60 | 112 | 0.788 | 0.60 | 1.03 | 0.002 |  |
| d2_eric | 0.012 | 3.00 | 104 | 0.841 | 1.35 | 1.05 | 0.005 |  |
| d3_dylan | 0.010 | 3.38 | 97 | 0.910 | 1.21 | 1.03 | 0.003 |  |
| e1_stream_pm_ctx1000 | 0.033 | 5.74 | 138 | 0.670 | 3.18 | 1.00 | 0.000 |  |
| e2_stream_pm_ctx2000 | 0.023 | 4.87 | 116 | 0.746 | 3.99 | 0.93 | 0.000 |  |
| e3_stream_rmvpe_ctx1000 | 0.003 | 2.24 | 86 | 0.890 | 3.50 | 1.06 | 0.001 |  |
| e4_stream_rmvpe_ctx2000 | 0.009 | 3.59 | 108 | 0.841 | 3.57 | 1.08 | 0.001 | BEST_STREAM |

### Proxy table: Goku (target 145 Hz; partial matrix)

| variant | octave_err | jump/s | smooth cents | f0 corr | register err st | voiced ratio | hiss | flags |
|---|---|---|---|---|---|---|---|---|
| ref_harvest_idx05 | 0.006 | 2.39 | 92 | 0.917 | 0.30 | 1.02 | 0.067 |  |
| v00_baseline | 0.015 | 4.31 | 116 | 0.607 | 3.65 | 0.83 | 0.021 |  |
| a1_pm_idx05 | 0.016 | 4.94 | 137 | 0.596 | 3.60 | 0.87 | 0.028 |  |
| a4_rmvpe_post_idx05 | 0.006 | 2.89 | 106 | 0.868 | 4.05 | 0.96 | 0.060 |  |
| a5_harvest_idx05 | 0.006 | 1.74 | 85 | 0.929 | 4.00 | 0.97 | 0.047 | BEST_OVERALL |
| b1_shift_fixed_0 | 0.008 | 2.18 | 93 | 0.905 | 4.00 | 0.99 | 0.066 |  |
| b2_shift_fixed_p2 | 0.007 | 2.68 | 86 | 0.908 | 3.66 | 1.01 | 0.063 |  |
| b4_autoshift | 0.009 | 2.44 | 87 | 0.886 | 0.54 | 1.00 | 0.056 |  |
| b5_autoshift_half | 0.006 | 2.55 | 89 | 0.895 | 2.00 | 0.99 | 0.060 |  |
| c1_idx03 | 0.007 | 3.02 | 95 | 0.893 | 0.52 | 1.01 | 0.054 |  |
| c2_idx075 | 0.006 | 1.94 | 87 | 0.913 | 0.57 | 1.03 | 0.071 |  |
| c5_protect033 | 0.005 | 2.90 | 99 | 0.872 | 0.52 | 1.03 | 0.058 |  |
| c6_protect05 | 0.009 | 2.62 | 94 | 0.836 | 0.47 | 1.03 | 0.069 |  |
| e4_stream_rmvpe_ctx2000 | 0.012 | 2.95 | 109 | 0.900 | 3.63 | 1.07 | 0.027 | BEST_STREAM |

Read with care: `register err` is the output median f0 vs the target median. Stage A/B/C rows carry the stage's own base (stage A has no auto shift). `retrieval_dist` and index-blended scores are circular (see plan 4); `hiss` for Goku (about 0.05-0.07) is a model property. Metrics use harvest on the output, which itself yields about 3 jumps/s on good audio.

### Headline proxy improvements (Obama, baseline v00 -> chosen)

| metric | v00_baseline (pm) | c4/c5 (rmvpe + auto shift + index + nprobe32 / protect) |
|---|---|---|
| f0 jumps per s | 5.9 | 3.3 |
| octave error rate | 0.020 | 0.006-0.007 |
| f0 track corr | 0.74 | 0.88-0.905 |
| register error (st) | 3.7 | 1.9-2.1 (auto shift 0.5), 1.0 with strength 1.0 (b4) |
| smoothness (cents) | 138 | 98-99 |

Goku: auto shift cuts register error 3.6-4.0 -> 0.3-0.5 st; harvest/rmvpe + index lifts f0 corr 0.61 -> 0.87-0.93 and jumps 4.3 -> 1.7-2.9/s. Stream (rmvpe, ctx 2000): Obama join_max about 2.0, Goku about 3.5 (above the ~2 target); pm stream is clearly worse (octave 0.02-0.03, corr 0.67-0.75). Streams have no auto pitch (fixed shift only), so their register error stays about 3.5 st.

### Mutation tripwires

| mutation | result |
|---|---|
| index path fix reverted (`read_index(index_path_stripped)`) | KILLED (legacy tests) |
| rmvpe unavailable falls back silently to pm | SURVIVED at first; test `test_rmvpe_unavailable_is_an_error_not_a_silent_pm_fallback` added; now KILLED |
| octave repair removed | KILLED |
| `use_index` default ON | KILLED |
| `protect` not forwarded | KILLED |
| `rms_mix_rate` not forwarded | KILLED |
| stream_open accepts any f0_method | KILLED |
| stream_open drops `f0_algo` | KILLED |

### RMVPE provenance

`rmvpe.pt`, 181184272 bytes, sha256 `6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193` (re-verified), at `/home/cbash23/.cache/listen-speed/rvc-quality/weights/rmvpe.pt` (env `RVC_RMVPE_PATH` overrides). Pickle inspected without executing: only `collections.OrderedDict`, `torch.HalfStorage`, `torch._utils._rebuild_tensor_v2`; no STACK_GLOBAL/os/builtins/eval. Loaded with `torch.load(weights_only=True)` (`legacy/rvc_f0.py`), strict state-dict match. Record: `legacy/rvc_f0_weights.md`. Runtime stays offline.

### Recommended config (proxies only; enable per profile after listening)

Whole path, Obama and Goku (shared; the auto-shift targets are per model):

```yaml
speech:
  voice_conversion:
    f0_method_override: rmvpe
    use_index: true
    index_rate: 0.5
    index_nprobe: 32
    protect: 0.33
    auto_pitch: true
    auto_pitch_strength: 0.5      # Obama; 1.0 measures a smaller register error, listen to both
    target_f0_median_hz: {Obama: 108, Goku: 145}
    f0_post_octave_repair: true   # safety only; median/clamp raised jumps and roughness (a2 vs a3), leave off
    streaming: false              # whole path first; stream has no auto pitch and Goku join_max is 3.5
```

Stream (if kept on): `stream_f0_algo: rmvpe`, `stream_context_ms: 2000`, plus the same `use_index`, `index_rate`, `protect`.

Caveats: rmvpe + post-processing is not better than plain rmvpe on these proxies; index rate 0.5 vs 0.75 is within noise; for Goku `f0_method_override: harvest` with the index scored best in the unshifted stage (a5: corr 0.93, jumps 1.7), so try that too. One `f0_method_override` applies to both profiles.

### Listen to first (Obama; same sentence s0..s2 each; compare against v00_baseline)

1. `samples/Obama/c4_nprobe32/` (BEST_OVERALL proxy: rmvpe + index nprobe 32 + auto shift 0.5).
2. `samples/Obama/c5_protect033/` (best f0 track corr 0.905; breathier consonants protected).
3. `samples/Obama/d3_dylan/` (different Qwen source speaker, Dylan: best smoothness/corr, tests the "wrong timbre" complaint at its source; `d1_aiden` has the lowest register error 0.6 st).
Reference: `samples/Obama/v00_baseline/` (today). Then Goku: `samples/Goku/ref_harvest_idx05/`, then `samples/Goku/c2_idx075/`. Base path `/home/cbash23/.cache/listen-speed/rvc-quality/`.

### Open issues

1. Nobody has listened; every number is a proxy. Index blending can change timbre of both paths (off by default).
2. Streams cannot auto-shift; a fixed per-profile shift for streams is untested. Goku stream join_max 3.5 exceeds the ~2 target.
3. `f0_method_override` is global, not per profile.
4. Goku matrix is partial (14 variants, no D stage); `change_rms` follows the input envelope coarsely (0.5 s hop).
5. `convert` with `index_nprobe` changes the cached index's nprobe for the next stream open.
6. `rmvpe_available()` checks only the default path, not `RVC_RMVPE_PATH`.
7. `rvc_cli` is a dirty submodule (edits to named functions only). Existing stash `stash@{0}` origin unknown.
8. The RMVPE tests that need pyworld/parselmouth skip under the bot venv; they pass under the legacy venv.

### Status (after RVC output quality)

| ID | Status | Note |
|---|---|---|
| Q1-Q4 RVC quality build | DONE (code), reviewed | all new settings default to old behaviour |
| RVC quality listen | READY for owner | order above; then edit `config/app.yaml` |
| RVC quality enable | BLOCKED | after listening |
