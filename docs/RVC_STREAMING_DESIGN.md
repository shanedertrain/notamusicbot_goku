# Streaming RVC: design

Status: design + measured prototype. Streaming is a NEW opt-in path, default OFF
(`speech.voice_conversion.streaming: false`). The whole-file `convert` path,
its protocol and its tests are unchanged.

## 1. Goal and result of the feasibility work

TTS stream (mono float32, 24 kHz, ~0.2-0.3 s chunks) -> Goku voice as a
continuous stream -> existing gain/limit -> 48 kHz stereo s16le. Targets:
first converted audio within ~250 ms of the first TTS chunk, RTF well below 1,
no clicks at joins, quality close to whole-utterance conversion.

A working reference prototype exists (GPU, real Goku model, f0 `pm`):

- `/home/cbash23/.cache/listen-speed/rvc-stream/proto_stream.py` (engine)
- `/home/cbash23/.cache/listen-speed/rvc-stream/eval_stream.py` (metrics)

It is the algorithm the builders port into `legacy/rvc_stream.py`.

### Facts found in the code (all verified by reading / running)

| Fact | Consequence |
|---|---|
| Goku: RVC v1 (HuBERT layer 9 + final_proj, 256-d), `tgt_sr` 40000, `is_half` fp16, sid 0, yaml f0 `harvest`, index_rate 0.4 | model rate 40 kHz -> must resample to 48 kHz |
| f0 methods in `vc_infer_pipeline.py`: `pm`, `harvest`, `dio`, `crepe`, `crepe-tiny`. NO rmvpe code and no rmvpe weights anywhere (only `hubert_base.pt`, `hubert_base.state_dict.pt`) | only `pm` is block-causal and cheap (1.5-2 ms per 1.5 s window). harvest/dio are whole-signal; crepe is GPU but out of scope. Streaming = `pm` only |
| `pm` = parselmouth autocorrelation pitch, 10 ms hop, window ~60 ms (floor 50 Hz) | edge frames of a window are unreliable: lookahead >= 60-100 ms and left context are needed |
| NSF generator (`SineGen`): fundamental phase starts at 0 at the START OF EACH WINDOW, harmonics get `torch.rand` initial phase, plus `randn` noise; synthesizer `infer` also draws `randn_like` | separately generated blocks are NOT phase continuous, even with identical f0 -> SOLA (cross-correlation alignment) is mandatory. Two whole-utterance runs already differ by mel L1 1.4-1.9 dB (stochastic noise floor) |
| Output frame rate: one synth frame per 10 ms (HuBERT 20 ms frames, features interpolated x2). Output is ~10-20 ms shorter than input | pad/trim so output length is EXACT (see 3.4) |
| HuBERT/net_g accept any length; `vc()` is already called per segment in whole-file mode with 1 s reflect pad (`x_pad`) | window-wise inference is within what the model code does today |
| **The faiss index is silently NOT used today.** `index_path_stripped` is relative to `rvc_cli/` but the worker `chdir`s to the repo root, so `os.path.exists(file_index)` is false in `pipeline()` | whole path = no index. Streaming keeps `use_index` false by default to stay equivalent. (Also `pipeline()` re-reads the index from disk on every call; a streaming engine must never do that.) |
| `pipeline()` calls `torch.cuda.empty_cache()` after every segment | streaming omits it (slow, pointless) |
| `rvc_cli` `Config()` runs argparse on `sys.argv` at import | any harness script must clear `sys.argv` before importing `voice_converter` |
| Worker is single threaded, line JSON, lock-step (one reply per request), SIGALRM timeouts; bot client serializes with `_transaction_lock` | streaming = request/reply per push, no protocol redesign needed |
| `listen/turn_taking.py` only reads `synth.can_stream` and `synth.stream_sentence(chunk)` | no change to turn_taking, Discord untouched |

## 2. Measured defaults (RTX 4070 Ti SUPER, GPU shared with the live bot, 2 TTS clips of 4.2 s and 5.1 s)

Metrics vs whole-utterance conversion with the same f0 `pm`. The whole-vs-whole row is the noise floor
of the model itself.

| config (first/block/ctx/look/xfade/search ms) | RTF | step ms (warm) | f0 corr | f0 median cents | mel L1 dB | worst join ratio |
|---|---|---|---|---|---|---|
| whole vs whole (noise floor) | - | - | 0.991-0.9999 | 2.1-2.3 | 1.4-1.9 | 1.85-2.45 (whole audio itself) |
| 100/240/1000/100/40/20 | 0.10-0.16 | 21-36 | 0.988 | 6.7 | 3.0 | 1.9 |
| 100/320/1000/100/40/20 (**default**) | 0.08-0.15 | 21-42 | 0.987-0.9995 | 5.6-5.9 | 3.2-3.6 | 2.0-2.1 |
| 100/320/1000/140/40/20 | 0.07-0.09 | 20-25 | 0.978-0.999 | 5.6-8.0 | 3.3-3.8 | 2.6-2.9 |
| 60/320/1000/80/40/20 (low latency) | 0.08-0.09 | 21-24 | 0.974-0.999 | 6.7 | 3.6 | 1.8-2.1 |
| 100/400/1000/100/40/20 | 0.06 | 22 | 0.980-0.998 | 4.8-9.0 | 3.1-4.1 | 1.8-2.4 |
| 80/160/1000/100/40/20 (small blocks) | 0.14 | 21 | 0.953 | 9.5 | 3.6 | 2.8 (worse) |
| 100/240/500/100/40/20 (short context) | 0.10 | 21 | 0.990 | 6.1 | 3.5 | 2.1 |
| 100/240/2000/100/40/20 | 0.12 | 25 | 0.985 | 12.2 | 2.8 | 2.2 |
| 100/240/1000/60/40/20 (short lookahead) | 0.10 | 22 | 0.981 | 6.5 | 3.2 | 2.3 |
| 100/240/1000/200/40/20 | 0.10 | 22 | 0.957 | 12.3 | 3.4 | 2.1 |
| 100/240/1000/100/60/30 (long xfade/search) | 0.10 | 22 | 0.966 | 10.3 | 3.8 | 2.5 |
| 100/240/1000/100/20/10 | 0.10 | 20 | 0.987 | 5.8 | 3.0 | 1.9 |

Reading: join clicks are at the level of the unconverted-whole audio (ratio ~2, no outliers), RTF is 0.06-0.16, f0 tracking
is within ~6 cents median. mel L1 is ~1.5-2 dB above the model's own run-to-run noise. Block >= 320 beat 160-240;
context 1000 ms is the knee (500 loses spectral accuracy, 2000 costs time, no gain); lookahead 100 ms is the knee
(200 hurts f0 correlation because `pm` voiced decisions drift, 60 hurts the end of each block); crossfade 40 / search 20.
Single-run numbers carry noise of roughly +-0.5 dB mel and +-0.01 f0 corr; the reviewer re-measures (section 9).

**Defaults:** `first_block_ms=100`, `block_ms=320`, `context_ms=1000`, `lookahead_ms=100`,
`crossfade_ms=40`, `search_ms=20`. All multiples of 20 ms. Window = search+context+block+crossfade+lookahead =
1.48 s. A smaller FIRST block is free (block 0 has no left join) and is what meets the latency target.

## 3. Algorithm (port of the original RVC realtime GUI idea, adapted)

Input timeline is 24 kHz samples. `pos` = nominal input position of the next block to emit. All lengths below in
24 kHz samples (`ms * 24`); output is 48 kHz (`x2`).

### 3.1 Trigger
Block `n` has length `blen = first_block if pos == 0 else block`. It is processed as soon as
`buffered_total >= pos + blen + crossfade + lookahead`. First output therefore needs
`first_block + crossfade + lookahead = 240 ms` of TTS audio (one 0.25 s chunk, or two if a chunk is shorter).

### 3.2 Window
`window = input[pos - search - context : pos + blen + crossfade + lookahead]`; samples before the stream start are
filled by reflect padding (like `x_pad` in `pipeline()`), samples beyond the end (only at flush) by reflect padding.
Windows therefore always have full length -> constant tensor shapes (no cudnn / kernel re-tuning after the first call).

### 3.3 Window conversion (`convert_window`, stateless function of the window)
1. 24k -> 16k `scipy.signal.resample_poly(win, 2, 3)`; 48 Hz high-pass `filtfilt` (same `bh, ah` as `vc_infer_pipeline`).
2. f0: `VC.get_f0_pm_computation` on the 16 kHz window, then the SAME mel quantisation + pitch shift as `VC.get_f0`
   (copy those ~12 lines instead of calling `get_f0`, which `print`s on every call). `p_len = len16 // 160`.
3. HuBERT `extract_features(output_layer=9)` + `final_proj` (v1) or layer 12 (v2), `interpolate x2`, optional index blend
   (OFF by default), `net_g.infer(feats, p_len, pitch, pitchf, sid)` under `torch.no_grad`. No `empty_cache`.
4. 40 kHz -> 48 kHz `resample_poly(y, 6, 5)` (generic: `48000/gcd` up, `tgt_sr/gcd` down).
5. zero-pad / truncate to exactly `2 * len(window24)` samples. Output index 0 == window start.

### 3.4 SOLA join (`StreamEngine`)
`seg = y[2*context:]` (starts at nominal `pos - search`). Fixed `Fo = 2*crossfade`, `So = 2*search`.
- First block: emit `seg[So : So + 2*blen]` with a 2 ms linear fade-in; `tail = seg[So+2*blen : So+2*blen+Fo]`.
- Later blocks: `cand = seg[:So+Fo]`; `d = argmax_k  corr(cand[k:k+Fo], tail) / ||cand[k:k+Fo]||` for `k in [0, So]`
  (`np.correlate` valid mode + running energy, as in the prototype); `s2 = seg[d:]`;
  emit `crossfade(tail, s2[:Fo])` (rising raised-cosine) `+ s2[Fo : 2*blen]` (exactly `2*blen` samples);
  `tail = s2[2*blen : 2*blen+Fo]`. Search always looks at most `search` ms AHEAD of the nominal start, so content
  can slip by up to `search` ms, never accumulates (each window is nominal-anchored) and never repeats samples audibly.
- `pos += blen`. Every emitted block is exactly `2*blen` samples, so `output_total == 2 * pos`.
- **Flush** (`close`): remaining output = `2*total_input - 2*pos` (< `2*(blen+crossfade+lookahead)`), converted from one
  final window (reflect-padded past the end), same SOLA join, truncated to that exact length, 10 ms linear fade-out.
  Result: total output samples == 2 x total input samples, always, so the reply cap logic and durations are unchanged.
- Engine memory: keep only `input[pos - search - context:]` (trim after each step).
- The engine output is a pure function of the input sample sequence given a deterministic `convert_window`, **independent
  of how the input was chunked**. This is a unit-test property.

### 3.5 What deliberately differs from whole-utterance conversion
- No cross-window state in HuBERT (it sees 1 s of left context only); stochastic synth noise differs per window.
- No faiss index (the whole path does not use it either, see section 1).
- `pm` only. A profile / override asking for another f0 method is ignored for streams (log once) because the
  other methods cannot stream.

## 4. Architecture

```
Qwen TTS stream_pcm(): mono f32 24k chunks
        |   (bot, main venv)
        v
ReplySynthesizer.stream_sentence()           listen/reply.py   [T4]
  backend == rvc & streaming enabled ->
  converter.convert_stream(chunks, profile)  VoiceConverterRouter -> RvcVoiceConverter   [T3]
        |  JSON lines on worker stdin/stdout, PCM as base64 f32le
        v
legacy/rvc_worker.py stream_open/push/close  [T2]   (legacy venv, torch 2.5)
        |
        v
legacy/rvc_stream.py StreamEngine            [T1]   mono f32 48k blocks
        ^
        |  back in the bot: cap + fade, fixed gain, soft limit, mono->stereo, s16le  (existing encode() in reply.py)
        v
Discord player.play_stream()  (unchanged)
```

Worker process, model loading, GPU lease and the `_transaction_lock` lock-step transport are reused. A push is just a
request; between pushes other requests (a backchannel `convert`) may run. One stream at a time per worker
(`stream_open` replaces any open stream; the bot additionally serialises with `_stream_lock`).

## 5. Wire protocol (additions to the line-JSON protocol; existing actions unchanged)

PCM is carried inside the JSON line as **base64 of little-endian float32 mono samples**. (A binary side channel would
break the line protocol and the stderr redirection trick; measured size is small: 0.25 s in = 24 KB, 0.32 s out
= 61 KB raw / 82 KB base64, encode+decode < 1 ms.) Reply lines can therefore reach ~120 KB; both ends read whole lines.

```
-> {"id":10,"action":"stream_open","stream_id":"s-1a2b","pitch_shift":0,"f0_method":"pm",
    "first_block_ms":100,"block_ms":320,"context_ms":1000,"lookahead_ms":100,"crossfade_ms":40,"search_ms":20}
<- {"id":10,"status":"ok","stream_id":"s-1a2b","input_sample_rate":24000,"output_sample_rate":48000,
    "sample_format":"f32le","channels":1,"min_input_samples_for_first_output":5760,
    "replaced_stream":false,"model":"Goku","f0_method":"pm"}

-> {"id":11,"action":"stream_push","stream_id":"s-1a2b","pcm_b64":"<f32le 24k mono>","samples":6000}
<- {"id":11,"status":"ok","stream_id":"s-1a2b","pcm_b64":"<f32le 48k mono, may be empty string>","samples":0,
    "input_total":6000,"output_total":0,"compute_ms":0.0}

-> {"id":12,"action":"stream_close","stream_id":"s-1a2b","flush":true}
<- {"id":12,"status":"ok","stream_id":"s-1a2b","closed":true,"pcm_b64":"...","samples":9600,
    "input_total":96000,"output_total":192000,"compute_ms_total":412.0,"steps":6}
```
- `stream_id`: `^[A-Za-z0-9_-]{1,64}$`, chosen by the client. All tuning fields optional (engine defaults), validated
  (multiples of 20 ms, ranges in `StreamParams.validate`). `f0_method` optional, only `"pm"` accepted (else error).
- `stream_push`: `samples` must equal the decoded length; max 2.0 s (48000 samples) per push; NaN/Inf or length mismatch ->
  error AND the stream is closed (`"stream_closed": true` in the error reply). Output is validated finite.
- `stream_close`: `flush:true` returns the final audio (normal end); `flush:false` aborts and discards (cancel path).
  Closing an unknown / already closed id is `status ok` with `"closed": false` (idempotent, so cancel races are harmless).
- Errors: `{"id":..,"status":"error","error":"...","traceback":"...","stream_closed":true|false}`. Any engine
  exception closes the stream. `no model loaded` / `no open stream` / `stream id mismatch` are errors with `stream_closed:false`.
- `load-model` while a stream is open aborts that stream first. `convert` and `health` stay legal during a stream
  (health gains `"stream": {"open": bool, "stream_id": str|null}`).
- Timeouts (SIGALRM, worker side): `RVC_WORKER_STREAM_OPEN_TIMEOUT` 30 s, `RVC_WORKER_STREAM_TIMEOUT` 10 s per push/close.
  Bot side: push/close request timeout 10 s (`stream_timeout_s`), open 30 s. A bot-side timeout kills the worker exactly
  like today (`WorkerTimeoutError`, restarted lazily on next use).
- Abandoned stream (client died): the next `stream_open` replaces it; there is no background timer.

## 6. Config (`speech.voice_conversion`)

```yaml
voice_conversion:
  f0_method_override: null
  streaming: false          # opt-in; needs the rvc backend, f0 is forced to pm for streams
  stream_first_block_ms: 100
  stream_block_ms: 320
  stream_context_ms: 1000
  stream_lookahead_ms: 100
  stream_crossfade_ms: 40
  stream_search_ms: 20
```
All `*_ms` are multiples of 20 (validated). `streaming:false` => every code path behaves exactly as today.

## 7. Latency budget (first converted audio after the first TTS chunk)

| step | ms |
|---|---|
| wait for 240 ms of audio (one 0.25 s chunk; if the first chunk is shorter, one more chunk) | 0 (or one chunk interval) |
| base64 + JSON + pipe | ~2 |
| resample + high-pass + `pm` f0 | ~4 |
| HuBERT (1.48 s window, fp16) | ~10 |
| synthesizer + NSF | ~10-15 |
| resample 40->48k + SOLA + base64 reply + bot gain/limit | ~3 |
| **total after the unlocking chunk (warm)** | **~30-45 (measured 22-42)** |
| cold first call (CUDA kernels) | 150 ms -> hidden by `warm_stream` at app warm-up |

Steady state: each 320 ms of TTS audio costs ~25-40 ms of GPU, RTF 0.08-0.16, plus 480 ms (block+crossfade+lookahead)
buffer behind the TTS head, which is hidden because TTS generates faster than real time. Low-latency preset for TTS
chunks shorter than 240 ms: `first_block_ms=60, lookahead_ms=80` (needs 180 ms).

## 8. Failure and cancel semantics

| Event | Behaviour |
|---|---|
| streaming off / rvc unavailable / router closed | `can_stream` False -> existing whole-sentence path (same errors/policy as today) |
| barge-in / cancel while streaming | async generator `finally` -> `stream_close(flush=false)` (shielded, 2 s budget); TTS generator closed after the converter generator (order matters) |
| TTS error mid-stream | converter aborts the stream, error propagates -> `ReplySynthesisError` (existing handling, partial audio already played, same as passthrough streaming) |
| worker error reply on push | stream is closed worker-side; bot raises `RvcStreamError`; no mid-stream retry |
| worker dies / pipe closes | `WorkerDiedError`; worker restarted lazily by the next request; no mid-stream recovery |
| push timeout | worker terminated (existing policy) |
| cap reached in `reply.py` | consumer stops pulling -> same cancel path |
| `shutdown()` during a stream | takes the transaction lock between pushes; the next push fails with `WorkerError`, generator ends with an error |
| second concurrent stream | waits on `_stream_lock` (5 s) else `RvcStreamError("another stream is active")` |

## 9. Quality evaluation plan (reviewer, short GPU runs under `flock .../gpu.lock`, < 30 s each)

`legacy/rvc_stream_eval.py` (T1) takes WAVs at 24 kHz, runs whole-utterance (`pm`) twice and the streaming engine, and prints JSON:
`f0_corr`, `f0_cents_med`, `voiced_agree`, `mel_l1_db`, `join_max`, `jump_ref_whole`, `out_ratio`, `rtf`,
`first_latency_ms`. Acceptance: `out_ratio == 1.0` exactly (sample count), `f0_corr >= 0.98`, `f0_cents_med <= 10`,
`mel_l1_db <= noise_floor + 2.5`, `join_max <= max(2.5, 1.3 * jump_ref_whole)`, `rtf <= 0.3`, warm first-step compute
<= 100 ms, chunk-size invariance (push 20 ms vs 250 ms vs 1 s chunks gives identical samples). Also listen to 3 sentences.

## 10. What stays unchanged

`VoiceConverter.convert`, `convert` worker action, `Worker.convert`, `RvcVoiceConverter.convert`, SpeechPipeline,
passthrough streaming (`_Upsampler2x` path), `turn_taking.py`, `discord_bot`, `rvc_cli/` submodule, `.venv`. Existing
tests must pass unmodified. The N1 scan (`test_listen_local_only.py`) covers `listen/`: no network imports there.

## 11. Open items for the owner (not blocking)
1. The faiss index is not applied today (path bug). Fixing it changes timbre of BOTH paths; decide separately.
   `StreamParams.use_index` exists but is not exposed in config.
2. Stream gain for RVC output (`RVC_STREAM_GAIN_DB`, -0.8 dB, calibrated 2026-10-26) needs calibration against postprocess.
3. rmvpe would give better f0 but no weights exist locally; nothing downloaded.
