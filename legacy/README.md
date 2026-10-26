# legacy/ — isolated legacy RVC runtime

This directory hosts the **only** sanctioned bridge to the legacy RVC stack
(`rvc_cli/` submodule, commit `535d364`). Per REWORK_HANDOFF.md sections 4.3
and 10.3, nothing outside `legacy/` may import `rvc_cli` or its dependency
graph; Discord / orchestration code must go through the `VoiceConverter`
interface in `src/` and talk to the worker over the JSON IPC protocol below.

## Why isolated

The legacy stack requires Python 3.10 + `fairseq==0.12.2` +
`torch 2.5.0+cu121`. The new application environment is Python 3.12; the
legacy dependency graph must never leak into it.

## Files

| File | Purpose |
|---|---|
| `rvc_worker.py` | Long-running voice-conversion worker (line-based JSON IPC on stdin/stdout) |
| `rvc_stream.py` | Streaming engine (`StreamEngine`, `StreamParams`, `open_engine`); lazy-imported by the worker on `stream_open` only |
| `README.md` | This document |

## Runtime facts (verified)

- Worker interpreter: repo-root **`.venv/bin/python`** — Python **3.10.22**
  (uv-managed CPython), already containing:
  - `fairseq 0.12.2`
  - `torch 2.5.0+cu121` (+ `torchaudio 2.5.0+cu121`), CUDA available
  - `faiss-cpu`, `librosa`, `soundfile`, `numpy 1.26.4`, `pydub`, `audiostretchy`
- The worker is **stdlib-only by itself**: it adds no new packages to `.venv`.
- `ffmpeg` **CLI** is a system prerequisite (`/usr/bin/ffmpeg`, 6.1.1). The
  legacy venv lacks the `ffmpeg-python` package (and `.venv` must not be
  modified), so the worker injects a minimal stdlib shim reproducing the one
  call pattern `rvc_cli/my_utils.py:load_audio` uses
  (`ffmpeg -nostdin -threads 0 -i <in> -f f32le -acodec pcm_f32le -ac 1 -ar <sr> -`
  → stdout pipe). If the real `ffmpeg-python` package is importable, the shim
  is a no-op.
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER. The worker disables cuDNN SDP
  (`torch.backends.cuda.enable_cudnn_sdp(False)`) because torch 2.5.0 cuDNN
  SDPA raises "No execution plans support the graph" inside HuBERT attention
  on this stack; math otherwise identical to the previously working setup.
- Hubert weights: `hubert_base.pt` at the **repo root**; the worker `chdir`s
  to the repo root itself (legacy isolation exemption) so `voice_converter.py`
  resolves it exactly as `audio_generator.py` historically did.

## Models

Models live in `rvc_cli/models/<NAME>/` with `<name>.yaml`, one `.pth`, one
`.index` (Goku, Obama, Biden, Eminem, Gojo, Junkook, KanyeWest, L,
MichaelJackson, PeterGriffin, Plankton, Shrek, Spongebob, Trump, Tupac,
Vader, Vegeta). Conversion defaults (`index_rate`, `f0_method`, `f0_pitch`,
…) come from each model's YAML — the same values the legacy
`audio_generator.py` used — preserving current behavior.

## Starting / stopping

```bash
# start (typically from the VoiceConverter adapter, not a shell)
.venv/bin/python legacy/rvc_worker.py
```

- The worker prints one JSON line `{"status":"ok","action":"ready",...}` on
  startup and then serves one reply per request until `shutdown` or stdin EOF.
- All incidental legacy output (faiss/torch/fairseq logs, `print()` banners)
  is redirected to **stderr**; stdout carries protocol JSON only.
- Stop via the `shutdown` action (clean) or by closing stdin / SIGTERM.

## Protocol (line-based JSON on stdin/stdout)

Every request is a single JSON object per line; every request produces
exactly one reply line `{"id": <echoed id>, "status": "ok"|"error", ...}`.
Errors carry `"error"` (message) and usually `"traceback"` (trimmed).

### `health`
```json
→ {"id":1,"action":"health"}
← {"id":1,"status":"ok","loaded_model":"Goku","torch":"2.5.0+cu121",
   "cuda_available":true,"device":"cuda:0","gpu_name":"NVIDIA GeForce RTX 4070 Ti SUPER",
   "stream":{"open":false,"stream_id":null}}
```

### `load-model`
```json
→ {"id":2,"action":"load-model","name":"Goku"}
← {"id":2,"status":"ok","model":"Goku","variant":"goku","config_yaml":"goku.yaml",
   "config_switched":false,"device":"cuda:0","target_sample_rate":40000,
   "version":"v1","index_rate":0.4,"f0_method":"harvest","default_f0_pitch":0}
```
Loads HuBERT + the target model **persistently** (one weight set per worker;
to switch model directories, start another worker). Model names are validated
against the `rvc_cli/models/` directory (no path traversal). Timeout:
`RVC_WORKER_LOAD_TIMEOUT` (default 300 s).

**Variant selection.** A model directory may contain several YAML configs
(`rvc_cli/models/Obama/` ships `obama.yaml` and `obama_trans.yaml`, sharing
`obama.pth` + `obama.index`). The optional `"variant"` field selects the EXACT
`<variant>.yaml` in the directory (matched case-insensitively against the yaml
stems); a missing variant yaml is a loud `status: "error"` — the worker never
falls back to "first config found". When `"variant"` is omitted the default is
the yaml whose stem matches the directory name (`Obama` → `obama.yaml`), or
the single yaml if the directory has exactly one; a multi-yaml directory
satisfying neither rule errors and asks for an explicit variant.

**In-process variant switching.** `config_switched: true` in the reply means
the requested yaml points at the already-resident weights (same `.pth`): the
worker only swapped the YAML-derived conversion config (`vc_args`), keeping
the loaded `VoiceConverter` — no weight reload, no process restart. The
adapter therefore treats `(model directory, variant)` as its cache identity
and switches variants without restarting the worker.

### `convert`
```json
→ {"id":3,"action":"convert",
   "input_wav_path":"/abs/path/in.wav",
   "output_wav_path":"/abs/path/out.wav",
   "pitch_shift":2,
   "f0_method":"pm"}
← {"id":3,"status":"ok","output_wav_path":"/abs/path/out.wav",
   "elapsed_seconds":0.755,"model":"Goku","pitch_shift":2,"f0_method":"pm"}
```
- Paths are JSON fields — **no shell is involved**; untrusted text never
  reaches a command line.
- `pitch_shift` is optional (semitones); omitted ⇒ the model YAML's
  `f0_pitch` default (usually 0).
- `f0_method` is optional; omitted ⇒ the model YAML's `f0_method` (shipped
  default `harvest`). When present it must be one of `pm`, `harvest`, `dio`,
  `crepe`, `crepe-tiny` (anything else is a loud `status: "error"`); it
  overrides the YAML value for this conversion only (the loaded config is
  never mutated). The adapter sends it when the app config sets
  `speech.voice_conversion.f0_method_override` — `pm` (parselmouth) is the
  fast CPU option, `crepe`/`crepe-tiny` extract pitch on the GPU. The reply
  echoes the f0 method actually used.
- The conversion runs in a scratch temp dir and the result is moved to
  `output_wav_path`; the worker validates the output (non-empty, finite
  samples, non-zero peak, positive duration) and fails the request loudly on
  NaN/Inf/silence.
- Timeout: `RVC_WORKER_CONVERT_TIMEOUT` (default 120 s, SIGALRM-enforced).

### Streaming: `stream_open` / `stream_push` / `stream_close`
Opt-in streaming conversion (design: `docs/RVC_STREAMING_DESIGN.md`). The engine lives in
`legacy/rvc_stream.py`; the worker only validates, frames and times requests. **`f0_method` is `pm`
only** (omitted => `pm`), and there is **one stream per worker**: `stream_open` replaces any open
stream. PCM is **base64 (standard alphabet) of little-endian float32 mono samples inside the JSON
line** (no binary framing). Input 24 kHz, output 48 kHz.
```json
→ {"id":10,"action":"stream_open","stream_id":"s-1a2b","pitch_shift":0,"f0_method":"pm",
   "first_block_ms":100,"block_ms":320,"context_ms":1000,"lookahead_ms":100,"crossfade_ms":40,"search_ms":20}
← {"id":10,"status":"ok","stream_id":"s-1a2b","input_sample_rate":24000,"output_sample_rate":48000,
   "sample_format":"f32le","channels":1,"min_input_samples_for_first_output":5760,
   "replaced_stream":false,"model":"Goku","f0_method":"pm"}
→ {"id":11,"action":"stream_push","stream_id":"s-1a2b","pcm_b64":"...","samples":6000}
← {"id":11,"status":"ok","stream_id":"s-1a2b","pcm_b64":"...","samples":12000,
   "input_total":6000,"output_total":12000,"compute_ms":3.2}
→ {"id":12,"action":"stream_close","stream_id":"s-1a2b","flush":true}
← {"id":12,"status":"ok","stream_id":"s-1a2b","closed":true,"pcm_b64":"...","samples":0,
   "input_total":6000,"output_total":12000,"compute_ms_total":3.2,"steps":1}
```
- `stream_id` matches `^[A-Za-z0-9_-]{1,64}$`; `pitch_shift` and the `*_ms` fields
  (`first_block_ms`, `block_ms`, `context_ms`, `lookahead_ms`, `crossfade_ms`, `search_ms`) are
  integers, all optional (range/multiple-of-20 checks are the engine's `StreamParams`).
- `stream_push`: `samples` must be an int equal to the decoded length, `0..48000` (2 s); `pcm_b64`
  may be `""` in replies. `flush:true` on close returns the tail so `output_total == 2*input_total`;
  `flush:false` aborts and discards (`samples` 0, `pcm_b64` `""`). Closing an unknown or
  non-matching id is `ok` with `closed:false` (idempotent).
- Errors: `{"id","status":"error","error",["traceback"],"stream_closed":bool}`. Bad base64, byte length
  not a multiple of 4, samples mismatch or range, NaN/Inf input, non-finite output, engine exception
  or timeout abort and drop the stream (`stream_closed:true`). `no model loaded; send load-model
  first`, `no open stream`, `stream id mismatch`, bad `stream_open` fields leave state as it was
  (`stream_closed:false`).
- `load-model` while a stream is open aborts it first. `convert` and `health` stay legal during a
  stream; `health` gains `"stream":{"open":bool,"stream_id":str|null}` (additive).
- Timeouts (SIGALRM): `RVC_WORKER_STREAM_OPEN_TIMEOUT` (default 30 s) for open,
  `RVC_WORKER_STREAM_TIMEOUT` (default 10 s) per push/close.
- **Fake mode**: `RVC_WORKER_FAKE=1` makes `load-model` return canned info (device `fake`) without
  importing torch/rvc_cli, `health` skip torch, and the default stream engine a 2x sample-repeat stub
  (a push whose first sample is exactly `1234.5` raises `fake boom`). Used by
  `tests/unit/test_rvc_worker_stream.py` and bot-side tests.

### `shutdown`
```json
→ {"id":9,"action":"shutdown"}
← {"id":9,"status":"ok","action":"shutdown"}
```
Worker exits 0. Malformed lines get `{"id":null,"status":"error",...}` and
the loop continues.

## Testing

```bash
# real GPU integration test (test process under main venv, worker under .venv)
.venv-main/bin/python -m pytest tests/integration/test_rvc_worker.py -m "rvc and integration" -s
```

The test synthesizes a sine fixture under `tests/fixtures/`, starts the
worker, loads a real model (Goku preferred), converts, and asserts existence,
duration, no NaN/Inf, sane peak; it also writes observed latencies to
`tests/fixtures/rvc_worker_last_latency.txt` for the section-35 baseline.
Without the legacy assets the tests skip cleanly.

## Ownership / constraints

- Do not modify `rvc_cli/` (submodule) or `.venv`.
- Do not import `legacy/rvc_worker.py` from `src/` code; spawn the process
  and use the IPC protocol.
- Never pass `.env` values through the protocol.
