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
   "cuda_available":true,"device":"cuda:0","gpu_name":"NVIDIA GeForce RTX 4070 Ti SUPER"}
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
   "pitch_shift":2}
← {"id":3,"status":"ok","output_wav_path":"/abs/path/out.wav",
   "elapsed_seconds":0.755,"model":"Goku","pitch_shift":2}
```
- Paths are JSON fields — **no shell is involved**; untrusted text never
  reaches a command line.
- `pitch_shift` is optional (semitones); omitted ⇒ the model YAML's
  `f0_pitch` default (usually 0).
- The conversion runs in a scratch temp dir and the result is moved to
  `output_wav_path`; the worker validates the output (non-empty, finite
  samples, non-zero peak, positive duration) and fails the request loudly on
  NaN/Inf/silence.
- Timeout: `RVC_WORKER_CONVERT_TIMEOUT` (default 120 s, SIGALRM-enforced).

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
