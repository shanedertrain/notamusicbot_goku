# RVC output-quality plan

Status: plan (2026-10-26). Owner feedback: converted speech sounds robotic/metallic, has pitch wobble / is off-key, and does not sound like the target (right-ish pitch, wrong timbre). No code is changed by this document. Work files: `/home/cbash23/.cache/listen-speed/rvc-quality/`.

## 1. Evidence (measured 2026-10-26)

| # | Finding | Evidence |
|---|---------|----------|
| E1 | The faiss index never loads, in either path | `VoiceConverter.strip_file_index` stores `models/Obama/obama.index` (relative to `rvc_cli/`); the worker does `os.chdir(REPO_ROOT)`, so `os.path.exists()` in `VC.pipeline` is false. Both indexes read fine in the legacy venv: faiss 1.15.1, Obama `IndexIVFFlat` 112043x768 (nlist 2872, **nprobe 1**), Goku `IndexIVFFlat` 32839x256 (nlist 842, nprobe 7). `reconstruct_n` works (0.19 s / 0.02 s load), search 8 ms. Likely the biggest cause of "not the target": timbre comes from Ryan's HuBERT features unblended. |
| E2 | `pm` f0 drops voiced frames and makes octave errors | Ryan sources s0..s5: parselmouth (`voicing_threshold 0.6`) marks 29-54 % of frames voiced; pyworld harvest marks 43-76 %. pm p95 reaches 481 and 660 Hz on two sentences (octave errors; harvest p95 235-260). Unvoiced frames in voiced speech get noise excitation in the NSF = hoarse/metallic; octave jumps = wobble. |
| E3 | `protect` and `rms_mix_rate` do not exist in `rvc_cli/vc_infer_pipeline.py` | `VC.vc` / `VC.pipeline` have no such arguments (this is the old RVC pipeline). Both must be implemented, not just plumbed. |
| E4 | No RMVPE code and no weights anywhere | `find / -iname '*rmvpe*'` empty; only `hubert_base.pt` and `hubert_base.state_dict.pt`. `safetensors`, `huggingface_hub` are NOT installed in the legacy venv; `hf` CLI and `curl` exist on the box. Available f0 libs in the legacy venv: pyworld, parselmouth, torchcrepe, librosa, scipy, soundfile, torchaudio. |
| E5 | f0 backend speed (16 kHz, 7-11 s clips) | pm 3-6 ms, harvest 0.8-1.9 s, dio fast (CPU), crepe GPU. Only pm is cheap and block-causal today; RMVPE is GPU, expected ~10-30 ms per 1-2 s window, so it can stream. |
| E6 | Ryan's register varies per sentence | Harvest median per clip: 156, 90, 136, 104, 103, 163 Hz (overall about 125). Obama target about 100-115 Hz, Goku about 130-160 Hz. A fixed shift of 0 is therefore wrong in a sentence-dependent way, and sentences with a high median (156-163) land well above Obama's range. |
| E7 | HuBERT version per model is already consistent | Obama `.pth`: `version: v2` (layer 12, no `final_proj`), index d=768. Goku `.pth`: no `version` key, loader defaults to `v1` (layer 9 + `final_proj`), index d=256. Matches `VC.vc` and `rvc_stream`. Lever 5 needs only a regression test, no code change. |
| E8 | Existing tests pin pm-only streams | `tests/unit/test_rvc_stream_engine.py::test_harvest_message` (match `streaming supports only pm`), `tests/unit/test_rvc_worker_stream.py::test_non_pm_rejected` (`stream_open f0_method="rmvpe"` must be an error mentioning pm), `test_rvc_stream_adapter.py` (stream forces `f0_method` pm). These expectations must stay, so RMVPE for streams uses a NEW field (`f0_algo`), never `f0_method`. |
| E9 | Other items | Whole path writes the model rate (40 kHz) and the bot's `postprocess.py` resamples with soxr (high quality); stream path uses scipy `resample_poly` 6/5. Hiss is checked by metric (section 4), not assumed. Qwen CustomVoice speakers (config `spk_id`): serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan; plausible English male alternatives: aiden (and eric/dylan, which may carry dialect). fp16 HuBERT/synth (`Config.is_half=True`) is a minor unproven lever. |

## 2. Ranked levers

| Rank | Lever | Targets | Cost | Notes |
|------|-------|---------|------|-------|
| 1 | Fix index path + `index_rate` 0.3-0.75, raise `nprobe` (Obama 1 -> 16+) | timbre | low | Cache the index and `reconstruct_n` rows per model (today `pipeline()` re-reads per call). Guard zero scores. |
| 2 | RMVPE f0 + post-processing (median filter, octave-jump repair, clamp 50..500 Hz speech range, short-gap fill) | robotic, wobble | med | New weights, one download. Works per window for streams. |
| 3 | Register match: fixed per-profile shift and optional per-utterance auto shift toward a target median | off-key, timbre | low | Obama about -2..-5 st for typical Ryan, Goku 0..+2. Auto shift = round(12*log2(target/median_src)), clamped, strength param. Streams use the fixed per-profile value only. |
| 4 | `protect` 0.33-0.5 and `rms_mix_rate` | metallic consonants/breaths, level pumping | med | Implement in `vc_infer_pipeline.py`; `rms_mix_rate` whole-path only (breaks block continuity in streams). `protect` works in streams. |
| 5 | Alternative Qwen source speaker (aiden / eric / dylan) | timbre, less shift | med | Needs a generation script run in the qwen venv; compare at best settings. |
| 6 | Stream params (context 2000 ms) | stream-only artefacts | low | Re-measure after levers 1-4 using the join metric. |
| 7 | fp32 (`is_half` False) | slight metallic haze | low | Env switch `RVC_WORKER_FP32=1`, default unchanged. |
| 8 | HuBERT version/layer | none | none | Verified OK (E7); regression test only. |
| 9 | Output stage / hiss | hiss | low | Measure only; fix only if the HF metric shows a problem. |

## 3. Interfaces (the contract the tasks code against)

### 3.1 `legacy/rvc_f0.py` (Q1) — import-safe without torch

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class F0PostConfig:
    median_width: int = 0          # odd int >= 3 applies a median filter over voiced runs; 0 = off
    octave_repair: bool = False    # fold isolated +-12 st (and +-19 st) jumps back to the local median
    clamp_min_hz: float | None = None   # voiced frames below -> octave-shifted into range if possible else 0
    clamp_max_hz: float | None = None
    fill_gap_frames: int = 0       # fill unvoiced gaps <= N frames between voiced frames (interpolate)

RMVPE_DEFAULT_PATH: Path   # /home/cbash23/.cache/listen-speed/rvc-quality/weights/rmvpe.pt (env RVC_RMVPE_PATH overrides)

def rmvpe_available() -> bool
def extract_f0(x16: np.ndarray, p_len: int, method: str, *, f0_min: float = 50.0,
               f0_max: float = 1100.0, device: str | None = None,
               post: F0PostConfig | None = None) -> np.ndarray
    # x16: float32 mono, 16 kHz. Returns float32 [p_len] Hz, 0 = unvoiced, 10 ms hop
    # (hop 160 samples), padded / cropped to p_len exactly. method in
    # {"rmvpe","pm","dio","harvest","crepe"}. "rmvpe" raises RuntimeError("rmvpe weights not found: <path>") if absent.
    # Never prints. Model is loaded once and cached per (path, device).
def postprocess_f0(f0: np.ndarray, post: F0PostConfig, f0_min: float, f0_max: float) -> np.ndarray
def median_f0(f0: np.ndarray) -> float            # median of voiced (>0) frames, 0.0 if none
def auto_pitch_shift(source_median_hz: float, target_median_hz: float, *, max_abs: int = 12,
                     strength: float = 1.0) -> int   # round(strength*12*log2(target/src)) clamped; 0 if src <= 0
```

### 3.2 Worker wire protocol additions (Q2); every field optional, absent = today's behaviour

`convert` request: `use_index: bool` (false), `index_rate: float 0..1` (null = model yaml value, only used when `use_index`), `index_nprobe: int 1..256` (null = index own), `protect: float 0..0.5` (null/0.5 = off), `rms_mix_rate: float 0..1` (null/1.0 = off), `f0_method` additionally accepts `"rmvpe"`, `f0_post: {median_width:int, octave_repair:bool, clamp_min_hz:float, clamp_max_hz:float, fill_gap_frames:int}`, `auto_pitch: {target_median_hz:float, max_abs:int=12, strength:float=1.0}` (when present the applied shift = `pitch_shift` + `auto_pitch_shift(median_f0(source), target)`, source median measured with the requested f0 method).
`convert` reply adds: `index_used: bool`, `index_path: str|null`, `pitch_shift_applied: int`, `source_f0_median_hz: float|null`, `f0_method`.
`stream_open` request: `f0_method` stays pm-only (E8). New fields: `f0_algo: "pm"|"rmvpe"` (default pm), `use_index`, `index_rate`, `index_nprobe`, `protect`, `f0_post`. Reply adds `f0_algo`, `index_used`, `protect`; `f0_method` still `"pm"`.
`health` adds `rmvpe_available: bool`, `faiss: bool`.

Engine contract (Q2 implements, Q4 calls): in `legacy/rvc_worker.py`, module function

```python
def render_file(worker, input_wav: str, output_wav: str, **convert_fields) -> dict   # = worker.convert(...) with the fields above
```
and for streams `rvc_stream.StreamParams(f0_algo=..., use_index=..., index_rate=..., protect=..., f0_post=..., pitch_shift=...)` + `rvc_stream.open_engine(vc, vc_args, params)` (existing). `Worker.convert(input_wav_path, output_wav_path, pitch_shift=None, f0_method=None, **new_fields)` keeps its first four positional arguments.

### 3.3 Bot-side config (Q3), all default to today

`speech.voice_conversion`: `use_index: bool=False`, `index_rate: float|None=None`, `index_nprobe: int|None=None`, `protect: float|None=None`, `rms_mix_rate: float|None=None`, `pitch_shift_offset: int=0` (added to the profile's `pitch_shift`), `auto_pitch: bool=False`, `auto_pitch_strength: float=1.0`, `auto_pitch_max: int=12`, `target_f0_median_hz: dict[str,float]` (per voice-profile model name, e.g. `{Obama: 108, Goku: 145}`), `f0_post_median: int=0`, `f0_post_octave_repair: bool=False`, `f0_post_clamp: tuple[float,float]|None=None`, `stream_f0_algo: "pm"|"rmvpe"="pm"`; `f0_method_override` pattern gains `rmvpe`. Streams send `f0_algo`, `use_index`, `index_rate`, `protect`, `f0_post`, `pitch_shift` (profile shift + offset; no auto). `config/app.example.yaml` documents the keys (commented out).

## 4. Metrics (objective proxies; none replaces listening)

All computed by `tools/rvc_quality_lab.py` on the rendered wav resampled to 16 kHz, using an f0 tracker independent of the one that drove synthesis where possible (pyworld harvest on the OUTPUT; the reference f0 is harvest on the SOURCE times `2^(shift_applied/12)`).

| Metric | Definition | Better |
|--------|------------|--------|
| `f0_octave_err_rate` | fraction of frames voiced in both whose output f0 deviates from the shifted source f0 by >= 8 st AND is within 1.5 st of +-12/+-24 | lower |
| `f0_jump_rate` | frame-to-frame voiced jumps > 3 st per second of voiced speech (wobble/glitch) | lower |
| `f0_smooth_cents` | std of the second difference of log2 f0 inside voiced runs, in cents | lower |
| `f0_track_corr` | correlation of output f0 with shifted-source f0 on commonly voiced frames | higher |
| `f0_median_hz`, `register_err_st` | output voiced median, and semitone distance to the target register (Obama 108 Hz, Goku 145 Hz) | err lower |
| `voiced_frac_ratio` | output voiced fraction / source voiced fraction (dropout detector) | near 1 |
| `retrieval_dist` | mean L2 distance of the output's HuBERT features (the model's own layer: v1 L9+final_proj, v2 L12) to their 8 nearest neighbours in the model's faiss index (nprobe 32 for the measurement); `retrieval_dist_src` is the same for the Ryan source as control | lower = more like target. Caveat: index-blended renders score better by construction, so compare it across settings with the same index_rate, and read the `src` control |
| `mel_dist_ref_db` | log-mel L1 (dB, 80 mels, 16 kHz, time-aligned by cross-correlation) to the reference render `ref_harvest_idx05` (harvest + index 0.5, shift = best-known) | informational (distance from a known-good recipe) |
| `hf_hiss_ratio` | energy above 12 kHz / total energy in unvoiced frames (for 48 kHz output; 40 kHz models are band-limited at 20 kHz) and spectral flatness of unvoiced frames | lower |
| `rtf`, `peak`, `clip_frac`, `nan` | render speed and sanity | rtf lower, no nan/clip |
| `join_max` (stream variants only) | ratio of the max abs first difference at SOLA joins to the median of the whole-render, as in `legacy/rvc_stream_eval.py::_jump_ratio` | lower (<= ~2 as in design doc) |
| `stream_vs_whole_mel_db` | log-mel L1 between a stream render and the whole render of the same settings | lower |

Flagging: per sentence-set mean; a variant is "best by metric" when it ranks best on `f0_octave_err_rate`, `f0_jump_rate`, `retrieval_dist` (excluding variants whose `voiced_frac_ratio` is < 0.85 or `hf_hiss_ratio` is the worst quartile); the lab marks the lowest mean rank (rank-average of the three) as `BEST_OVERALL` plus the best per individual metric. These are proxies; the owner's ears decide.

## 5. Experiment matrix

Source: the six Ryan sentences in `/home/cbash23/.cache/listen-speed/qwen-fast/samples/fast_nonstream_s{0..5}.wav` (24 kHz; s0 and s5 are questions, s3 is an exclamation). Extra-speaker sources are generated once to `.../rvc-quality/sources/<speaker>/s{i}.wav` with the same sentences (qwen venv, `HF_HUB_OFFLINE=1`). Primary model: **Obama** (variant obama, v2, target register ~108 Hz). Secondary: **Goku** (v1, ~145 Hz). Staged, one factor at a time, each stage keeps the winner of the previous by the section-4 ranking (the lab does not auto-chain; Q4 provides `--stage` presets with the defaults below, and the owner/reviewer re-runs with the winner).

| Stage | Variant id | Settings (all else baseline) |
|-------|-----------|------------------------------|
| 0 | `v00_baseline` | pm, shift 0, no index (today) |
| 0 | `ref_harvest_idx05` | harvest, index 0.5, nprobe 16, shift = auto (reference for mel distance) |
| A | `a1_pm_idx05` | pm, index 0.5 (isolates the index) |
| A | `a2_rmvpe` | rmvpe, no index |
| A | `a3_rmvpe_post` | rmvpe + median 5 + octave repair + clamp 50-500 + gap fill 2 |
| A | `a4_rmvpe_post_idx05` | a3 + index 0.5 |
| A | `a5_harvest_idx05` | harvest + index 0.5 (does RMVPE beat harvest?) |
| B | `b1_shift_fixed_m3` .. `b3_shift_fixed_m5` (Goku: 0, +2) | a4 with pitch_shift -3, -5 (Obama) |
| B | `b4_autoshift` / `b5_autoshift_half` | a4 with auto_pitch target 108 Hz (Obama) / 145 (Goku), strength 1.0 and 0.5 |
| C | `c1_idx03` `c2_idx075` | a4 best shift, index_rate 0.3 / 0.75 |
| C | `c3_nprobe1` `c4_nprobe32` | index nprobe 1 (today's default for Obama) vs 32 |
| C | `c5_protect033` `c6_protect05` | protect 0.33 / 0.5 (0.5 = off) |
| C | `c7_rms025` `c8_rms10` | rms_mix_rate 0.25 / 1.0 |
| C | `c9_fp32` | RVC_WORKER_FP32=1 |
| D | `d1_aiden` `d2_eric` `d3_dylan` | best settings, source speaker swapped (shift re-derived by auto) |
| E | `e1_stream_pm_ctx1000` `e2_stream_pm_ctx2000` `e3_stream_rmvpe_ctx1000` `e4_stream_rmvpe_ctx2000` | best settings through `rvc_stream` (index + protect, no rms_mix) |

Goku runs the same matrix reduced to stages 0, A (a1,a4,a5), B (fixed 0/+2, auto), C (best index_rate, protect), E (rmvpe ctx2000). Full run budget: about 35 variants x 6 clips x ~1-3 s, all under `flock /home/cbash23/.cache/listen-speed/gpu.lock`.

Output: `/home/cbash23/.cache/listen-speed/rvc-quality/samples/<model>/<variant>/s{0..5}.wav`, `.../rvc-quality/index.md` (table per variant: settings, key metrics, absolute wav paths, BEST flags), `.../rvc-quality/results.json`.

## 6. Task split (disjoint file ownership)

| Task | Owns | Depends on |
|------|------|-----------|
| Q1 | `legacy/rvc_f0.py`, `tests/unit/test_rvc_f0.py`, `legacy/rvc_f0_weights.md` (vetting record) | none |
| Q2 | `legacy/rvc_worker.py`, `legacy/rvc_stream.py`, `rvc_cli/vc_infer_pipeline.py`, `rvc_cli/voice_converter.py`, `rvc_cli/models.py`, `legacy/README.md`, new tests `tests/unit/test_rvc_quality_params.py`, `tests/unit/test_rvc_worker_quality.py` | Q1 interface via lazy import with fallback |
| Q3 | `src/notamusicbot/config.py`, `src/notamusicbot/domain/speech.py`, `src/notamusicbot/speech/pipeline.py`, `src/notamusicbot/speech/voice_conversion/rvc.py`, `src/notamusicbot/app.py` (one wiring spot), `config/app.example.yaml`, new `tests/unit/test_voice_quality_config.py` | Q2 wire names (section 3.2) |
| Q4 | `tools/rvc_quality_lab.py`, `tools/rvc_quality_sources.py`, `tests/unit/test_rvc_quality_lab_metrics.py` | Q1+Q2 interfaces; reviewer may finish |

Cross-cutting rules: no commit/push; bot stopped; heavy GPU under flock; existing tests untouched; ruff check + format touched Python (legacy/ keeps its own style); `rvc_cli` is a submodule with an already dirty tree, so touch only the named functions and never reformat the file.

## 7. Risks

| Risk | Mitigation |
|------|-----------|
| RMVPE weights are a pickle (code-exec risk on load) | Q1 inspects with `pickletools` without executing, allows only torch rebuild + OrderedDict globals, loads with `torch.load(weights_only=True)`; if anything else appears, do not load, report. Record sha256/size. Runtime never touches the network. |
| Homemade RMVPE net mismatches the checkpoint | Test: strict `load_state_dict` (no missing/unexpected keys); f0 corr >= 0.9 vs harvest on the six clips; wrong net => keys fail loudly. |
| Index blending changes timbre of BOTH paths | Off by default; enable per config; owner listens before turning on. |
| Metrics reward index-blending by construction | Report `retrieval_dist_src` control; compare same-index settings; owner listens. |
| Per-utterance auto shift moves sentences to different pitches | Cap +-12 (Q3 default may cap lower), strength param, compare against fixed shift in stage B; streams stay fixed. |
| Streaming: RMVPE/index latency, `rms_mix_rate` breaks continuity | rms_mix is whole-path only; index search on 2 s windows is ms; measure `rtf`, `join_max`. |
| RMVPE needs min window length | Q1 pads internally (reflect) to a multiple of the model hop and crops. |
| Existing tests pin pm-only streams (E8) | `f0_algo` field; `f0_method` unchanged. |
| Zero retrieval score gives inf weight | eps clamp in new index code path only. |
| Submodule edits are not part of the main repo diff | Report touched files; owner decides about submodule commit. |
