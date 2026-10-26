# RMVPE pitch model: provenance and vetting

| Item | Value |
|---|---|
| URL | https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt |
| Downloaded | 2026-10-26, `curl -L`, into empty dir `/home/cbash23/.cache/listen-speed/rvc-quality/weights_dl/` |
| Size | 181184272 bytes |
| sha256 | `6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193` |
| Final path | `/home/cbash23/.cache/listen-speed/rvc-quality/weights/rmvpe.pt` (`RMVPE_DEFAULT_PATH`; env `RVC_RMVPE_PATH` overrides) |

## Pickle vetting (not executed)

`python -I` + `zipfile` + `pickletools.genops` over `rmvpe2/data.pkl` (743 zip members).

GLOBAL targets (no STACK_GLOBAL, no other imports):
- `collections.OrderedDict`
- `torch.HalfStorage`
- `torch._utils._rebuild_tensor_v2`

Opcodes: PROTO 1, GLOBAL 3, REDUCE 1483, BINPERSID 741, SETITEMS 1, TUPLE/TUPLE1/TUPLE2/EMPTY_TUPLE,
MARK, NEWFALSE, BINPUT/LONG_BINPUT/BINGET, BININT/BININT1/BININT2, BINUNICODE, STOP.
No os / posix / builtins / subprocess / eval / exec.

## Load method

`torch.load(path, map_location="cpu", weights_only=True)` -> `OrderedDict`, 741 entries, all `torch.Tensor`
(float16). Keys: `unet.*`, `cnn.*`, `fc.0.gru.*`, `fc.1.*`. `legacy/rvc_f0.py` loads it with
`load_state_dict(..., strict=True)` after casting to float32 (no missing / unexpected keys).
Runtime is offline; the file is only read from disk.
