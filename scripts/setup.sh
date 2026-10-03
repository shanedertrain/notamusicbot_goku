#!/usr/bin/env bash
# notamusicbot — one-command setup for a fresh machine.
# Idempotent: safe to re-run. Detects what exists and only builds what's missing.
#
# Usage:
#   ./scripts/setup.sh              # full setup + non-GPU sanity checks
#   ./scripts/setup.sh --gpu-check  # additionally run Qwen TTS + RVC smoke tests
#   ./scripts/setup.sh --no-rvc     # skip the legacy Python 3.10 RVC environment
#
# PLATFORM REQUIREMENT (P2-21): the runtime targets **Linux / WSL only**.
# Windows is not natively supported — use WSL2 on Windows. The script and the
# application rely on POSIX paths (.venv-main/bin/python), Bash and ffmpeg.
#
# REQUIRED RVC ASSETS (not distributed with the repo; must be supplied locally):
#   - hubert_base.pt                 at the repository root (HuBERT feature
#                                    extractor; obtain from the RVC project
#                                    releases, e.g. fatetudio/retrieve or the
#                                    Retrieval-based-Voice-Conversion repo).
#   - rvc_cli/models/<Name>/*.pth    per-voice model weights (one directory per
#                                    voice, mixed-case directory names such as
#                                    models/Obama, models/Goku).
#   - rvc_cli/models/<Name>/*.index  matching FAISS index per voice.
#   - rvc_cli/models/<Name>/*.yaml   per-voice (or per-variant) config; the yaml
#                                    stem is the legacy model name (e.g.
#                                    obama_trans.yaml defines OBAMA_TRANS, which
#                                    shares the Obama weights).
# Without hubert_base.pt the legacy RVC worker cannot start; voice profiles can
# still use voice_conversion.backend: passthrough.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

GPU_CHECK=0; WITH_RVC=1
for arg in "$@"; do
  case "$arg" in
    --gpu-check) GPU_CHECK=1 ;;
    --no-rvc)    WITH_RVC=0 ;;
  esac
done

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  OK \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m WARN\033[0m %s\n' "$*"; }
# `set -e`-compatible: fail() terminates the script with a non-zero exit code
# (P1-15), so missing prerequisites actually abort setup instead of printing
# and continuing.
fail() { printf '\033[1;31m FAIL\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
log "1/7 System prerequisites"
for cmd in git curl ffmpeg; do
  command -v "$cmd" >/dev/null && ok "$cmd found ($(command -v $cmd))" || fail "$cmd missing — install it first (e.g. apt install ffmpeg build-essential)"
done

# Git submodules (legacy rvc_cli lives in a submodule) — required for RVC.
if [ -f .gitmodules ] && grep -q rvc_cli .gitmodules; then
  git submodule update --init --recursive && ok "rvc_cli submodule initialised" \
    || fail "git submodule update --init --recursive failed — the legacy RVC code could not be fetched"
else
  ok "no rvc_cli submodule configured (already vendored?)"
fi
if command -v espeak-ng >/dev/null || command -v espeak >/dev/null; then ok "espeak found"; else warn "espeak-ng not found (only needed for legacy TTS fallbacks): apt install espeak-ng"; fi

# GPU / CUDA
GPU_OK=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  ok "NVIDIA GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  GPU_OK=1
else
  warn "No NVIDIA GPU detected — TTS/RVC/LLM will not run on GPU; CPU inference will be very slow."
fi

# ---------------------------------------------------------------------------
log "2/7 uv + Python 3.12"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
uv python install 3.12

# ---------------------------------------------------------------------------
log "3/7 Main application environment (.venv-main, Python 3.12)"
[ -d .venv-main ] || uv venv --python 3.12 .venv-main
# Torch: default PyPI wheels bundle CUDA on Linux; done via pyproject/uv sync.
uv pip install -p .venv-main -e ".[discord,media,dev]"
uv pip install -p .venv-main "discord.py[voice]" yt-dlp yt-dlp-ejs soundfile soxr
ok "main venv ready"

# ---------------------------------------------------------------------------
log "4/7 Secrets (.env) + app config"
if [ -f .env ]; then
  ok ".env already present (left untouched)"
else
  cp .env.example .env
  warn "Created .env from .env.example — EDIT IT and fill in DISCORD_TOKEN etc. before running the bot."
fi

# Non-secret app config: scaffold config/app.yaml from the example only if
# missing (P0-3). load_app_settings() reads config/app.yaml by default.
if [ -f config/app.yaml ]; then
  ok "config/app.yaml already present (left untouched)"
else
  cp config/app.example.yaml config/app.yaml
  ok "Scaffolded config/app.yaml from config/app.example.yaml"
fi

# ---------------------------------------------------------------------------
log "5/7 Model pre-downloads (HF cache)"
PY=".venv-main/bin/python"
$PY - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",):
    print("downloading", repo, flush=True)
    snapshot_download(repo)
print("TTS model ready")
EOF
ok "Qwen3-TTS-0.6B-CustomVoice cached"
# Content LLM (~8 GB) — skipped if already cached
$PY - <<'EOF'
import os
from huggingface_hub import try_to_load_from_cache
try:
    from transformers.utils import cached_file
    cached_file("Qwen/Qwen3-4B-Instruct-2507", "config.json")
    print("LLM already cached")
except Exception:
    from huggingface_hub import snapshot_download
    print("downloading Qwen/Qwen3-4B-Instruct-2507 (~8 GB)...", flush=True)
    snapshot_download("Qwen/Qwen3-4B-Instruct-2507")
EOF
ok "content LLM cached"

# ---------------------------------------------------------------------------
# Review Round-3 P2: validate required configured RVC assets per enabled
# profile. Pure function over (voices_dir, models_root) so tests can run it
# directly against a temp-dir fixture (tests/unit/test_setup_rvc_assets.py).
# For every config/voices/*.yaml profile whose voice_conversion backend is
# "rvc", this checks that:
#   1. the model directory        rvc_cli/models/<model>/        exists
#   2. the profile's variant yaml <model-dir>/<profile-stem>.yaml exists
#   3. one *.pth (weights) and one *.index (FAISS index) exist in the dir
# Missing assets are reported on stdout; the return value is the failure count
# (non-zero => setup must fail).
validate_rvc_assets() {
  local voices_dir="$1" models_root="$2"
  local missing=0
  local profile id model mdir pth idx variant want
  for profile in "$voices_dir"/*.yaml; do
    [ -e "$profile" ] || continue
    grep -Eq '^[[:space:]]*backend:[[:space:]]*rvc([[:space:]]|$)' "$profile" || continue
    id=$(grep -E '^[[:space:]]*id:' "$profile" | head -1 | sed 's/.*id:[[:space:]]*//')
    model=$(awk '/^voice_conversion:/{f=1;next} f&&/^[[:space:]]*model:/{print $2; exit}' "$profile")
    # Review Round-4 P2: read the explicit voice_conversion.variant field
    # instead of inferring it from the profile filename. Profile ID and RVC
    # variant are allowed to differ (e.g. president_alt.yaml -> obama_trans).
    variant="$(python3 -c '
import sys
in_vc = False
with open(sys.argv[1], encoding="utf-8") as fh:
    for raw in fh:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if in_vc and not line[0].isspace():
            break  # left the voice_conversion block
        if not in_vc:
            if stripped == "voice_conversion:":
                in_vc = True
            continue
        if stripped.startswith("variant:"):
            val = stripped.split(":", 1)[1].strip().strip("\"").strip("'\''")
            if val:
                print(val)
            break
' "$profile")"
    if [ -z "$model" ]; then
      printf '  FAIL profile %s: voice_conversion.backend is rvc but no model is configured\n' "${id:-$profile}"
      missing=$((missing + 1))
      continue
    fi
    mdir="$models_root/$model"
    if [ ! -d "$mdir" ]; then
      # Legacy RVC resolves model names case-insensitively (profile "GOKU"
      # matches models/Goku); fall back to a case-insensitive directory match.
      local ci=""
      for d in "$models_root"/*/; do
        if [ "$(basename "$d")" ] && [ "${model,,}" = "$(basename "$d" | tr 'A-Z' 'a-z')" ]; then
          ci="$(basename "$d")"
          break
        fi
      done
      if [ -n "$ci" ]; then
        mdir="$models_root/$ci"
      else
        printf '  FAIL profile %s: model directory missing: %s\n' "$id" "$mdir"
        missing=$((missing + 1))
        continue
      fi
    fi
    if [ -n "$variant" ]; then
      # Explicit variant: validate exactly <variant>.yaml in the model dir.
      want="$variant"
    else
      # No explicit variant: validate the directory-default yaml — the yaml
      # whose stem matches the model dir name, else the single yaml present.
      want="$(ls "$mdir" | grep -iF "$(basename "$mdir").yaml" | head -1)"
      if [ -z "$want" ]; then
        # Exactly one yaml in the dir acts as the directory default.
        local y n=0 cand=""
        for y in "$mdir"/*.yaml; do
          [ -e "$y" ] || continue
          n=$((n + 1))
          cand="$y"
        done
        if [ "$n" -eq 1 ]; then
          want="$(basename "$cand")"
        fi
      fi
    fi
    if [ -z "$want" ] || ! ls "$mdir" | grep -qiF "$want"; then
      printf '  FAIL profile %s: variant yaml missing in %s (expected %s.yaml)\n' "$id" "$mdir" "${variant:-<directory-default>}"
      missing=$((missing + 1))
    fi
    pth="$(ls "$mdir"/*.pth 2>/dev/null | head -1)"
    idx="$(ls "$mdir"/*.index 2>/dev/null | head -1)"
    if [ -z "$pth" ]; then
      printf '  FAIL profile %s: no .pth weights in %s\n' "$id" "$mdir"
      missing=$((missing + 1))
    fi
    if [ -z "$idx" ]; then
      printf '  FAIL profile %s: no .index FAISS file in %s\n' "$id" "$mdir"
      missing=$((missing + 1))
    fi
  done
  return "$missing"
}

# ---------------------------------------------------------------------------
log "6/7 Legacy RVC environment (Python 3.10, fairseq)"
if [ "$WITH_RVC" = "1" ]; then
  if [ -x .venv/bin/python ] && .venv/bin/python -c "import fairseq" 2>/dev/null; then
    ok "legacy .venv (py3.10 + fairseq) already present"
  else
    warn "Building legacy RVC venv (large: torch cu121 + fairseq build)..."
    uv venv --python 3.10 .venv
    uv pip install -p .venv "numpy==1.26.4" "cython<3" wheel "setuptools<81"
    uv pip install -p .venv fairseq==0.12.2
    uv pip install -p .venv "torch==2.5.0" "torchaudio==2.5.0" --index-url https://download.pytorch.org/whl/cu121
    uv pip install -p .venv setuptools librosa faiss-cpu soundfile praat-parselmouth resampy soxr pyworld torchcrepe audiostretchy pydub gTTS pyttsx3 pyyaml python-dotenv "cython<3"
    # ffmpeg-python shim: legacy rvc_cli imports it; system ffmpeg CLI is used.
    # customtkinter (rvc_cli GUI surface) is intentionally not installed — the
    # worker imports only the conversion path.
    uv pip install -p .venv ffmpeg-python || true
    ok "legacy RVC venv built"
  fi
  [ -f hubert_base.pt ] || fail "hubert_base.pt missing at repo root — required by RVC. See 'REQUIRED RVC ASSETS' in this script's header for where to obtain it."
  ls rvc_cli/models >/dev/null 2>&1 && ok "RVC voice models present ($(ls rvc_cli/models | wc -l) models)" || warn "no rvc_cli/models — target voices unavailable"
  # Review Round-3 P2: hard-fail on missing required assets for every enabled
  # rvc profile (model dir, variant yaml, weights, index). Previously this was
  # a warn, so RVC setup could "succeed" without usable target voices.
  echo "validating configured RVC voice-profile assets..."
  if RVC_MISSING=$(validate_rvc_assets config/voices rvc_cli/models); then
    ok "all enabled rvc profiles have complete model assets"
  else
    printf '%s\n' "$RVC_MISSING"
    fail "missing required RVC assets for enabled voice profile(s) — supply the model directory, variant yaml, .pth and .index files, or re-run with --no-rvc"
  fi
else
  warn "Skipping legacy RVC (--no-rvc); voice profiles must use voice_conversion.backend: passthrough"
fi

# ---------------------------------------------------------------------------
log "7/7 Sanity checks"
$PY -m pytest tests/unit -q && ok "unit tests PASS" || fail "unit tests FAILED"
$PY -c "import notamusicbot; print('package imports OK')"
$PY -c "import torch; print('CUDA available:', torch.cuda.is_available())"

if [ "$GPU_CHECK" = "1" ]; then
  log "GPU smoke tests (may take a few minutes on first run)"
  $PY -m pytest tests/integration -m "rvc and integration" -q && ok "RVC integration PASS" || warn "RVC integration FAILED/SKIPPED (see output)"
  $PY -m notamusicbot.tools.speech_test --profile goku --text "Hey, it's me, Goku!" --no-vc --output var/speech/setup_smoke.wav \
    && ok "Qwen TTS smoke PASS (var/speech/setup_smoke.wav)" || warn "Qwen TTS smoke FAILED"
fi

log "Setup complete. Next steps:"
echo "  1. Edit .env (DISCORD_TOKEN, DISCORD_GUILD, ...)"
echo "  2. Test speech locally:  .venv-main/bin/python -m notamusicbot.tools.speech_test --profile goku --text \"Hey, it's me, Goku!\""
echo "  3. Run the bot:          .venv-main/bin/python -m notamusicbot"
