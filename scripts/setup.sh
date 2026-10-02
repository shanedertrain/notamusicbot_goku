#!/usr/bin/env bash
# notamusicbot — one-command setup for a fresh machine.
# Idempotent: safe to re-run. Detects what exists and only builds what's missing.
#
# Usage:
#   ./scripts/setup.sh              # full setup + non-GPU sanity checks
#   ./scripts/setup.sh --gpu-check  # additionally run Qwen TTS + RVC smoke tests
#   ./scripts/setup.sh --no-rvc     # skip the legacy Python 3.10 RVC environment
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
fail() { printf '\033[1;31m FAIL\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
log "1/7 System prerequisites"
for cmd in git curl ffmpeg; do
  command -v "$cmd" >/dev/null && ok "$cmd found ($(command -v $cmd))" || fail "$cmd missing — install it first (e.g. apt install ffmpeg build-essential)"
done
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
uv pip install -p .venv-main -e ".[discord,media]"
uv pip install -p .venv-main "discord.py[voice]" yt-dlp yt-dlp-ejs soundfile soxr
ok "main venv ready"

# ---------------------------------------------------------------------------
log "4/7 Secrets (.env)"
if [ -f .env ]; then
  ok ".env already present (left untouched)"
else
  cp .env.example .env
  warn "Created .env from .env.example — EDIT IT and fill in DISCORD_TOKEN etc. before running the bot."
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
log "6/7 Legacy RVC environment (Python 3.10, fairseq)"
if [ "$WITH_RVC" = "1" ]; then
  if [ -x .venv/bin/python ] && .venv/bin/python -c "import fairseq" 2>/dev/null; then
    ok "legacy .venv (py3.10 + fairseq) already present"
  else
    warn "Building legacy RVC venv (large: torch cu121 + fairseq build)..."
    uv venv --python 3.10 .venv
    uv pip install -p .venv "numpy==1.26.4" "cython<3" wheel setuptools
    uv pip install -p .venv fairseq==0.12.2
    uv pip install -p .venv "torch==2.5.0" "torchaudio==2.5.0" --index-url https://download.pytorch.org/whl/cu121
    uv pip install -p .venv librosa faiss-cpu soundfile praat-parselmouth resampy soxr pyworld torchcrepe "cython<3"
    # ffmpeg-python shim: legacy rvc_cli imports it; system ffmpeg CLI is used.
    uv pip install -p .venv ffmpeg-python || true
    ok "legacy RVC venv built"
  fi
  [ -f hubert_base.pt ] || fail "hubert_base.pt missing at repo root — required by RVC"
  ls rvc_cli/models >/dev/null 2>&1 && ok "RVC voice models present ($(ls rvc_cli/models | wc -l) models)" || warn "no rvc_cli/models — target voices unavailable"
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
