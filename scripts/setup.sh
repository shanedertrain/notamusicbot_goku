#!/usr/bin/env bash
# notamusicbot — one-command setup for a fresh machine.
# Idempotent: safe to re-run. Detects what exists and only builds what's missing.
#
# Usage:
#   ./scripts/setup.sh              # full setup + non-GPU sanity checks
#   ./scripts/setup.sh --gpu-check  # additionally run Qwen TTS + LLM smoke tests
#
# PLATFORM REQUIREMENT (P2-21): the runtime targets **Linux / WSL only**.
# Windows is not natively supported — use WSL2 on Windows. The script and the
# application rely on POSIX paths (.venv-main/bin/python), Bash and ffmpeg.
#
# VOICES: every voice profile (config/voices/*.yaml) is a Qwen3-TTS Base
# zero-shot clone — a reference WAV plus its transcript. There is no per-voice
# model to download and no second runtime environment to build.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

GPU_CHECK=0
for arg in "$@"; do
  case "$arg" in
    --gpu-check) GPU_CHECK=1 ;;
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
log "1/6 System prerequisites"
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
  warn "No NVIDIA GPU detected — TTS/LLM will not run on GPU; CPU inference will be very slow."
fi

# ---------------------------------------------------------------------------
log "2/6 uv + Python 3.12"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
uv python install 3.12

# ---------------------------------------------------------------------------
log "3/6 Main application environment (.venv-main, Python 3.12)"
[ -d .venv-main ] || uv venv --python 3.12 .venv-main
# Torch: default PyPI wheels bundle CUDA on Linux; done via pyproject/uv sync.
uv pip install -p .venv-main -e ".[discord,media,dev]"
uv pip install -p .venv-main "discord.py[voice]" yt-dlp yt-dlp-ejs soundfile soxr
# Optional CUDA-graph accelerator for the clone TTS (the code falls back to
# stock qwen_tts when this is missing; --no-deps keeps its pins out of the env).
uv pip install -p .venv-main --no-deps "faster-qwen3-tts==0.3.1" \
  && ok "faster-qwen3-tts installed (optional clone accelerator)" \
  || warn "faster-qwen3-tts not installed; clone TTS will use the stock model"
ok "main venv ready"

# ---------------------------------------------------------------------------
log "4/6 Secrets (.env) + app config"
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
# Every configured voice needs its reference clip on disk.
missing_refs=0
for profile in config/voices/*.yaml; do
  [ -e "$profile" ] || continue
  ref=$(awk '/^[[:space:]]*reference_audio:/{print $2; exit}' "$profile")
  if [ -z "$ref" ] || [ ! -f "$ref" ]; then
    printf '  FAIL profile %s: reference audio missing (%s)\n' "$(basename "$profile")" "${ref:-<none>}"
    missing_refs=$((missing_refs + 1))
  fi
done
if [ "$missing_refs" -gt 0 ]; then
  fail "voice reference clips missing under config/voices/refs/ — supply a WAV per profile"
fi
ok "voice clone references present"

# ---------------------------------------------------------------------------
log "5/6 Model pre-downloads (HF cache)"
PY=".venv-main/bin/python"
$PY - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3-TTS-12Hz-0.6B-Base",):
    print("downloading", repo, flush=True)
    snapshot_download(repo)
print("TTS model ready")
EOF
ok "Qwen3-TTS-0.6B-Base cached"
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
log "6/6 Sanity checks"
$PY -m pytest tests/unit -q && ok "unit tests PASS" || fail "unit tests FAILED"
$PY -c "import notamusicbot; print('package imports OK')"
$PY -c "import torch; print('CUDA available:', torch.cuda.is_available())"

if [ "$GPU_CHECK" = "1" ]; then
  log "GPU smoke tests (may take a few minutes on first run)"
  $PY -m pytest tests/integration -m "gpu and integration" -q && ok "GPU integration PASS" || warn "GPU integration FAILED/SKIPPED (see output)"
  $PY -m notamusicbot.tools.speech_test --profile obama --text "Look, let me be clear. That last track was something else." --output var/speech/setup_smoke.wav \
    && ok "Qwen clone TTS smoke PASS (var/speech/setup_smoke.wav)" || warn "Qwen clone TTS smoke FAILED"
fi

log "Setup complete. Next steps:"
echo "  1. Edit .env (DISCORD_TOKEN, DISCORD_GUILD, ...)"
echo "  2. Test speech locally:  .venv-main/bin/python -m notamusicbot.tools.speech_test --profile obama --text \"Look, let me be clear.\""
echo "  3. Run the bot:          .venv-main/bin/python -m notamusicbot"
