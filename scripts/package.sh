#!/usr/bin/env bash
# Build a review/delivery package using an ALLOWLIST (never zip the tree).
# Nothing is included unless explicitly listed. Refuses to run if any
# credential-bearing file matches the allowlist patterns.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
OUT="${1:-/tmp/notamusicbot_review_package.zip}"

# --- allowlist -------------------------------------------------------------
# Directories/files that are safe to ship: source, tests, config examples,
# docs. Never: .env*, weights (*.pt/*.pth/*.index), caches, var/, .git.
ALLOW_ARGS=(
  src tests legacy scripts docs
  README.md IMPLEMENTATION_REPORT.md TEST_RESULTS.md
  BENCHMARK_RESULTS.md DEPENDENCY_NOTES.md REVIEW_NOTES.md REWORK_HANDOFF.md
  pyproject.toml .gitignore .env.example LEGACY_VENV.md
)
# config: examples and voice profiles only — NEVER the live users database
# (it contains real Discord user profile metadata).
ALLOW_ARGS+=(
  config/app.example.yaml
  config/users.example.yaml
  config/voices
)
# legacy_flat is included WITHOUT its own .env-like files (none exist, but
# the secret-scan below is the real gate).
ALLOW_ARGS+=(legacy_flat)

# --- secret gate -----------------------------------------------------------
echo "==> secret scan (patterns over allowlisted files only)"
if grep -rInE \
  '^\s*(DISCORD_TOKEN|NEWS_API_KEY|REDDIT_CLIENT_SECRET|SPOTIFY_CLIENT_SECRET|PLAY_HT_API_KEY|OPENAI_API_KEY|GEMINI_API_KEY|GEMENI_API_KEY)\s*=\s*[^$\s{][^ ]{8,}' \
  "${ALLOW_ARGS[@]}" 2>/dev/null | grep -vE '=($|<|${|\{\{|example|your[-_])'; then
  echo "SECRET SCAN FAILED: credential-like values found above." >&2
  exit 1
fi
echo "secret scan: clean"

# --- build -----------------------------------------------------------------
rm -f "$OUT"
zip -qr "$OUT" "${ALLOW_ARGS[@]}" \
  -x "*__pycache__*" "*.pyc" "*.pt" "*.pth" "*.index" \
     "*.egg-info*" ".ruff_cache/*" ".pytest_cache/*"
echo "==> packaged: $OUT"
unzip -l "$OUT" | tail -2

# --- verification: no .env-like files inside -------------------------------
if unzip -l "$OUT" | grep -qE "\.env$|\.env\.[^e]"; then
  echo "PACKAGING FAILED: .env-like file inside package." >&2
  exit 1
fi
echo "package verified: no credential-bearing files"
