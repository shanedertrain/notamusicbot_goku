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

# Review Round-3 P2: raw benchmark/evidence data backing the review claims.
# Included explicitly (never a blanket pattern): results CSVs, metadata JSON
# and evidence JSON/CSV. Audio WAVs and model weights stay excluded.
EVIDENCE_FILES=()
while IFS= read -r f; do EVIDENCE_FILES+=("$f"); done < <(
  find benchmark evidence -type f \
    \( -name 'results.csv' -o -name 'metadata.json' \
       -o -path 'evidence/*' -name '*.json' -o -path 'evidence/*' -name '*.csv' \) \
    2>/dev/null | sort
)
if [ "${#EVIDENCE_FILES[@]}" -gt 0 ]; then
  echo "==> including raw evidence: ${#EVIDENCE_FILES[@]} file(s)"
  ALLOW_ARGS+=("${EVIDENCE_FILES[@]}")
else
  echo "==> no raw evidence files found (benchmark/, evidence/) — skipping"
fi

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
# Review Round-3 P2: explicit logic, no character-class tricks — exactly one
# file named ".env.example" is allowed; EVERY other ".env*" path is rejected.
env_bad=0
while IFS= read -r entry; do
  base="$(basename "$entry")"
  if [ "$base" = ".env.example" ]; then
    continue
  fi
  case "$base" in
    .env*) echo "PACKAGING FAILED: .env-like file inside package: $entry" >&2; env_bad=1 ;;
  esac
done < <(unzip -Z1 "$OUT" | grep -E '(^|/)\.env' )
if [ "$env_bad" -ne 0 ]; then
  exit 1
fi
echo "package verified: no credential-bearing files"
