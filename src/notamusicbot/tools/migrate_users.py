"""Migrate legacy ``input/users.json`` (dict- or list-shaped) to ``config/users.yaml``.

Legacy model names are resolved **explicitly** to new voice-profile ids:

* the legacy name is uppercased and matched case-insensitively against the
  available voice-profile ids (``OBAMA`` → ``obama``) and, failing that,
  against the RVC model directory each profile points at
  (``GOKU`` → ``goku`` → model dir ``Goku``);
* a legacy name with no corresponding profile is a hard error: the migration
  exits non-zero with a clear message. It never silently defaults to Goku.

Alias note (verified against ``rvc_cli/models.py``): legacy model names are
keyed by the *yaml stem* uppercased inside each ``rvc_cli/models/*`` directory
(``Model.model_name = yaml_file.stem``). There is **no** separate
``obama_trans`` directory and no separate ``obama_trans`` weights: the
``OBAMA_TRANS`` variant is defined by ``rvc_cli/models/Obama/obama_trans.yaml``
and shares ``rvc_cli/models/Obama/obama.pth`` + ``obama.index`` with
``OBAMA``. The dedicated ``obama_trans`` profile therefore points at the same
model directory (``model: Obama``) and user entries resolved through such an
alias carry ``voice_profile_source: legacy_alias`` so the conflation is
explicit and auditable.

The source file is never modified; a timestamped backup of the previous
destination is written next to it before overwriting.

Usage:
    python -m notamusicbot.tools.migrate_users [--source input/users.json] \
        [--dest config/users.yaml] [--voices-dir config/voices]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from notamusicbot.domain.users import load_users_json, parse_users
from notamusicbot.domain.voice_profiles import load_voice_profiles

logger = logging.getLogger(__name__)


class MigrationError(RuntimeError):
    """Raised when a legacy user references a model with no voice profile."""


def resolve_legacy_profile(
    legacy_model_name: str,
    profiles: dict[str, Any],
) -> str:
    """Resolve a legacy model name to a voice-profile id, or raise.

    Resolution order:

    1. case-insensitive match against profile ids (``OBAMA`` → ``obama``,
       ``OBAMA_TRANS`` → ``obama_trans``);
    2. case-insensitive match against the RVC model directory each profile's
       ``voice_conversion.model`` points at (``GOKU`` → ``goku``).

    Anything else raises :class:`MigrationError` — never a silent Goku default.
    """
    key = legacy_model_name.strip().upper()
    for profile_id in profiles:
        if profile_id.upper() == key:
            return profile_id
    for profile_id, profile in profiles.items():
        model = getattr(getattr(profile, "voice_conversion", None), "model", None)
        if model and str(model).upper() == key:
            return profile_id
    available = ", ".join(sorted(profiles)) or "<none>"
    msg = (
        f"Legacy model {legacy_model_name!r} has no corresponding voice profile. "
        f"Available profiles: {available}. "
        "Create the missing profile in config/voices/ and re-run the migration."
    )
    raise MigrationError(msg)


def migrate_users_to_profiles(
    source: Path,
    dest: Path,
    voices_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Migrate ``source`` users JSON to ``dest`` YAML with explicit profiles.

    Returns the written entries keyed by user id. Raises :class:`MigrationError`
    (before touching ``dest``) if any referenced legacy model cannot be resolved.
    """
    raw_users = load_users_json(source)
    if not raw_users:
        msg = f"No valid user entries found in {source}"
        raise ValueError(msg)
    profiles = load_voice_profiles(voices_dir)
    if not profiles:
        msg = f"No voice profiles found in {voices_dir}"
        raise MigrationError(msg)

    entries: dict[str, dict[str, Any]] = {}
    for user_id, user in raw_users.items():
        legacy = user.legacy_model_name
        if not legacy:
            msg = f"User {user_id} has no legacy model_name; cannot resolve a voice profile"
            raise MigrationError(msg)
        profile_id = resolve_legacy_profile(legacy, profiles)
        entry: dict[str, Any] = {
            "screen_name": user.screen_name,
            "real_name": user.real_name,
            "background": user.background,
            "voice_profile": profile_id,
            "legacy_model_name": legacy,
        }
        # Alias marker: mark entries whose profile does not own its own RVC
        # weights (profile id != its voice_conversion model directory). This is
        # the OBAMA_TRANS case: the obama_trans profile shares the Obama
        # weights (models/Obama/obama.pth), so the sharing is recorded instead
        # of being silently conflated.
        vc = profiles[profile_id].voice_conversion
        vc_model = getattr(vc, "model", None)
        if vc_model and str(vc_model).lower() != profile_id.lower():
            entry["voice_profile_source"] = "legacy_alias"
        # Preserve the YAML config variant (round-3 P0): a legacy name that
        # resolved to a profile pointing at a NON-default yaml of its model
        # directory (OBAMA_TRANS -> obama_trans.yaml in models/Obama/) must
        # carry the variant, otherwise the worker would load whichever yaml
        # it happens to find first.
        vc_variant = getattr(vc, "variant", None)
        if vc_variant:
            entry["variant"] = vc_variant
        entries[user_id] = entry

    payload = {"users": {uid: entries[uid] for uid in sorted(entries)}}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    # Round-trip validation: the written file must re-parse with the same
    # number of users.
    reread = parse_users((yaml.safe_load(dest.read_text(encoding="utf-8")) or {}).get("users", {}))
    if len(reread) != len(entries):
        msg = f"Migration round-trip mismatch: wrote {len(entries)}, read back {len(reread)}"
        raise ValueError(msg)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m notamusicbot.tools.migrate_users",
        description="Migrate legacy users.json to config/users.yaml.",
    )
    parser.add_argument("--source", default="input/users.json")
    parser.add_argument("--dest", default="config/users.yaml")
    parser.add_argument("--voices-dir", default="config/voices")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    source, dest, voices_dir = Path(args.source), Path(args.dest), Path(args.voices_dir)
    if not source.is_file():
        logger.error("Source file not found: %s", source)
        return 2
    if dest.is_file():
        backup = dest.with_name(f"{dest.stem}.backup-{time.strftime('%Y%m%d-%H%M%S')}.yaml")
        shutil.copyfile(dest, backup)
        logger.info("Backed up existing destination to %s", backup)
    try:
        entries = migrate_users_to_profiles(source, dest, voices_dir)
    except MigrationError as exc:
        logger.error("Migration failed: %s", exc)
        return 1
    counts: dict[str, int] = {}
    for entry in entries.values():
        counts[entry["voice_profile"]] = counts.get(entry["voice_profile"], 0) + 1
    logger.info("Migrated %d users to %s: %s", len(entries), dest, counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
