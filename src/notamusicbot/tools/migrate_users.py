"""Migrate a legacy ``input/users.json`` (dict- or list-shaped) to ``config/users.yaml``.

Legacy model names are resolved **explicitly** to voice-profile ids:

* the legacy name is uppercased and matched case-insensitively against the
  available voice-profile ids (``OBAMA`` → ``obama``);
* failing an exact match, the longest profile id that prefixes the legacy name
  wins (``OBAMA_TRANS`` → ``obama``, ``GOKU_RAW`` → ``goku``);
* a legacy name with no corresponding profile is a hard error: the migration
  exits non-zero with a clear message. It never silently defaults to Goku.

The source file is never modified; a timestamped backup of the previous
destination is written next to it before overwriting.

Usage:
    python -m notamusicbot.tools.migrate_users [--source input/users.json] \
        [--dest config/users.yaml] [--voices-dir config/voices]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from notamusicbot.domain.users import parse_users
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

    1. case-insensitive match against profile ids (``OBAMA`` → ``obama``);
    2. longest profile id that prefixes the legacy name
       (``OBAMA_TRANS`` → ``obama``, ``GOKU_RAW`` → ``goku``).

    Anything else raises :class:`MigrationError` — never a silent Goku default.
    """
    key = legacy_model_name.strip().upper()
    for profile_id in profiles:
        if profile_id.upper() == key:
            return profile_id
    prefixed = [profile_id for profile_id in profiles if key.startswith(profile_id.upper())]
    if prefixed:
        return max(prefixed, key=len)
    available = ", ".join(sorted(profiles)) or "<none>"
    msg = (
        f"Legacy model {legacy_model_name!r} has no corresponding voice profile. "
        f"Available profiles: {available}. "
        "Create the missing profile in config/voices/ and re-run the migration."
    )
    raise MigrationError(msg)


def _raw_users(raw: Any) -> list[tuple[str, dict[str, Any]]]:
    """Legacy JSON entries as ``(user_id, entry)``, in either supported shape."""
    items: list[tuple[str, dict[str, Any]]] = []
    if isinstance(raw, dict):
        for uid, entry in raw.items():
            if isinstance(entry, dict):
                items.append((str(uid), entry))
    elif isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            uid = entry.get("id") or entry.get("user_id")
            if uid is not None:
                items.append((str(uid), entry))
    return items


def migrate_users_to_profiles(
    source: Path,
    dest: Path,
    voices_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Migrate ``source`` users JSON to ``dest`` YAML with explicit profiles.

    Returns the written entries keyed by user id. Raises :class:`MigrationError`
    (before touching ``dest``) if any referenced legacy model cannot be resolved.
    """
    raw = json.loads(source.read_text(encoding="utf-8"))
    entries: dict[str, dict[str, Any]] = {}
    profiles = load_voice_profiles(voices_dir)
    if not profiles:
        msg = f"No voice profiles found in {voices_dir}"
        raise MigrationError(msg)

    for user_id, user in _raw_users(raw):
        legacy = user.get("model_name") or user.get("legacy_model_name")
        if not legacy:
            msg = f"User {user_id} has no legacy model_name; cannot resolve a voice profile"
            raise MigrationError(msg)
        profile_id = resolve_legacy_profile(str(legacy), profiles)
        entries[user_id] = {
            "screen_name": user.get("screen_name", ""),
            "real_name": user.get("real_name", ""),
            "background": user.get("background", ""),
            "voice_profile": profile_id,
        }

    if not entries:
        msg = f"No valid user entries found in {source}"
        raise ValueError(msg)

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
