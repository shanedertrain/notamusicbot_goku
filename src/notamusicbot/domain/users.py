"""Typed user profile representation and lookup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """A Discord user's association with a voice/persona profile."""

    user_id: str
    screen_name: str = ""
    real_name: str = ""
    background: str = ""
    voice_profile: str = Field(default="goku", alias="voice_profile")
    legacy_model_name: str | None = Field(default=None, alias="legacy_model_name")

    model_config = {"populate_by_name": True}


class UserRegistry:
    """Read-only typed view over user profiles."""

    def __init__(self, users: dict[str, UserProfile]) -> None:
        self._users = users

    def lookup(self, user_id: str) -> UserProfile | None:
        return self._users.get(str(user_id))

    def voice_profile_id(self, user_id: str) -> str | None:
        user = self.lookup(user_id)
        return user.voice_profile if user else None

    def __len__(self) -> int:
        return len(self._users)


def _coerce_entry(user_id: str, data: dict[str, Any]) -> UserProfile:
    legacy = data.get("model_name") or data.get("legacy_model_name")
    return UserProfile(
        user_id=str(user_id),
        screen_name=data.get("screen_name", ""),
        real_name=data.get("real_name", ""),
        background=data.get("background", ""),
        voice_profile=data.get("voice_profile") or "goku",
        legacy_model_name=legacy,
    )


def parse_users(raw: Any) -> dict[str, UserProfile]:
    """Parse old user data in either supported shape.

    Accepts a dict keyed by user ID or a list of user objects (each with an
    ``id``/``user_id`` key). Returns users keyed by string user ID.
    """
    users: dict[str, UserProfile] = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            uid = entry.get("id") or entry.get("user_id")
            if uid is None:
                continue
            items.append((str(uid), entry))
    else:
        return users
    for uid, entry in items:
        if not isinstance(entry, dict):
            continue
        users[str(uid)] = _coerce_entry(str(uid), entry)
    return users


def load_users_yaml(path: Path) -> dict[str, UserProfile]:
    """Load users from the migrated YAML format (``users: {...}`` mapping)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return parse_users(data.get("users", {}))


def load_users_json(path: Path) -> dict[str, UserProfile]:
    """Load users from a legacy JSON file (dict- or list-shaped)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return parse_users(raw)


def write_users_yaml(users: dict[str, UserProfile], path: Path) -> None:
    """Write users in the canonical migrated YAML format."""
    payload = {
        "users": {
            uid: {
                "screen_name": u.screen_name,
                "real_name": u.real_name,
                "background": u.background,
                "voice_profile": u.voice_profile,
                "legacy_model_name": u.legacy_model_name,
            }
            for uid, u in sorted(users.items())
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def migrate_users_file(source: Path, dest: Path) -> dict[str, UserProfile]:
    """Migrate a legacy users file to the new YAML format.

    Validation round-trips: the written file is re-read and re-parsed before
    the migration is considered successful. The source file is never modified.
    """
    raw = json.loads(source.read_text(encoding="utf-8"))
    users = parse_users(raw)
    if not users:
        msg = f"No valid user entries found in {source}"
        raise ValueError(msg)
    write_users_yaml(users, dest)
    reread = load_users_yaml(dest)
    if len(reread) != len(users):
        msg = f"Migration round-trip mismatch: wrote {len(users)}, read back {len(reread)}"
        raise ValueError(msg)
    return reread
