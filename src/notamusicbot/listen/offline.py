"""Local-only guard for the listen system (plan section 13).

Listen mode makes no API calls to any AI or speech service. Model weights come
from the local HF cache; these switches stop huggingface_hub / transformers
from reaching the network even if a loader forgets ``local_files_only=True``.
No discord.py, no I/O beyond the environment mapping passed in.

The glue (task P9) must call :func:`enforce_offline_models` before any listen
model (ASR, VAD) loads, and may call :func:`assert_offline` as a cheap check.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def enforce_offline_models(environ: MutableMapping[str, str] = os.environ) -> dict[str, str]:
    """Force the offline variables to "1"; return ``{name: new_value}`` of changes.

    Not ``setdefault``: a stray ``HF_HUB_OFFLINE=0`` must be overridden. A
    variable already "1" is left alone and omitted from the result.
    """
    changed: dict[str, str] = {}
    for name, value in OFFLINE_ENV.items():
        if environ.get(name) != value:
            environ[name] = value
            changed[name] = value
    return changed


def assert_offline(environ: MutableMapping[str, str] = os.environ) -> None:
    """Raise RuntimeError unless ``HF_HUB_OFFLINE`` is "1"."""
    if environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError(
            "listen models must load offline: call enforce_offline_models() first "
            "(HF_HUB_OFFLINE != '1')"
        )
