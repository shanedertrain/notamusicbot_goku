"""No-Discord import smoke test (review P1-14).

The ordinary unit suite must run without discord.py installed. This test
spawns a subprocess with an import hook that raises on any ``discord`` import
and imports every non-I/O surface the unit tests exercise.
"""

from __future__ import annotations

import subprocess
import sys

NON_DISCORD_MODULES = [
    "notamusicbot.discord_bot.player",
    "notamusicbot.discord_bot.guild_state",
    "notamusicbot.discord_bot.commentary",
    "notamusicbot.discord_bot",  # package __init__ must stay lightweight
    "notamusicbot.media.youtube",
    "notamusicbot.media.downloader",
    "notamusicbot.domain.media",
    "notamusicbot.speech.pipeline",
    "notamusicbot.speech.postprocess",
    "notamusicbot.speech.artifacts",
    "notamusicbot.domain.listen",
    "notamusicbot.speech.asr",
    "notamusicbot.speech.asr.base",
    "notamusicbot.speech.vad",
    "notamusicbot.speech.vad.base",
    "notamusicbot.listen",  # package __init__ must stay lightweight
    "notamusicbot.speech.vad.energy",
    "notamusicbot.speech.vad.silero",  # silero-vad itself is imported lazily
    "notamusicbot.listen.ingress",
    "notamusicbot.listen.endpointer",
    "notamusicbot.listen.conversation",
    "notamusicbot.listen.reply",
    "notamusicbot.listen.metrics",
    "notamusicbot.listen.turn_taking",
    "notamusicbot.listen.offline",
    "notamusicbot.speech.asr._common",
    "notamusicbot.speech.asr.parakeet_onnx",  # onnx-asr itself is imported lazily
    "notamusicbot.speech.asr.whisper_hf",  # transformers itself is imported lazily
    "notamusicbot.speech.asr.factory",
    "notamusicbot.content.spark_chat",  # httpx, loopback only (plan section 17)
    "notamusicbot.content.fallback",
]

_HOOK = """
import sys

class DiscordBlocker:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name == "discord" or name.startswith("discord."):
            raise ImportError("discord.py import is forbidden in the unit suite")
        return None

sys.meta_path.insert(0, DiscordBlocker())
"""


def test_non_discord_surfaces_import_without_discord() -> None:
    code = "\n".join(
        [
            _HOOK,
            *[f"import {module}" for module in NON_DISCORD_MODULES],
            "print('OK')",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=".",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")


def test_lazy_package_getattr_exposes_bot_without_importing_discord() -> None:
    code = "\n".join(
        [
            _HOOK,
            "import notamusicbot.discord_bot as pkg",
            # The package stays lightweight: importing it never pulls discord.
            "assert callable(pkg.__getattr__)",
            "import sys",
            "assert 'discord' not in sys.modules",
            "print('OK')",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=".",
        check=False,
    )
    assert result.returncode == 0, result.stderr
