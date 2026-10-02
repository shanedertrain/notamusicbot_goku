"""Mocked-startup smoke test for the Phase E Discord wiring (no real token, no network).

Run: .venv-main/bin/python tools/smoke_discord_startup.py
"""

from __future__ import annotations

import asyncio
import sys


def main() -> int:
    from notamusicbot.app import Application
    from notamusicbot.config import AppSettings, SecretsSettings, load_app_settings
    from notamusicbot.discord_bot.bot import build_bot

    settings = load_app_settings()
    assert isinstance(settings, AppSettings)
    app = Application(settings, SecretsSettings(discord_token=None))
    bot = build_bot(app)
    from notamusicbot.discord_bot.commands import setup_commands

    setup_commands(bot)  # normally invoked from setup_hook after login
    names = sorted(cmd.name for cmd in bot.commands)
    expected = {"deleteallmessages", "join", "play", "play_spotify", "playnow", "skip", "stop"}
    missing = expected - set(names)
    if missing:
        print(f"MISSING COMMANDS: {sorted(missing)}")
        return 1
    print(f"prefix={bot.command_prefix!r} commands={names}")

    # Guild-scoped state sanity: registry is keyed per guild, never global.
    s1 = bot.guild_state(1)
    s2 = bot.guild_state(2)
    assert s1 is not s2 and s1.queue is not s2.queue

    # run_discord with no token must refuse cleanly (no connection attempt).
    app.run_discord()

    # setup_hook without the model: speech model loading is skipped here; only
    # verify command setup path is importable and player creation is guild-bound.
    player = bot.get_player(type("G", (), {"id": 1})())  # minimal guild stand-in
    assert player.state.guild_id == 1
    asyncio.run(asyncio.sleep(0))
    print("MOCKED STARTUP SMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
