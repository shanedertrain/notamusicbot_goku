"""``!glisten`` command surface and the voice-state hook (fakes only).

``listen_glue`` is never imported: the bot is a bare ``commands.Bot`` carrying a
fake registry and a fake session factory, so no model, voice or Discord login
is involved.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("discord", reason="discord.py is an optional runtime dependency")

import discord
from discord.ext import commands

from notamusicbot.config import AppSettings, ListenSettings
from notamusicbot.discord_bot.bot import NotAMusicBot
from notamusicbot.discord_bot.commands import LISTEN_ON_MESSAGE, setup_commands
from notamusicbot.discord_bot.guild_state import GuildStateRegistry

GUILD_ID = 7
CHANNEL_ID = 100
OTHER_CHANNEL_ID = 200
OPERATOR_ID = 1
USER_ID = 2


class FakeSession:
    def __init__(self, app, bot, guild, channel) -> None:
        self.channel = channel
        self.opted_in: set[int] = set()
        self.left: list[int] = []
        self.started = 0
        self.stops = 0
        self.fail_start = False
        self.events: list[str] = []

    @property
    def channel_id(self) -> int:
        return self.channel.id

    async def start(self) -> None:
        self.started += 1
        if self.fail_start:
            raise RuntimeError("boom: secret transcript text")

    async def stop(self) -> None:
        self.stops += 1

    def opt_in(self, user_id: int) -> None:
        self.opted_in.add(user_id)

    def opt_out(self, user_id: int) -> None:
        self.opted_in.discard(user_id)
        self.events.append(f"opt_out:{user_id}")

    def on_member_left(self, user_id: int) -> None:
        self.left.append(user_id)
        self.opt_out(user_id)

    def status(self) -> str:
        return f"listening in {self.channel_id}"


class FakeRegistry:
    def __init__(self) -> None:
        self.sessions: dict[int, FakeSession] = {}

    def get(self, guild_id: int):
        return self.sessions.get(guild_id)

    def add(self, session) -> None:
        gid = session.guild_id
        if gid in self.sessions:
            raise RuntimeError("exists")
        self.sessions[gid] = session

    def remove(self, guild_id: int) -> None:
        self.sessions.pop(guild_id, None)

    def active(self, guild_id: int) -> bool:
        return guild_id in self.sessions


def _channel(channel_id: int = CHANNEL_ID, members=()):
    return SimpleNamespace(id=channel_id, members=list(members))


def _member(user_id: int, channel=None, *, admin=False, bot=False):
    return SimpleNamespace(
        id=user_id,
        bot=bot,
        display_name=f"user{user_id}",
        voice=SimpleNamespace(channel=channel) if channel is not None else None,
        guild_permissions=SimpleNamespace(administrator=admin),
    )


class Ctx:
    def __init__(self, author, guild) -> None:
        self.author = author
        self.guild = guild
        self.sent: list[str] = []

    async def send(self, text: str = "") -> None:
        self.sent.append(text)


def _guild(voice_client=None):
    return SimpleNamespace(id=GUILD_ID, voice_client=voice_client)


def make_bot(**listen_kwargs):
    listen_kwargs.setdefault("enabled", True)
    listen_kwargs.setdefault("allowed_channel_ids", [CHANNEL_ID])
    listen_kwargs.setdefault("operator_user_ids", [OPERATOR_ID])
    bot = commands.Bot(command_prefix="!g", intents=discord.Intents.none())
    bot.app = SimpleNamespace(settings=AppSettings(listen=ListenSettings(**listen_kwargs)))
    bot.listen_sessions = FakeRegistry()
    bot.guild_states = GuildStateRegistry()
    bot.made: list[FakeSession] = []

    def make_listen_session(guild, channel):
        session = FakeSession(bot.app, bot, guild, channel)
        session.guild_id = guild.id
        bot.made.append(session)
        return session

    async def stop_listen_session(guild_id):
        session = bot.listen_sessions.get(guild_id)
        if session is not None:
            await session.stop()
            bot.listen_sessions.remove(guild_id)

    bot.make_listen_session = make_listen_session
    bot.stop_listen_session = stop_listen_session
    setup_commands(bot)
    return bot


def run(bot, path: tuple[str, ...], ctx) -> None:
    command = bot.get_command(" ".join(path))
    assert command is not None
    asyncio.run(command.callback(ctx))


def start_session(bot):
    guild = _guild()
    ctx = Ctx(_member(OPERATOR_ID, _channel()), guild)
    run(bot, ("listen", "start"), ctx)
    return bot.made[-1], ctx


# ------------------------------------------------------------------ start


def test_non_operator_is_refused() -> None:
    bot = make_bot()
    ctx = Ctx(_member(USER_ID, _channel()), _guild())
    run(bot, ("listen", "start"), ctx)
    assert bot.made == []
    assert "operator" in ctx.sent[0]


def test_admin_is_operator_when_no_operator_ids() -> None:
    bot = make_bot(operator_user_ids=[])
    ctx = Ctx(_member(USER_ID, _channel(), admin=True), _guild())
    run(bot, ("listen", "start"), ctx)
    assert len(bot.made) == 1
    plain = Ctx(_member(USER_ID + 1, _channel(), admin=False), _guild(GUILD_ID))
    bot2 = make_bot(operator_user_ids=[])
    run(bot2, ("listen", "start"), plain)
    assert bot2.made == []


def test_channel_not_allowed_is_refused() -> None:
    bot = make_bot()
    ctx = Ctx(_member(OPERATOR_ID, _channel(OTHER_CHANNEL_ID)), _guild())
    run(bot, ("listen", "start"), ctx)
    assert bot.made == []
    assert "not allowed" in ctx.sent[0]


def test_empty_allowed_list_refuses() -> None:
    bot = make_bot(allowed_channel_ids=[])
    ctx = Ctx(_member(OPERATOR_ID, _channel()), _guild())
    run(bot, ("listen", "start"), ctx)
    assert bot.made == []


def test_disabled_listen_refuses() -> None:
    bot = make_bot(enabled=False)
    ctx = Ctx(_member(OPERATOR_ID, _channel()), _guild())
    run(bot, ("listen", "start"), ctx)
    assert bot.made == []
    assert "disabled" in ctx.sent[0]


def test_operator_outside_voice_is_refused() -> None:
    bot = make_bot()
    ctx = Ctx(_member(OPERATOR_ID), _guild())
    run(bot, ("listen", "start"), ctx)
    assert bot.made == []


def test_music_voice_client_blocks_start() -> None:
    bot = make_bot()
    ctx = Ctx(_member(OPERATOR_ID, _channel()), _guild(voice_client=object()))
    run(bot, ("listen", "start"), ctx)
    assert bot.made == []
    assert "Music" in ctx.sent[0]


def test_queued_music_blocks_start() -> None:
    bot = make_bot()
    bot.guild_state = bot.guild_states.get_or_create(GUILD_ID)
    bot.guild_state.queue.put_nowait(object())
    ctx = Ctx(_member(OPERATOR_ID, _channel()), _guild())
    run(bot, ("listen", "start"), ctx)
    assert bot.made == []


def test_start_registers_and_stop_removes() -> None:
    bot = make_bot()
    session, ctx = start_session(bot)
    assert session.started == 1
    assert bot.listen_sessions.get(GUILD_ID) is session
    stop_ctx = Ctx(_member(OPERATOR_ID, _channel()), _guild())
    run(bot, ("listen", "stop"), stop_ctx)
    assert session.stops == 1
    assert not bot.listen_sessions.active(GUILD_ID)


def test_second_start_is_refused() -> None:
    bot = make_bot()
    start_session(bot)
    ctx = Ctx(_member(OPERATOR_ID, _channel()), _guild())
    run(bot, ("listen", "start"), ctx)
    assert len(bot.made) == 1
    assert "already" in ctx.sent[0]


def test_failed_start_unregisters_and_hides_the_error() -> None:
    bot = make_bot()
    original = bot.make_listen_session

    def failing(guild, channel):
        session = original(guild, channel)
        session.fail_start = True
        return session

    bot.make_listen_session = failing
    ctx = Ctx(_member(OPERATOR_ID, _channel()), _guild())
    run(bot, ("listen", "start"), ctx)
    assert not bot.listen_sessions.active(GUILD_ID)
    assert bot.made[0].stops == 1
    assert ctx.sent == ["Could not start listen mode."]


def test_stop_requires_operator() -> None:
    bot = make_bot()
    session, _ = start_session(bot)
    ctx = Ctx(_member(USER_ID, _channel()), _guild())
    run(bot, ("listen", "stop"), ctx)
    assert session.stops == 0
    assert bot.listen_sessions.active(GUILD_ID)


# --------------------------------------------------------------- on / off


def test_on_from_inside_the_channel_opts_in() -> None:
    bot = make_bot()
    session, _ = start_session(bot)
    run(bot, ("listen", "on"), Ctx(_member(USER_ID, _channel()), _guild()))
    assert session.opted_in == {USER_ID}


def test_on_from_another_channel_or_no_voice_is_refused() -> None:
    bot = make_bot()
    session, _ = start_session(bot)
    run(bot, ("listen", "on"), Ctx(_member(USER_ID, _channel(OTHER_CHANNEL_ID)), _guild()))
    run(bot, ("listen", "on"), Ctx(_member(USER_ID), _guild()))
    assert session.opted_in == set()


def test_on_without_session_is_refused() -> None:
    bot = make_bot()
    ctx = Ctx(_member(USER_ID, _channel()), _guild())
    run(bot, ("listen", "on"), ctx)
    assert "not on" in ctx.sent[0]


def test_off_clears_synchronously_before_replying() -> None:
    bot = make_bot()
    session, _ = start_session(bot)
    session.opt_in(USER_ID)
    ctx = Ctx(_member(USER_ID, _channel()), _guild())
    original_send = ctx.send

    async def checking_send(text: str = "") -> None:
        assert session.events == [f"opt_out:{USER_ID}"]  # cleared before the reply
        await original_send(text)

    ctx.send = checking_send
    run(bot, ("listen", "off"), ctx)
    assert session.opted_in == set()
    assert ctx.sent


def test_off_works_from_outside_the_channel() -> None:
    bot = make_bot()
    session, _ = start_session(bot)
    session.opt_in(USER_ID)
    run(bot, ("listen", "off"), Ctx(_member(USER_ID), _guild()))
    assert session.opted_in == set()


def test_status_shows_session_text_only() -> None:
    bot = make_bot()
    start_session(bot)
    ctx = Ctx(_member(USER_ID), _guild())
    run(bot, ("listen", "status"), ctx)
    assert ctx.sent == [f"listening in {CHANNEL_ID}"]
    idle = make_bot()
    ctx = Ctx(_member(USER_ID), _guild())
    run(idle, ("listen", "status"), ctx)
    assert "not on" in ctx.sent[0]


# ---------------------------------------------------------- music guards


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("play", ("https://youtu.be/x",)),
        ("playnow", ("https://youtu.be/x",)),
        ("play_spotify", ("song",)),
        ("join", ()),
        ("podcast", ()),
    ],
)
def test_active_session_blocks_music_commands(name, args) -> None:
    bot = make_bot()
    start_session(bot)
    ctx = Ctx(_member(USER_ID, _channel()), _guild())
    # Any further work would touch attributes the fake bot lacks and raise.
    asyncio.run(bot.get_command(name).callback(ctx, *args))
    assert ctx.sent == [LISTEN_ON_MESSAGE]


def test_music_commands_are_not_blocked_without_a_session() -> None:
    bot = make_bot()
    ctx = Ctx(_member(USER_ID), _guild())
    with pytest.raises(commands.CommandError):  # reached the legacy path ("not in voice")
        asyncio.run(bot.get_command("join").callback(ctx))
    assert LISTEN_ON_MESSAGE not in ctx.sent


# ------------------------------------------------------- voice-state hook


def hook_bot(session, *, bot_id: int = 999):
    stops: list[int] = []
    registry = FakeRegistry()
    if session is not None:
        registry.sessions[GUILD_ID] = session
    fake = SimpleNamespace(
        _listen_sessions=registry,
        user=SimpleNamespace(id=bot_id),
        _spawn_listen_stop=stops.append,
    )
    return fake, stops


def voice_update(fake, member, before_channel, after_channel) -> None:
    before = SimpleNamespace(channel=before_channel)
    after = SimpleNamespace(channel=after_channel)
    asyncio.run(NotAMusicBot.on_voice_state_update(fake, member, before, after))


def _hook_member(user_id: int, *, bot: bool = False):
    return SimpleNamespace(id=user_id, bot=bot, guild=SimpleNamespace(id=GUILD_ID))


def test_leaving_the_session_channel_calls_on_member_left() -> None:
    session = FakeSession(None, None, None, _channel())
    fake, stops = hook_bot(session)
    other = _hook_member(5)
    remaining = _hook_member(6)
    voice_update(fake, other, _channel(members=[remaining]), None)
    assert session.left == [5]
    assert stops == []


def test_moving_out_calls_on_member_left() -> None:
    session = FakeSession(None, None, None, _channel())
    fake, stops = hook_bot(session)
    voice_update(
        fake, _hook_member(5), _channel(members=[_hook_member(6)]), _channel(OTHER_CHANNEL_ID)
    )
    assert session.left == [5]


def test_other_channels_and_mute_toggles_are_ignored() -> None:
    session = FakeSession(None, None, None, _channel())
    fake, stops = hook_bot(session)
    voice_update(fake, _hook_member(5), _channel(OTHER_CHANNEL_ID), None)
    voice_update(fake, _hook_member(5), _channel(), _channel())  # mute toggle
    voice_update(fake, _hook_member(5), None, _channel())  # joining never opts in
    assert session.left == []
    assert stops == []


def test_last_human_leaving_stops_the_session() -> None:
    session = FakeSession(None, None, None, _channel())
    fake, stops = hook_bot(session)
    voice_update(fake, _hook_member(5), _channel(members=[_hook_member(999, bot=True)]), None)
    assert session.left == [5]
    assert stops == [GUILD_ID]


def test_bot_disconnect_stops_the_session() -> None:
    session = FakeSession(None, None, None, _channel())
    fake, stops = hook_bot(session)
    voice_update(fake, _hook_member(999, bot=True), _channel(), None)
    assert stops == [GUILD_ID]
    assert session.left == []


def test_no_session_means_no_action() -> None:
    fake, stops = hook_bot(None)
    voice_update(fake, _hook_member(5), _channel(), None)
    assert stops == []
