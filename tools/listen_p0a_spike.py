"""Listen-mode P0a spike: can we receive and decrypt Discord voice (DAVE E2EE)?

Joins ONE named voice channel for a short, fixed time, counts what arrives and
leaves. It answers: does ``discord-ext-voice-recv`` hand us audio that decodes
once DAVE is on, or only transport-decrypted frames that still need
``davey`` decryption (the library has no DAVE code of its own).

Privacy, by construction:
  * Nothing is written to disk and no transcription runs. Only counters and
    a level (dBFS) per user are kept, in memory, and printed at the end.
  * On join the bot posts a notice in the channel's chat. Only users who react
    with the check mark (or are passed by ``--consent-ids``) are analysed.
    Every other user's packets are dropped BEFORE DAVE decrypt and opus decode.
  * It never speaks. It leaves after ``--seconds``.

    .venv-main/bin/python tools/listen_p0a_spike.py --channel-id 123 --seconds 60
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import davey
import discord
import numpy as np
from discord.ext import voice_recv
from discord.opus import Decoder as OpusDecoder

from notamusicbot.config import SecretsSettings

CONSENT_EMOJI = "✅"
# An opus silence frame (what clients send between talk spurts).
_OPUS_SILENCE = b"\xf8\xff\xfe"


@dataclass
class UserStats:
    name: str
    packets: int = 0
    silence_frames: int = 0
    dave_ok: int = 0
    dave_fail: int = 0
    dave_first_error: str = ""
    opus_ok: int = 0
    opus_fail: int = 0
    opus_first_error: str = ""
    rms_dbfs_max: float = -120.0
    decoder: OpusDecoder | None = field(default=None, repr=False)


class Spike(voice_recv.AudioSink):
    def __init__(self, client: voice_recv.VoiceRecvClient, consented: set[int]) -> None:
        super().__init__()
        self._client = client
        self.consented = consented
        self.stats: dict[int, UserStats] = {}
        self.dropped_no_consent = 0
        self.unknown_source = 0
        self.dave_versions: set[int] = set()

    def wants_opus(self) -> bool:
        return True

    def cleanup(self) -> None:
        return None

    def write(self, user: discord.abc.User | None, data: voice_recv.VoiceData) -> None:
        # Reader thread. Consent check first: no decrypt, no decode for others.
        if user is None:
            self.unknown_source += 1
            return
        if user.id not in self.consented or user.id == self._client.user.id:
            self.dropped_no_consent += 1
            return
        st = self.stats.setdefault(user.id, UserStats(name=user.display_name))
        st.packets += 1
        opus = data.opus
        if not opus:
            return
        if opus == _OPUS_SILENCE:
            st.silence_frames += 1
            return

        conn = self._client._connection
        self.dave_versions.add(conn.dave_protocol_version)
        payload = opus
        if conn.dave_protocol_version > 0 and conn.dave_session is not None:
            try:
                payload = conn.dave_session.decrypt(user.id, davey.MediaType.audio, opus)
                st.dave_ok += 1
            except Exception as exc:  # noqa: BLE001 - the spike reports any failure
                st.dave_fail += 1
                st.dave_first_error = st.dave_first_error or f"{type(exc).__name__}: {exc}"
                return

        if st.decoder is None:
            st.decoder = OpusDecoder()
        try:
            pcm = st.decoder.decode(payload, fec=False)
        except Exception as exc:  # noqa: BLE001
            st.opus_fail += 1
            st.opus_first_error = st.opus_first_error or f"{type(exc).__name__}: {exc}"
            return
        st.opus_ok += 1
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
        db = 20 * math.log10(rms) if rms > 0 else -120.0
        st.rms_dbfs_max = max(st.rms_dbfs_max, db)


def verdict(sink: Spike) -> str:
    if not sink.stats:
        return "NO DATA: nobody opted in and spoke, or no packets arrived"
    ok = sum(s.opus_ok for s in sink.stats.values())
    dave_fail = sum(s.dave_fail for s in sink.stats.values())
    opus_fail = sum(s.opus_fail for s in sink.stats.values())
    loud = max(s.rms_dbfs_max for s in sink.stats.values())
    if ok and loud > -60 and dave_fail * 5 < ok:
        return "PASS: audio decrypts and decodes with real signal levels (a few frames fail)"
    if ok and loud > -60:
        return "PARTIAL: decodes, but many DAVE decrypt failures"
    if ok and loud <= -60:
        return "UNCLEAR: frames decode but only near-silence; speak louder or longer"
    if dave_fail:
        return "FAIL: DAVE decrypt errors (see first error)"
    if opus_fail:
        return "FAIL: opus decode errors (likely still DAVE-encrypted)"
    return "NO DECODED AUDIO"


async def run(args: argparse.Namespace) -> int:
    token = SecretsSettings().discord_token
    if not token:
        print("DISCORD_TOKEN missing", file=sys.stderr)
        return 2

    intents = discord.Intents.default()
    intents.members = True
    intents.voice_states = True
    intents.guild_reactions = True
    client = discord.Client(intents=intents)
    done = asyncio.Event()
    result = {"code": 1}

    consented: set[int] = set(args.consent_ids or [])
    notice_ids: set[int] = set()

    @client.event
    async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
        if payload.message_id in notice_ids and str(payload.emoji) == CONSENT_EMOJI:
            if payload.user_id != client.user.id:
                consented.add(payload.user_id)

    @client.event
    async def on_raw_reaction_remove(payload: discord.RawReactionActionEvent) -> None:
        if payload.message_id in notice_ids and str(payload.emoji) == CONSENT_EMOJI:
            consented.discard(payload.user_id)

    @client.event
    async def on_voice_state_update(member, before, after) -> None:  # noqa: ANN001
        # Leaving the channel withdraws consent at once.
        if before.channel and before.channel != after.channel:
            consented.discard(member.id)

    @client.event
    async def on_ready() -> None:
        try:
            channel = client.get_channel(args.channel_id)
            if not isinstance(channel, discord.VoiceChannel | discord.StageChannel):
                print(f"channel {args.channel_id} is not a voice channel", file=sys.stderr)
                return
            humans = [m.display_name for m in channel.members if not m.bot]
            print(f"channel: {channel.guild.name} / {channel.name}; people present: {humans}")
            if args.consent_present:
                # Owner vouches that everyone present has agreed (relayed, not per-user).
                consented.update(m.id for m in channel.members if not m.bot)
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient, timeout=30)
            notice = await channel.send(
                f"Audio test (Goku bot). For the next {args.seconds} s I will check that I can "
                f"receive voice here. Nothing is recorded, saved or transcribed, and I will not "
                f"speak. React {CONSENT_EMOJI} to this message if you are happy to be included; "
                f"anyone who does not react is ignored, and leaving the channel withdraws it."
            )
            await notice.add_reaction(CONSENT_EMOJI)
            notice_ids.add(notice.id)
            sink = Spike(vc, consented)
            vc.listen(sink)
            print(f"listening for {args.seconds} s; opted in so far: {sorted(consented)}")
            await asyncio.sleep(args.seconds)
            vc.stop_listening()

            print("\n== P0a result ==")
            print(f"dave protocol versions seen: {sorted(sink.dave_versions)}")
            print(
                f"opted-in users: {len(sink.stats)}; "
                f"dropped (no consent): {sink.dropped_no_consent}; "
                f"unmapped source: {sink.unknown_source}"
            )
            by = defaultdict(dict)
            for s in sink.stats.values():
                by[s.name] = {
                    "packets": s.packets,
                    "silence": s.silence_frames,
                    "dave_ok": s.dave_ok,
                    "dave_fail": s.dave_fail,
                    "opus_ok": s.opus_ok,
                    "opus_fail": s.opus_fail,
                    "peak_dbfs": round(s.rms_dbfs_max, 1),
                    "dave_err": s.dave_first_error,
                    "opus_err": s.opus_first_error,
                }
            for name, row in by.items():
                print(f"  {name}: {row}")
            print(f"VERDICT: {verdict(sink)}")
            await notice.edit(content=f"{notice.content}\n\n(Test over. Nothing was kept.)")
            await vc.disconnect(force=True)
            result["code"] = 0
        except Exception as exc:  # noqa: BLE001
            print(f"spike error: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            done.set()

    async with client:
        await client.login(token)
        connect = asyncio.create_task(client.connect(reconnect=False))
        await done.wait()
        await client.close()
        connect.cancel()
    return result["code"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--channel-id", type=int, required=True)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument(
        "--consent-present",
        action="store_true",
        help="opt in everyone in the channel at join (owner confirms they agreed)",
    )
    ap.add_argument("--consent-ids", type=int, nargs="*", help="user ids opted in up front")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
