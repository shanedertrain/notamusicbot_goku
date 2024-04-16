from dataclasses import dataclass
from typing import AsyncIterator
import asyncio
import json
import subprocess
from pathlib import Path
import youtube_dlc
import uuid

import discord

from users import User

@dataclass
class VideoInfo:
    title: str
    url: str
    duration: int
    uploader: str
    formats: list

@dataclass
class Media:
    id:str

@dataclass
class Video(Media):
    requester: User
    video_info: VideoInfo
    requested_channel: discord.TextChannel

    def __repr__(self) -> str:
        # Customize the representation of Video to exclude video_info details
        return (f"Video(id={self.id!r}, requester={self.requester!r}), requested_channel={self.requested_channel!r})")

@dataclass
class Audio(Media):
    filepath: Path

async def extract_video_info(url:str) -> AsyncIterator[VideoInfo]:
    process = await asyncio.create_subprocess_exec(
        'youtube-dlc', '--skip-download', '--dump-json', '--format', 'bestaudio', url,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    while True:
        line = await process.stdout.readline()
        if not line:
            break

        info = json.loads(line.decode())
        yield VideoInfo(
            title=info['title'],
            url=info['url'],
            duration=info['duration'],
            uploader=info['uploader'],
            formats=info['formats'],
        )

    await process.communicate()