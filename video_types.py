from dataclasses import dataclass
from typing import AsyncIterator
from typing import Optional
import asyncio
import json
import subprocess
from pathlib import Path

import discord

from users import User

@dataclass
class Video:
    title: str
    url: str
    duration: int
    uploader: str
    view_count: int
    formats: list
    channel: discord.TextChannel  # Add the channel property
    requester: Optional[User] = None  
    path_pre_play: Optional[Path] = None 

async def extract_info(url, channel: discord.TextChannel) -> AsyncIterator[Video]:
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
        yield Video(
            title=info['title'],
            url=info['url'],
            duration=info['duration'],
            uploader=info['uploader'],
            view_count=info['view_count'],
            formats=info['formats'],
            channel=channel  # Assign the channel property
        )

    await process.communicate()