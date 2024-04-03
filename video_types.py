from dataclasses import dataclass
from typing import AsyncIterator
from typing import Optional
import asyncio
import json
import subprocess
from pathlib import Path

import discord

import configuration as cfg

@dataclass
class User:
    name: str
    id: int
    real_name: str = None  # Default value None for real_name

    def __post_init__(self):
        # Load real names from JSON file
        with open(cfg.JSON_USERS, 'r') as file:
            users_dict = json.load(file)
        
        # Check if the user's name exists in the JSON data
        if self.name in users_dict:
            self.real_name = users_dict[self.name]['real_name']
        else:
            # If real name not found, set it to the same as username
            self.real_name = self.name

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