import os

import discord
from dotenv import load_dotenv

import configuration as cfg
from bot_manager import BotManager

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
intents.typing = False
intents.presences = False
intents.members = True


def clear_folder_contents(folder: str):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Deletes a file or symbolic link
        except Exception as e:
            cfg.LOGGER.error(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == "__main__":
    clear_folder_contents(cfg.FOLDER_TTS)
    clear_folder_contents(cfg.FOLDER_OUTPUT)

    bot_manager = BotManager(command_prefix="!g", intents=intents)

    bot_manager.run(TOKEN)
