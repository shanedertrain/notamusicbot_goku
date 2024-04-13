# Discord Bot for Playing Audio

This Discord bot is designed to play audio in a voice channel based on user requests. It supports adding videos or playlists to a queue and playing them sequentially. The bot also generates a pre-play audio file using text-to-speech synthesis for each video added to the queue.

## Features

- **Queue Management**: Add videos or playlists to the queue and start playing them in the voice channel.
- **Pre-Play Audio**: Generate audio commentary for each video using text-to-speech synthesis.
- **Playback Control**: Skip, stop, or leave the voice channel at any time during playback.

## Setup

### Prerequisites

- Python 3.9 >=< 3.10
- `ffmpeg` installed and added to the system PATH: https://ffmpeg.org/download.html
- Windows 10 SDK installed to build fairseq: https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/
- hubert_base.pt placed into in the root directory: https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt

### Installation

1. Clone this repository to your local machine:

    git clone <repository_url>

2A.

        1. Run the `setup.sh` script:

            ./setup.sh

OR

2B.

        1. Initialize and update submodules:

            git submodule update --init --recursive


        2. Install the required Python packages:

            pip install -r requirements.txt


3. Update the `.env` file in the root directory of the project:

    DISCORD_TOKEN=<your_discord_token>  
    DISCORD_GUILD=<discord_server_name>  
        As it appears in the Discord client  
    FILENAME_START_SOUND=<startup_sound>  
        This plays when the bot first runs or joins a channel
    GENERATOR=<gpu_or_cpu>  
        GPU or CPU depending on what you want to use. If no CUDA cores (NVIDIA), then use CPU
    NEWS_API_KEY=<news_api_key>
        API key for newsapi: https://newsapi.org/

4. Update the `.env` file in the rvc_cli directory of the project:  
    GEMENI_API_KEY=<your_api_key>
    ESTIMATE_MULTIPLIER=<estimate_multiplier>
        Time it takes times the length of the file. Ex. 4.22 = 422% longer than length of file.

5. Run `goku.py` and then update the `users.json` file with the backgrounds of the users in your Discord server. 

6. Restart `goku.py` for the changes to take effect. 

## Adding models 

1. 

## Usage

1. Start the bot by running the `goku.py` script:

2. Use the command prefix `!g` followed by the desired command to interact with the bot. For example:

- `!gplay <video_url>`: Add a video or playlist to the queue and start playing.
- `!gstop`: Stop playing the audio and disconnect from the voice channel.
- `!gskip`: Skip the current song and move to the next in the playlist.

## Contributors

- Guy Man - Developer | https://github.com/shanedertrain 
- AshtonScalise - Developer | https://github.com/AshtonScalise

## License

This project is licensed under the [MIT License](LICENSE).
