#!/bin/bash

# Open the Windows SDK download link
start https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/

# Open the FFmpeg download link
start https://ffmpeg.org/download.html

# Open the hubert_base.pt download link
start https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt

# Update and initialize submodules
git submodule update --init --recursive

# Install Python dependencies
pip install -r requirements.txt

echo "Setup complete."
