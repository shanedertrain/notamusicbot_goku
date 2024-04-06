#!/bin/bash

# Open the Windows SDK download link
xdg-open https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/

# Open the FFmpeg download link
xdg-open https://ffmpeg.org/download.html

# Open the hubert_base.pt download link
xdg-open https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt

# Update and initialize submodules
git submodule update --init --recursive

# Install Python dependencies
pip install -r requirements.txt

echo "Setup complete."
