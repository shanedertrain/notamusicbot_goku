#!/bin/bash

# Open the Windows SDK download link
xdg-open https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/

# Open the FFmpeg download link
xdg-open https://ffmpeg.org/download.html

# Update and initialize submodules
git submodule update --init --recursive

# Install Python dependencies
pip install -r requirements.txt

echo "Setup complete."
