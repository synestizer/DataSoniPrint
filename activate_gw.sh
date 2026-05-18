#!/bin/bash
# Activate GW Explorer environment with all necessary libraries

# Activate virtual environment
source "$(dirname "$0")/.venv/bin/activate"

# Set library paths for pyo and audio libraries
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/flac/lib:/opt/homebrew/opt/libsndfile/lib:/opt/homebrew/opt/portaudio/lib:$DYLD_LIBRARY_PATH

echo "✓ GW Explorer environment activated"
echo "You can now run: python3 src/gw_explorer.py"
