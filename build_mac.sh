#!/bin/bash
set -e

echo "Setting up Python virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment. Is python3 installed and in your PATH?"
    exit 1
fi

echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install PyQt6 pillow numpy matplotlib pyinstaller
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    exit 1
fi

echo "Running PyInstaller..."
pyinstaller --noconfirm --clean \
  --name "Sprite Forge Enhanced" \
  --onefile \
  --windowed \
  --icon=icons/app_icon.icns \
  sprite_forge_enhanced.py

if [ $? -ne 0 ]; then
    echo "PyInstaller build failed."
    exit 1
fi

echo "Build complete! Application bundle is in the dist/ directory."
