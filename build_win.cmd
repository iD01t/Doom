@echo off
echo "Setting up Python virtual environment..."
python -m venv venv
if %errorlevel% neq 0 (
    echo "Failed to create virtual environment. Is python installed and in your PATH?"
    exit /b 1
)

echo "Installing dependencies..."
call venv\Scripts\pip install --upgrade pip
call venv\Scripts\pip install PyQt6 pillow numpy matplotlib pyinstaller
if %errorlevel% neq 0 (
    echo "Failed to install dependencies."
    exit /b 1
)

echo "Running PyInstaller..."
call venv\Scripts\pyinstaller --noconfirm --clean ^
  --name "Sprite Forge Enhanced" ^
  --onefile ^
  --windowed ^
  --icon=icons/app_icon.ico ^
  sprite_forge_enhanced.py

if %errorlevel% neq 0 (
    echo "PyInstaller build failed."
    exit /b 1
)

echo "Build complete! Executable is in the dist/ directory."
