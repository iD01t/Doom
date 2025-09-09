# Sprite Forge Enhanced v3.1.0

**Sprite Forge Enhanced** is a professional-grade tool for creating and manipulating 2D game sprites. It features a rich graphical user interface, an extensible plugin system, advanced image processing capabilities, and powerful export options tailored for game development, including Doom engine formats.

This document provides a guide to installing, using, and building the application.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start (GUI)](#quick-start-gui)
- [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Batch Processing](#batch-processing)
  - [Self-Test](#self-test)
- [Building from Source](#building-from-source)
- [Plugin System](#plugin-system)

## Features

*   **Modern GUI:** An intuitive PyQt6 interface for visual editing.
*   **Advanced Canvas:** Zoom, pan, and grid features for precise control.
*   **Image Processing:** A suite of built-in tools like Pixelate, Doom Palette, Enhance, Auto-crop, and Background Removal.
*   **Plugin System:** Extend the application's functionality with custom plugins.
*   **Game-Ready Exports:** Export to PNG, GIF, ZIP, and specialized game formats like PK3 (for Doom 3 / Quake 4) and WAD (directory-based for classic Doom).
*   **Headless Operation:** Automate your workflow with a powerful command-line interface.
*   **Built-in Self-Test:** Verify the application's integrity with a single command.

## Installation

The application requires Python 3.8+ and the following packages: `PyQt6`, `Pillow`, `numpy`, `matplotlib`.

When you first run the application, it will check for these dependencies. If any are missing, it will prompt you to install them automatically.

To manually install, run:
```bash
pip install PyQt6 Pillow numpy matplotlib
```

## Quick Start (GUI)

To launch the graphical user interface, run the script from your terminal:

```bash
python3 sprite_forge_enhanced.py
```

1.  Click **"Open Image"** to load a sprite.
2.  Use the **Plugin Selection** panel on the left to choose an effect. Adjust its parameters and click **"Preview"** or **"Apply"**.
3.  Use the **Quick Tools** for common operations.
4.  Use the **Export Options** on the right to select a format and sprite name.
5.  Click **"Export"** to save your work in the chosen format.

## Command-Line Interface (CLI)

The application can be run from the command line for automation and testing.

### Batch Processing

The `--batch` flag allows you to run a sequence of image processing operations on an input file and save the result without opening the GUI.

**Synopsis:**
```bash
python3 sprite_forge_enhanced.py --batch \
  --input <path> \
  --ops "op1:param=val;op2" \
  --export <FORMAT> \
  --sprite-name <name> \
  --out <path>
```

**Arguments:**
*   `--input`: Path to the source image.
*   `--ops`: A semicolon-separated string of operations.
    *   Each operation can have colon-separated parameters (e.g., `pixelate:factor=2`).
    *   Parameters are comma-separated key-value pairs (e.g., `enhance:brightness=1.5,contrast=1.2`).
    *   Available ops: `pixelate`, `doom_palette`, `enhance`, `auto_crop`.
*   `--export`: The export format (e.g., `PNG`, `PK3`, `WAD`, `GIF`, `ZIP`).
*   `--sprite-name`: The internal name for the sprite, used by PK3 and WAD formats.
*   `--out`: The path to the output file or directory.

**Example:**
```bash
python3 sprite_forge_enhanced.py --batch \
  --input "my_sprite.png" \
  --ops "pixelate:factor=2;enhance:brightness=1.2" \
  --export "PK3" \
  --sprite-name "IMPX0" \
  --out "dist/my_sprite.pk3"
```

### Self-Test

The `--selftest` flag runs a built-in suite of smoke tests to ensure the core functionality is working correctly. It will generate a test image, process it, export it to all formats in a temporary directory, and report the results.

```bash
python3 sprite_forge_enhanced.py --selftest
```
On success, the script will exit with code 0. On failure, it will exit with a non-zero code.

## Building from Source

Build scripts and a PyInstaller spec file are provided to package the application into a standalone executable for Windows, macOS, and Linux.

1.  **Install PyInstaller:** `pip install pyinstaller`
2.  **Run the build script** for your operating system:
    *   **Windows:** `build_win.cmd`
    *   **macOS:** `sh build_mac.sh`
    *   **Linux:** `sh build_linux.sh`
3.  The final executable will be located in the `dist/` directory.

## Plugin System

The plugin system is not yet fully documented for external developers. You can find examples of the built-in plugins within the `sprite_forge_enhanced.py` script.
