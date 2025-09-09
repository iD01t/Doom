# Sprite Forge Pro 2025

**Version:** 3.0.0
**Author:** Sprite Forge Team (Refactored by Jules)

## 1. Overview

Sprite Forge Pro 2025 is a state-of-the-art sprite and texture creation toolkit designed for professional game development and modding. It provides a comprehensive suite of tools for image processing, sprite sheet generation, and project management, accessible through both a graphical user interface (GUI) and a powerful command-line interface (CLI).

This version represents a complete architectural overhaul, unifying previously separate scripts into a single, robust, and extensible application.

## 2. Key Features

- **Project-Based Workflow:** Manage all your assets within a single `.sfp` project file. This file bundles your images, animations, and settings into a convenient, portable package.
- **Advanced Sprite Sheet Packing:** Generate optimized sprite sheets with a custom, reliable packing algorithm. Control padding and apply automatic bleed to prevent texture artifacts.
- **Powerful Image Processing:** A rich set of tools to manipulate your images:
    - Palette application (includes enhanced Doom palette)
    - Color correction (brightness, contrast, HSL)
    - Grayscale conversion
    - Color replacement and keying
    - Glow and outline effects
- **Animation Support:** Define and manage sprite animations within your project.
- **Extensible Plugin System:** Extend the application's functionality by creating your own Python plugins. A simple "Invert Colors" plugin is included as an example.
- **Flexible Configuration:** Configure settings through `config.yaml`, `settings.json`, or environment variables.
- **Undo/Redo:** Non-destructive workflow is supported by a robust undo/redo system.
- **Multiple Export Formats:** Export images and sprite sheets to standard formats like PNG, GIF, WebP, and AVIF.

## 3. Installation

The script includes a dependency auto-installer that will attempt to install required Python packages when run. To disable this, set the environment variable `SPRITE_FORGE_NO_AUTO_INSTALL=1`.

The core dependencies are: `Pillow`, `numpy`, `PyQt6`, `PyYAML`, `scipy`.

## 4. Usage

Sprite Forge Pro can be run in two modes: GUI mode and Headless (CLI) mode.

### 4.1. GUI Mode

To launch the graphical user interface, simply run the script without any arguments:

```bash
python3 sprite_forge_pro.py
```

This will open the main application window, from which you can manage your project, process images, and generate sprite sheets.

### 4.2. Headless / CLI Mode

The command-line interface is ideal for automation and batch processing.

**Synopsis:**
```bash
python3 sprite_forge_pro.py --headless --input <path> --output <path> [--pack] [--run-plugin <name>]
```

**Arguments:**
- `--headless`: (Required) Run in command-line mode without launching the GUI.
- `--input <path>`: (Required) Path to the source assets. This can be:
    - A directory of images (`/path/to/images/`).
    - A project file (`/path/to/project.sfp`).
- `--output <path>`: (Required for packing) The base path for the output files (e.g., `/path/to/my_sheet`). The file extension will be added automatically.
- `--pack`: A flag that triggers the sprite sheet packing process.
- `--run-plugin <name>`: The name of a plugin to execute on the project (e.g., `"Invert Colors"`).

**Example: Pack a directory of images into a sprite sheet**
```bash
python3 sprite_forge_pro.py --headless --input ./my_sprites --output ./sheets/my_sheet --pack
```

## 5. Plugin Development

To create your own plugin:
1. Create a new Python file in the `plugins/` directory.
2. Define a class that inherits from `BasePlugin`.
3. Set the `info` class attribute with your plugin's metadata.
4. Implement the `run(self, **kwargs)` method. This method has access to the core engine via `self.core`.
5. The application will automatically discover and load your plugin on startup.
