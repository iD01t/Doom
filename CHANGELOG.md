# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2025-09-08

### Added
- **Unified Codebase:** Merged the functionality of `sprite_forge_pro.py`, `sprite_forge_enhanced.py`, and `sprite_forge_ultimate.py` into a single, professional-grade script.
- **Project System:** Implemented a robust project management system. Projects are saved as `.sfp` files (zipped archives) containing a `project.json` manifest and all image assets.
- **Advanced Image Processing:** Added a comprehensive suite of image manipulation tools, including:
    - Nine-patch generation, padding, and bleed control.
    - Advanced color transformations (HSL, brightness/contrast, grayscale, color replacement).
    - Glow and outline effects.
    - Premultiplied alpha support.
- **Undo/Redo Functionality:** Integrated a command-based undo/redo manager to support non-destructive editing.
- **Plugin Architecture:** Created a flexible plugin system that can discover and run user-created plugins from a `plugins/` directory, with an API that gives plugins access to the core application engine.
- **Expanded Export Formats:** Added support for WebP and AVIF image formats.
- **Command-Line Interface (CLI):** Developed a powerful CLI for headless operation, allowing for batch processing, plugin execution, and sprite sheet packing from the terminal.
- **Configuration System:** Settings can now be loaded from JSON, YAML, and environment variables.
- **Comprehensive Test Suite:** Developed a `pytest` suite to ensure the stability and correctness of all major components.

### Changed
- **Authoritative Base:** Established `sprite_forge_pro.py` as the foundation and incrementally added features from other scripts and new requirements.
- **Refactored Architecture:** The application is now built around a central `CoreEngine` that manages a `Project` object, providing a clean and scalable architecture.
- **Sprite Sheet Packer:** Replaced the `rectpack` library dependency with a simpler, more reliable custom packing algorithm to resolve critical bugs encountered with the library in the execution environment.

### Removed
- **`rectpack` Dependency:** The external `rectpack` library has been removed in favor of a custom implementation.
- **Redundant Scripts:** The separate `sprite_forge_enhanced.py` and `sprite_forge_ultimate.py` files are no longer needed.
