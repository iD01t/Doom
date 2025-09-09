# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2025-09-08

### Added
- **Stability & Polish:** Implemented comprehensive error handling, graceful plugin loading, and a file-based logging system.
- **UX Upgrades:** Added keyboard shortcuts, improved canvas zooming/panning, and provided clear status messages for all operations.
- **Headless/Batch Mode:** Introduced a command-line interface (`--batch`) for scripted image processing without the GUI.
- **Self-Test Mode:** Added a `--selftest` argument to run automated end-to-end smoke tests.
- **Dependency Checker:** The application now checks for required dependencies at startup and offers to install them.
- **Settings Persistence:** Window size, export format, and sprite name are now saved between sessions.
- **Packaging Scripts:** Included build scripts (`build_win.cmd`, `build_mac.sh`, `build_linux.sh`) and a PyInstaller spec file for creating distributable packages.

### Changed
- **Image Pipeline:** Internal image handling is now consistently RGBA for more predictable processing.
- **Export Logic:** PK3 and WAD export formats have been improved with better validation and user prompts.
- **About Dialog:** The "About" dialog has been updated with the new version number and dependency information.

### Fixed
- **Canvas Rendering:** Corrected issues with the transparency checkerboard background and painter state.
- **Plugin Previews:** Ensured that applying a plugin preview does not mutate the original image.
- `auto_crop()` now works correctly with RGBA images.
