# Test Plan

This document outlines the testing strategy for the Sprite Forge Pro application.

## 1. Testing Framework

The test suite is built using the **pytest** framework. `pytest` was chosen for its powerful features, simple assertion syntax, and extensive plugin ecosystem.

## 2. Testing Strategy

A multi-layered testing strategy has been implemented to ensure code quality and application stability.

### 2.1. Unit Tests

Unit tests are focused on isolating and testing individual components (classes and methods) of the application. This ensures that the fundamental building blocks of the code work correctly.

- **ImageProcessor (`tests/test_image_processor.py`):** Each image manipulation function is tested with a small, predictable input image. Assertions are made on the output image's properties (e.g., dimensions, the color of a specific pixel) to verify the function's behavior.
- **Core Data Structures (`tests/test_project.py`):** Tests ensure that `Project`, `Sprite`, and `SpriteFrame` objects can be created and linked together correctly.
- **Undo Manager (`tests/test_undo_manager.py`):** The undo/redo mechanism is tested by executing commands and then undoing/redoing them, asserting that the application state changes as expected at each step.

### 2.2. Integration Tests

Integration tests are designed to verify that different components of the application work together correctly.

- **Project I/O (`tests/test_io.py`):** A critical integration test covers the entire project lifecycle: creating a complex project in memory, saving it to a `.sfp` file, loading it back into a new application instance, and asserting that the loaded project is identical to the original. This validates the serialization and deserialization logic.
- **Sprite Sheet Manager (`tests/test_sheet_manager.py`):** These tests verify that the manager can correctly populate its data from a project, run the packing algorithm, and generate a valid sprite sheet atlas.

### 2.3. Dependency Testing

During development, a critical bug was discovered in the `rectpack` library within the execution environment. A dedicated test (`test_rectpack_direct_usage`) was created to isolate and confirm the library's faulty behavior. This led to the decision to replace the dependency with a more reliable custom implementation, demonstrating a testing-driven approach to resolving external library issues.

## 3. How to Run Tests

To run the full test suite, follow these steps:

1.  **Install Dependencies:** Ensure all required dependencies are installed. The test environment requires `pytest`, `Pillow`, `numpy`, `scipy`, and `PyYAML`.
    ```bash
    pip install pytest Pillow numpy scipy PyYAML
    ```

2.  **Run Pytest:** From the root directory of the project, execute the following command. The `SPRITE_FORGE_NO_AUTO_INSTALL` environment variable is required to prevent the application's auto-installer from interfering with the test run.
    ```bash
    SPRITE_FORGE_NO_AUTO_INSTALL=1 python3 -m pytest
    ```
