# Sprite Forge 2025

**Professional Doom Sprite Creation Toolkit**

![Banner](docs/banner.png) <!-- optional placeholder for banner -->

Sprite Forge 2025 is the **all-in-one solution** for creating, editing, and packaging sprites for Doom, Doom II, and GZDoom in 2025.
It combines the power of Photoshop-style editing, WAD/PK3 packaging, and batch slicing into a single modern, plugin-driven application.

Built with **PyQt6**, **Pillow**, and **NumPy**, this tool is designed for both modders and professionals who want to streamline the entire sprite workflow.

---

## ✨ Features

* **Modern PyQt6 GUI**

  * Dark theme
  * Zoom/pan canvas with transparency checkerboard
  * Tabs for live preview and logs
  * Toolbar for quick zoom/reset

* **Full Sprite Workflow**

  * Auto-slice and combine frames
  * Pixelate and quantize to Doom palette
  * Apply outlines, recolors, shadows
  * Export directly to **PK3** or **WAD layout**

* **Plugin System (JSON-based)**

  * Drop `.json` configs into the `plugins/` folder
  * Auto-loads new tools without touching the core
  * Example plugins included:

    * **AI Recolor** (HSV shifting, skin-tone preservation)
    * **3D Preview** (isometric sprite projection)
    * **Edge Enhancement** (sharpen and enhance details)

* **Smart Export**

  * PK3 export with `TEXTURES` lump and ZScript stub
  * WAD layout with auto-mirrored rotations and JSON info file

* **Persistence**

  * Remembers last sprite name, type, and window geometry via `QSettings`

* **Cross-Platform**

  * Works on Windows, macOS, and Linux (X11 with Qt)

---

## 📦 Installation

Clone the repo and run directly:

```bash
git clone https://github.com/YOUR_USERNAME/sprite-forge-2025.git
cd sprite-forge-2025
python sprite_forge_enhanced.py
```

The script auto-installs dependencies if missing:

* [PyQt6](https://pypi.org/project/PyQt6/)
* [Pillow](https://pypi.org/project/pillow/)
* [NumPy](https://pypi.org/project/numpy/)

Manual install:

```bash
pip install PyQt6 pillow numpy
```

---

## 🚀 Usage

### Run the GUI

```bash
python sprite_forge_enhanced.py
```

1. Load a PNG/JPG/BMP sprite sheet.
2. Adjust **Sprite Settings** (name, type, size).
3. Apply processing (pixelate, Doom palette, transparency).
4. Preview results instantly.
5. Export as **PK3** or **WAD layout**.

### Export PK3

* Generates `sprites/` folder with correctly named frames.
* Creates `TEXTURES` lump with offsets.
* Includes a basic **ZScript actor** stub for quick testing.

### Export WAD Layout

* Saves mirrored rotations into a `_wad/` folder.
* Writes a `_info.json` file with metadata.

---

## 🔌 Plugin System

Plugins are defined in **JSON config files** placed in the `plugins/` folder.

Example:

```json
{
  "name": "AI Recolor",
  "version": "1.0.0",
  "description": "Intelligent color transformation",
  "author": "Sprite Forge Team",
  "category": "Color",
  "parameters": {
    "hue_shift": {"type": "slider", "min": -180, "max": 180, "default": 0, "label": "Hue Shift"},
    "saturation": {"type": "slider", "min": 0.0, "max": 2.0, "default": 1.0, "label": "Saturation"},
    "lightness": {"type": "slider", "min": 0.0, "max": 2.0, "default": 1.0, "label": "Lightness"}
  },
  "processing": {
    "type": "hsv_transform",
    "steps": [
      {"action": "convert_hsv"},
      {"action": "shift_hue", "param": "hue_shift"},
      {"action": "scale_saturation", "param": "saturation"},
      {"action": "scale_lightness", "param": "lightness"},
      {"action": "convert_rgb"}
    ]
  }
}
```

⚡ When the app starts, it automatically loads all plugins and creates UI widgets for their parameters.

---

## 📷 Screenshots

| Canvas View               | Plugin Panel              | PK3 Export                |
| ------------------------- | ------------------------- | ------------------------- |
| ![](docs/screenshot1.png) | ![](docs/screenshot2.png) | ![](docs/screenshot3.png) |

*(place images in `docs/` or use Imgur links)*

---

## ⚠️ Notes

* CLI batch mode is available in the original `sprite_forge_2025.py`.
* Enhanced version focuses on **GUI and plugins**.
* For automation pipelines, combine both versions.

---

## 🗺️ Roadmap

* [ ] Merge CLI + GUI for a unified tool
* [ ] Add AI-powered plugins (text-to-sprite, style transfer)
* [ ] Plugin marketplace with easy sharing
* [ ] Advanced PK3 packaging (DECORATE, ACS scripts)
* [ ] Mac/Linux `.app` and `.AppImage` builds

---

## 👨‍💻 Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingPlugin`)
3. Commit changes (`git commit -m 'Add AmazingPlugin'`)
4. Push branch (`git push origin feature/AmazingPlugin`)
5. Open a Pull Request

---

## 📜 License

MIT License.
See [LICENSE](LICENSE) for details.

---

## 💡 Credits

* Doom community & ZDoom forums
* Original authors of **Pillow**, **PyQt6**, **NumPy**
* Sprite Forge Team 2025

