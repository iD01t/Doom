#!/usr/bin/env python3
"""
Sprite Forge - Professional Doom Sprite Creation Toolkit
======================================================================

Features:
- Professional PyQt6 interface
- JSON-based plugin system
- Advanced image canvas with zoom/pan
- Real-time preview pipeline
- Enhanced batch processing
- Modular architecture

MIT License - See original file for full license text
"""

import argparse
import json
import os
import sys
import time
import zipfile
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Auto-install dependencies
def ensure_dependencies():
    """Auto-install required packages if missing."""
    required = {
        'pillow': 'PIL',
        'numpy': 'numpy',
        'PyQt6': 'PyQt6'
    }
    
    for package, import_name in required.items():
        try:
            if import_name == 'PIL':
                import PIL
            else:
                __import__(import_name)
        except ImportError:
            print(f"Installing {package}...")
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

ensure_dependencies()

# Import GUI framework
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox,
        QSpinBox, QSlider, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
        QTabWidget, QGroupBox, QScrollArea, QSplitter, QStatusBar,
        QMenuBar, QMenu, QToolBar, QFrame, QSpacerItem, QSizePolicy
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings, QSize
    from PyQt6.QtGui import (
        QPixmap, QPainter, QColor, QFont, QAction, QIcon, QPalette,
        QTransform, QPen, QBrush
    )
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np

# Core sprite processing functions (imported from original)
def validate_sprite_name(name: str) -> str:
    """Ensure sprite name is exactly 4 uppercase characters."""
    cleaned = ''.join(c for c in name.upper() if c.isalnum())
    if len(cleaned) < 4:
        cleaned = cleaned.ljust(4, 'X')
    elif len(cleaned) > 4:
        cleaned = cleaned[:4]
    return cleaned

def pixelate_image(image: Image.Image, factor: int = 4) -> Image.Image:
    """Apply pixelation effect."""
    w, h = image.size
    small = image.resize((max(1, w // factor), max(1, h // factor)), Image.Resampling.NEAREST)
    return small.resize((w, h), Image.Resampling.NEAREST)

def apply_doom_palette(image: Image.Image, preserve_transparency: bool = True) -> Image.Image:
    """Apply Doom-style palette quantization."""
    doom_palette = [
        (0, 0, 0), (31, 23, 11), (23, 15, 7), (75, 75, 75),
        (255, 255, 255), (27, 27, 27), (19, 19, 19), (11, 11, 11),
        (199, 199, 199), (119, 119, 119), (83, 83, 83), (47, 47, 47),
        (255, 155, 0), (231, 119, 0), (203, 91, 0), (175, 71, 0),
        (143, 59, 0), (119, 47, 0), (91, 35, 0), (71, 27, 0),
        (199, 0, 0), (167, 0, 0), (139, 0, 0), (107, 0, 0),
        (0, 255, 0), (0, 231, 0), (0, 203, 0), (0, 175, 0),
        (0, 143, 0), (0, 119, 0), (0, 91, 0), (0, 71, 0)
    ]
    
    src = image.convert('RGBA')
    new_pixels = []
    
    for px in src.getdata():
        r, g, b, a = px
        if a == 0 and preserve_transparency:
            new_pixels.append((0, 0, 0, 0))
            continue
            
        best = doom_palette[0]
        best_dist = float('inf')
        for col in doom_palette:
            dr, dg, db = r - col[0], g - col[1], b - col[2]
            dist = dr * dr + dg * dg + db * db
            if dist < best_dist:
                best_dist = dist
                best = col
        new_pixels.append((*best, a))
    
    result = Image.new('RGBA', src.size)
    result.putdata(new_pixels)
    return result

# Plugin System
@dataclass
class PluginInfo:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    author: str
    category: str
    file_path: str

class BasePlugin(ABC):
    """Base class for all plugins."""
    
    @abstractmethod
    def get_info(self) -> PluginInfo:
        """Return plugin information."""
        pass
    
    @abstractmethod
    def process_image(self, image: Image.Image, **kwargs) -> Image.Image:
        """Process an image and return the result."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return plugin parameters for UI generation."""
        return {}

class PluginManager:
    """Manages plugin loading and execution."""
    
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = plugin_dir
        self.plugins: Dict[str, BasePlugin] = {}
        self.load_plugins()
    
    def load_plugins(self):
        """Load all plugins from the plugins directory."""
        if not os.path.exists(self.plugin_dir):
            os.makedirs(self.plugin_dir)
            self.create_example_plugins()
            return
        
        for file_path in Path(self.plugin_dir).glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    plugin_config = json.load(f)
                
                plugin = self.create_plugin_from_config(plugin_config, str(file_path))
                if plugin:
                    info = plugin.get_info()
                    self.plugins[info.name] = plugin
                    
            except Exception as e:
                print(f"Failed to load plugin {file_path}: {e}")
    
    def create_plugin_from_config(self, config: Dict, file_path: str) -> Optional[BasePlugin]:
        """Create a plugin instance from JSON configuration."""
        try:
            return JSONConfigPlugin(config, file_path)
        except Exception as e:
            print(f"Error creating plugin: {e}")
            return None
    
    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get plugin by name."""
        return self.plugins.get(name)
    
    def list_plugins(self) -> List[PluginInfo]:
        """List all loaded plugins."""
        return [plugin.get_info() for plugin in self.plugins.values()]
    
    def create_example_plugins(self):
        """Create example plugin configurations."""
        # AI Recolor Plugin
        ai_recolor = {
            "name": "AI Recolor",
            "version": "1.0.0",
            "description": "Intelligent color transformation using HSV shifting",
            "author": "Sprite Forge Team",
            "category": "Color",
            "parameters": {
                "hue_shift": {"type": "slider", "min": -180, "max": 180, "default": 0, "label": "Hue Shift"},
                "saturation": {"type": "slider", "min": 0.0, "max": 2.0, "default": 1.0, "label": "Saturation"},
                "lightness": {"type": "slider", "min": 0.0, "max": 2.0, "default": 1.0, "label": "Lightness"},
                "preserve_skin": {"type": "checkbox", "default": True, "label": "Preserve Skin Tones"}
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
        
        # 3D Preview Plugin
        preview_3d = {
            "name": "3D Preview",
            "version": "1.0.0",
            "description": "Generate isometric 3D preview of sprite",
            "author": "Sprite Forge Team",
            "category": "Preview",
            "parameters": {
                "angle": {"type": "slider", "min": 0, "max": 360, "default": 45, "label": "View Angle"},
                "height": {"type": "slider", "min": 0.1, "max": 3.0, "default": 1.0, "label": "Extrude Height"},
                "shadow": {"type": "checkbox", "default": True, "label": "Cast Shadow"}
            },
            "processing": {
                "type": "isometric_transform",
                "steps": [
                    {"action": "create_depth_map"},
                    {"action": "apply_perspective", "param": "angle"},
                    {"action": "extrude", "param": "height"},
                    {"action": "add_shadow", "condition": "shadow"}
                ]
            }
        }
        
        # Edge Enhancement Plugin
        edge_enhance = {
            "name": "Edge Enhancement",
            "version": "1.0.0",
            "description": "Enhance sprite edges and details",
            "author": "Sprite Forge Team",
            "category": "Enhancement",
            "parameters": {
                "strength": {"type": "slider", "min": 0.0, "max": 5.0, "default": 1.0, "label": "Enhancement Strength"},
                "threshold": {"type": "slider", "min": 0, "max": 255, "default": 50, "label": "Edge Threshold"}
            },
            "processing": {
                "type": "edge_filter",
                "steps": [
                    {"action": "detect_edges", "param": "threshold"},
                    {"action": "enhance_edges", "param": "strength"},
                    {"action": "blend_original"}
                ]
            }
        }
        
        # Save example plugins
        plugins = [ai_recolor, preview_3d, edge_enhance]
        for plugin_config in plugins:
            filename = f"{plugin_config['name'].lower().replace(' ', '_')}.json"
            with open(os.path.join(self.plugin_dir, filename), 'w') as f:
                json.dump(plugin_config, f, indent=2)

class JSONConfigPlugin(BasePlugin):
    """Plugin created from JSON configuration."""
    
    def __init__(self, config: Dict, file_path: str):
        self.config = config
        self.file_path = file_path
    
    def get_info(self) -> PluginInfo:
        """Return plugin information from config."""
        return PluginInfo(
            name=self.config["name"],
            version=self.config["version"],
            description=self.config["description"],
            author=self.config["author"],
            category=self.config["category"],
            file_path=self.file_path
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return plugin parameters."""
        return self.config.get("parameters", {})
    
    def process_image(self, image: Image.Image, **kwargs) -> Image.Image:
        """Process image according to JSON configuration."""
        processing = self.config.get("processing", {})
        proc_type = processing.get("type", "")
        steps = processing.get("steps", [])
        
        if proc_type == "hsv_transform":
            return self._process_hsv_transform(image, steps, kwargs)
        elif proc_type == "isometric_transform":
            return self._process_isometric(image, steps, kwargs)
        elif proc_type == "edge_filter":
            return self._process_edge_filter(image, steps, kwargs)
        else:
            return image
    
    def _process_hsv_transform(self, image: Image.Image, steps: List[Dict], params: Dict) -> Image.Image:
        """Process HSV color transformation."""
        import colorsys
        
        img = image.convert('RGBA')
        pixels = list(img.getdata())
        new_pixels = []
        
        hue_shift = params.get('hue_shift', 0) / 360.0
        saturation = params.get('saturation', 1.0)
        lightness = params.get('lightness', 1.0)
        
        for r, g, b, a in pixels:
            if a == 0:
                new_pixels.append((r, g, b, a))
                continue
                
            h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            
            # Apply transformations
            h = (h + hue_shift) % 1.0
            s = min(1.0, s * saturation)
            v = min(1.0, v * lightness)
            
            nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
            new_pixels.append((int(nr*255), int(ng*255), int(nb*255), a))
        
        result = Image.new('RGBA', img.size)
        result.putdata(new_pixels)
        return result
    
    def _process_isometric(self, image: Image.Image, steps: List[Dict], params: Dict) -> Image.Image:
        """Create isometric 3D preview."""
        # Simplified isometric transformation
        angle = params.get('angle', 45)
        height_factor = params.get('height', 1.0)
        
        img = image.convert('RGBA')
        w, h = img.size
        
        # Create isometric projection
        iso_w = int(w * 1.5)
        iso_h = int(h * 1.5)
        result = Image.new('RGBA', (iso_w, iso_h), (0, 0, 0, 0))
        
        # Apply simple skew transformation
        transform = QTransform()
        transform.shear(0.5, 0.0)  # Isometric skew
        
        # For now, return original with slight modification
        # Full 3D transformation would require more complex math
        return ImageEnhance.Brightness(img).enhance(0.8)
    
    def _process_edge_filter(self, image: Image.Image, steps: List[Dict], params: Dict) -> Image.Image:
        """Apply edge enhancement filter."""
        strength = params.get('strength', 1.0)
        threshold = params.get('threshold', 50)
        
        img = image.convert('RGBA')
        
        # Detect edges using PIL filters
        edges = img.filter(ImageFilter.FIND_EDGES)
        enhanced = ImageEnhance.Sharpness(img).enhance(1.0 + strength)
        
        # Blend with original
        from PIL import Image as PILImage
        result = PILImage.blend(img, enhanced, 0.5)
        return result

# Enhanced Canvas Widget
class ImageCanvas(QWidget):
    """Professional image display widget with zoom/pan capabilities."""
    
    imageClicked = pyqtSignal(int, int)  # x, y coordinates
    
    def __init__(self):
        super().__init__()
        self.image = None
        self.pixmap = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.last_pan_point = None
        
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        
        # Set background
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 45))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
    
    def set_image(self, image: Image.Image):
        """Set the image to display."""
        self.image = image
        self.update_pixmap()
        self.reset_view()
        self.update()
    
    def update_pixmap(self):
        """Convert PIL image to QPixmap."""
        if not self.image:
            self.pixmap = None
            return
            
        # Convert PIL to QPixmap
        img_array = np.array(self.image)
        if img_array.shape[2] == 4:  # RGBA
            h, w, ch = img_array.shape
            bytes_per_line = ch * w
            from PyQt6.QtGui import QImage
            qt_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        else:  # RGB
            h, w, ch = img_array.shape
            bytes_per_line = ch * w
            qt_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        self.pixmap = QPixmap.fromImage(qt_image)
    
    def reset_view(self):
        """Reset zoom and pan to fit image."""
        if not self.pixmap:
            return
            
        widget_size = self.size()
        image_size = self.pixmap.size()
        
        scale_x = widget_size.width() / image_size.width()
        scale_y = widget_size.height() / image_size.height()
        self.scale_factor = min(scale_x, scale_y, 1.0)
        
        self.offset_x = (widget_size.width() - image_size.width() * self.scale_factor) // 2
        self.offset_y = (widget_size.height() - image_size.height() * self.scale_factor) // 2
        
        self.update()
    
    def zoom_in(self):
        """Zoom in by 25%."""
        self.scale_factor *= 1.25
        self.update()
    
    def zoom_out(self):
        """Zoom out by 25%."""
        self.scale_factor *= 0.8
        self.update()
    
    def paintEvent(self, event):
        """Paint the image with current transform."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        
        if self.pixmap:
            # Draw checkerboard background for transparency
            self.draw_transparency_background(painter)
            
            # Apply transform and draw image
            transform = QTransform()
            transform.translate(self.offset_x, self.offset_y)
            transform.scale(self.scale_factor, self.scale_factor)
            
            painter.setTransform(transform)
            painter.drawPixmap(0, 0, self.pixmap)
        else:
            # Draw placeholder
            painter.setPen(QPen(QColor(128, 128, 128)))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded")
    
    def draw_transparency_background(self, painter):
        """Draw checkerboard pattern for transparent areas."""
        if not self.pixmap:
            return
            
        # Calculate visible area
        img_rect = QTransform().translate(self.offset_x, self.offset_y).scale(
            self.scale_factor, self.scale_factor).mapRect(self.pixmap.rect())
        
        # Draw checkerboard
        tile_size = 16
        painter.fillRect(img_rect, QColor(200, 200, 200))
        
        for y in range(int(img_rect.top()), int(img_rect.bottom()), tile_size):
            for x in range(int(img_rect.left()), int(img_rect.right()), tile_size):
                if ((x - int(img_rect.left())) // tile_size + 
                    (y - int(img_rect.top())) // tile_size) % 2:
                    painter.fillRect(x, y, tile_size, tile_size, QColor(220, 220, 220))
    
    def mousePressEvent(self, event):
        """Handle mouse press for panning and clicking."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_pan_point = event.pos()
        elif event.button() == Qt.MouseButton.RightButton:
            # Emit click coordinates in image space
            img_pos = self.widget_to_image_coords(event.pos())
            if img_pos:
                self.imageClicked.emit(int(img_pos.x()), int(img_pos.y()))
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for panning."""
        if self.dragging and self.last_pan_point:
            delta = event.pos() - self.last_pan_point
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_pan_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.last_pan_point = None
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def widget_to_image_coords(self, widget_pos):
        """Convert widget coordinates to image coordinates."""
        if not self.pixmap:
            return None
            
        # Inverse transform
        transform = QTransform()
        transform.translate(self.offset_x, self.offset_y)
        transform.scale(self.scale_factor, self.scale_factor)
        
        inv_transform, ok = transform.inverted()
        if ok:
            return inv_transform.map(widget_pos)
        return None

# Processing Thread
class ProcessingThread(QThread):
    """Thread for background image processing."""
    
    finished = pyqtSignal(object)  # PIL Image
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, image: Image.Image, processor: Callable, **kwargs):
        super().__init__()
        self.image = image
        self.processor = processor
        self.kwargs = kwargs
    
    def run(self):
        """Run processing in background thread."""
        try:
            result = self.processor(self.image, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# Plugin Interface Widget
class PluginWidget(QGroupBox):
    """Widget for plugin parameters and execution."""
    
    parameterChanged = pyqtSignal()
    
    def __init__(self, plugin: BasePlugin):
        super().__init__(plugin.get_info().name)
        self.plugin = plugin
        self.param_widgets = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup plugin parameter UI."""
        layout = QVBoxLayout(self)
        
        # Plugin info
        info = self.plugin.get_info()
        info_label = QLabel(f"v{info.version} - {info.description}")
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info_label)
        
        # Parameters
        params = self.plugin.get_parameters()
        for param_name, param_config in params.items():
            param_widget = self.create_parameter_widget(param_name, param_config)
            if param_widget:
                layout.addWidget(param_widget)
                
        # Apply button
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.parameterChanged.emit)
        layout.addWidget(apply_btn)
    
    def create_parameter_widget(self, name: str, config: Dict) -> Optional[QWidget]:
        """Create widget for parameter based on config."""
        param_type = config.get("type", "")
        
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(config.get("label", name))
        layout.addWidget(label)
        
        if param_type == "slider":
            widget = QSlider(Qt.Orientation.Horizontal)
            widget.setMinimum(int(config.get("min", 0)))
            widget.setMaximum(int(config.get("max", 100)))
            widget.setValue(int(config.get("default", 0)))
            self.param_widgets[name] = widget
            
            value_label = QLabel(str(widget.value()))
            widget.valueChanged.connect(lambda v: value_label.setText(str(v)))
            
            layout.addWidget(widget)
            layout.addWidget(value_label)
            
        elif param_type == "checkbox":
            widget = QCheckBox()
            widget.setChecked(config.get("default", False))
            self.param_widgets[name] = widget
            layout.addWidget(widget)
            
        else:
            return None
            
        return container
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        params = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, QSlider):
                params[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
        return params

# Main Application Window
class SpriteForgeMainWindow(QMainWindow):
    """Enhanced main application window."""
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.processed_image = None
        self.plugin_manager = PluginManager()
        self.processing_thread = None
        
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Enhanced Sprite Forge 2025")
        self.setMinimumSize(1200, 800)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: #ffffff; }
            QWidget { background-color: #2b2b2b; color: #ffffff; }
            QGroupBox { 
                font-weight: bold; 
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #666;
                padding: 5px;
                border-radius: 3px;
                min-width: 80px;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:pressed { background-color: #606060; }
        """)
        
        # Create menu bar
        self.create_menus()
        
        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Canvas and preview
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 800])
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_menus(self):
        """Create application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open Image", self.open_image)
        file_menu.addAction("Save Image", self.save_image)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("Reload Plugins", self.reload_plugins)
        tools_menu.addAction("Plugin Manager", self.show_plugin_manager)
        
        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Zoom In", self.zoom_in)
        view_menu.addAction("Zoom Out", self.zoom_out)
        view_menu.addAction("Reset Zoom", self.reset_zoom)
    
    def create_left_panel(self) -> QWidget:
        """Create left control panel."""
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        
        # File controls
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
        
        open_btn = QPushButton("Open Image")
        open_btn.clicked.connect(self.open_image)
        file_layout.addWidget(open_btn)
        
        self.file_label = QLabel("No image loaded")
        self.file_label.setStyleSheet("color: #999; font-style: italic;")
        file_layout.addWidget(self.file_label)
        
        layout.addWidget(file_group)
        
        # Basic sprite settings
        settings_group = QGroupBox("Sprite Settings")
        settings_layout = QGridLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Name:"), 0, 0)
        self.name_edit = QLineEdit("CSTM")
        self.name_edit.setMaxLength(4)
        settings_layout.addWidget(self.name_edit, 0, 1)
        
        settings_layout.addWidget(QLabel("Type:"), 1, 0)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Character", "Weapon", "Item", "Wall", "Flat", "Decoration"])
        settings_layout.addWidget(self.type_combo, 1, 1)
        
        settings_layout.addWidget(QLabel("Size:"), 2, 0)
        self.size_combo = QComboBox()
        self.size_combo.addItems(["Auto", "64", "128", "256", "512"])
        settings_layout.addWidget(self.size_combo, 2, 1)
        
        layout.addWidget(settings_group)
        
        # Processing options
        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout(process_group)
        
        self.pixelate_check = QCheckBox("Pixelate")
        self.pixelate_check.setChecked(True)
        process_layout.addWidget(self.pixelate_check)
        
        self.palette_check = QCheckBox("Apply Doom Palette")
        self.palette_check.setChecked(True)
        process_layout.addWidget(self.palette_check)
        
        self.transparency_check = QCheckBox("Preserve Transparency")
        self.transparency_check.setChecked(True)
        process_layout.addWidget(self.transparency_check)
        
        # Preview button
        preview_btn = QPushButton("Preview Changes")
        preview_btn.clicked.connect(self.preview_changes)
        process_layout.addWidget(preview_btn)
        
        layout.addWidget(process_group)
        
        # Plugin section
        plugin_group = QGroupBox("Plugins")
        plugin_layout = QVBoxLayout(plugin_group)
        
        self.plugin_scroll = QScrollArea()
        self.plugin_widget = QWidget()
        self.plugin_layout = QVBoxLayout(self.plugin_widget)
        self.plugin_scroll.setWidget(self.plugin_widget)
        self.plugin_scroll.setWidgetResizable(True)
        self.plugin_scroll.setMaximumHeight(300)
        
        plugin_layout.addWidget(self.plugin_scroll)
        self.load_plugin_widgets()
        
        layout.addWidget(plugin_group)
        
        # Export section
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        pk3_btn = QPushButton("Export PK3")
        pk3_btn.clicked.connect(self.export_pk3)
        export_layout.addWidget(pk3_btn)
        
        wad_btn = QPushButton("Export WAD Layout")
        wad_btn.clicked.connect(self.export_wad)
        export_layout.addWidget(wad_btn)
        
        layout.addWidget(export_group)
        
        # Stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create right panel with canvas and tabs."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different views
        tab_widget = QTabWidget()
        
        # Main canvas tab
        canvas_tab = QWidget()
        canvas_layout = QVBoxLayout(canvas_tab)
        
        # Toolbar for canvas
        canvas_toolbar = self.create_canvas_toolbar()
        canvas_layout.addWidget(canvas_toolbar)
        
        # Image canvas
        self.canvas = ImageCanvas()
        self.canvas.imageClicked.connect(self.on_canvas_clicked)
        canvas_layout.addWidget(self.canvas)
        
        tab_widget.addTab(canvas_tab, "Canvas")
        
        # Log tab
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit { 
                background-color: #1e1e1e; 
                color: #ffffff;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        tab_widget.addTab(log_tab, "Log")
        
        layout.addWidget(tab_widget)
        return panel
    
    def create_canvas_toolbar(self) -> QWidget:
        """Create toolbar for canvas controls."""
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Zoom controls
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(self.zoom_in)
        layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(self.zoom_out)
        layout.addWidget(zoom_out_btn)
        
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_zoom)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        
        # Image info
        self.image_info_label = QLabel("No image")
        self.image_info_label.setStyleSheet("color: #999;")
        layout.addWidget(self.image_info_label)
        
        return toolbar
    
    def load_plugin_widgets(self):
        """Load plugin widgets into the scroll area."""
        # Clear existing widgets
        for i in reversed(range(self.plugin_layout.count())):
            self.plugin_layout.itemAt(i).widget().setParent(None)
        
        # Add plugin widgets
        for plugin in self.plugin_manager.plugins.values():
            plugin_widget = PluginWidget(plugin)
            plugin_widget.parameterChanged.connect(
                lambda p=plugin, w=plugin_widget: self.apply_plugin(p, w.get_parameters())
            )
            self.plugin_layout.addWidget(plugin_widget)
        
        self.plugin_layout.addStretch()
    
    def log_message(self, message: str):
        """Add message to log."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # Also update status bar temporarily
        self.status_bar.showMessage(message, 3000)
    
    def update_image_info(self):
        """Update image information display."""
        if self.current_image:
            name = validate_sprite_name(self.name_edit.text())
            w, h = self.current_image.size
            info = f"{name} | {w}×{h} pixels"
            self.image_info_label.setText(info)
        else:
            self.image_info_label.setText("No image")
    
    # Event handlers
    def open_image(self):
        """Open image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*)"
        )
        
        if file_path:
            try:
                self.current_image = Image.open(file_path).convert('RGBA')
                self.canvas.set_image(self.current_image)
                self.file_label.setText(os.path.basename(file_path))
                self.update_image_info()
                self.log_message(f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {e}")
                self.log_message(f"Error loading image: {e}")
    
    def save_image(self):
        """Save current processed image."""
        if not self.processed_image:
            QMessageBox.warning(self, "Warning", "No processed image to save")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "PNG Files (*.png);;All Files (*)"
        )
        
        if file_path:
            try:
                self.processed_image.save(file_path)
                self.log_message(f"Saved: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {e}")
    
    def preview_changes(self):
        """Preview processing changes."""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
            
        try:
            processed = self.apply_basic_processing(self.current_image)
            self.processed_image = processed
            self.canvas.set_image(processed)
            self.log_message("Preview generated")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {e}")
            self.log_message(f"Processing error: {e}")
    
    def apply_basic_processing(self, image: Image.Image) -> Image.Image:
        """Apply basic sprite processing."""
        img = image.copy()
        
        # Resize if specified
        size_text = self.size_combo.currentText()
        if size_text != "Auto":
            new_size = int(size_text)
            img = img.resize((new_size, new_size), Image.Resampling.NEAREST)
        
        # Apply pixelation
        if self.pixelate_check.isChecked():
            img = pixelate_image(img)
        
        # Apply Doom palette
        if self.palette_check.isChecked():
            img = apply_doom_palette(img, self.transparency_check.isChecked())
        
        return img
    
    def apply_plugin(self, plugin: BasePlugin, parameters: Dict[str, Any]):
        """Apply plugin with given parameters."""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
            
        self.log_message(f"Applying plugin: {plugin.get_info().name}")
        
        # Start processing thread
        base_image = self.processed_image or self.current_image
        self.processing_thread = ProcessingThread(base_image, plugin.process_image, **parameters)
        self.processing_thread.finished.connect(self.on_plugin_finished)
        self.processing_thread.error.connect(self.on_plugin_error)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.processing_thread.start()
    
    def on_plugin_finished(self, result_image: Image.Image):
        """Handle plugin processing completion."""
        self.processed_image = result_image
        self.canvas.set_image(result_image)
        self.progress_bar.setVisible(False)
        self.log_message("Plugin processing completed")
    
    def on_plugin_error(self, error_message: str):
        """Handle plugin processing error."""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Plugin Error", f"Plugin processing failed: {error_message}")
        self.log_message(f"Plugin error: {error_message}")
    
    def export_pk3(self):
        """Export sprite as PK3 package."""
        if not (self.processed_image or self.current_image):
            QMessageBox.warning(self, "Warning", "No image to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export PK3", "",
            "PK3 Files (*.pk3);;All Files (*)"
        )
        
        if file_path:
            try:
                image = self.processed_image or self.current_image
                base_name = validate_sprite_name(self.name_edit.text())
                
                # Create rotations (simplified - just mirrors for demo)
                rotations = {}
                rotations['1'] = image.copy()
                rotations['2'] = image.copy()
                rotations['3'] = image.copy()
                rotations['4'] = image.copy()
                rotations['5'] = ImageOps.mirror(image)
                rotations['6'] = ImageOps.mirror(image)
                rotations['7'] = ImageOps.mirror(image)
                rotations['8'] = ImageOps.mirror(image)
                
                self.create_pk3_package(file_path, base_name, rotations)
                self.log_message(f"PK3 exported: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Success", f"PK3 package created:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {e}")
                self.log_message(f"Export error: {e}")
    
    def create_pk3_package(self, file_path: str, base_name: str, rotations: Dict[str, Image.Image]):
        """Create PK3 package file."""
        import io
        
        with zipfile.ZipFile(file_path, 'w', compression=zipfile.ZIP_DEFLATED) as pk3:
            # Add sprite PNGs
            for rot, img in rotations.items():
                sprite_name = f"{base_name}A{rot}"
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                pk3.writestr(f"sprites/{sprite_name}.png", buf.getvalue())
            
            # Add basic TEXTURES lump
            textures_content = f"""// Generated by Enhanced Sprite Forge 2025
// Sprite: {base_name}

"""
            for rot, img in rotations.items():
                sprite_name = f"{base_name}A{rot}"
                width, height = img.size
                textures_content += f"""sprite {sprite_name} {width} {height}
{{
    XOffset {width // 2}
    YOffset {height}
    Patch {sprite_name} 0 0
}}

"""
            pk3.writestr("TEXTURES", textures_content)
    
    def export_wad(self):
        """Export WAD layout directory."""
        if not (self.processed_image or self.current_image):
            QMessageBox.warning(self, "Warning", "No image to export")
            return
            
        folder_path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        
        if folder_path:
            try:
                image = self.processed_image or self.current_image
                base_name = validate_sprite_name(self.name_edit.text())
                
                wad_dir = os.path.join(folder_path, f"{base_name}_wad")
                os.makedirs(wad_dir, exist_ok=True)
                
                # Create rotations and save
                rotations = {}
                rotations['1'] = image.copy()
                rotations['5'] = ImageOps.mirror(image)
                
                info = {
                    "base_name": base_name,
                    "frames": [],
                    "generated_by": "Enhanced Sprite Forge 2025"
                }
                
                for rot, img in rotations.items():
                    sprite_name = f"{base_name}A{rot}"
                    filename = f"{sprite_name}.png"
                    img.save(os.path.join(wad_dir, filename))
                    
                    info["frames"].append({
                        "name": sprite_name,
                        "filename": filename,
                        "angle": int(rot),
                        "width": img.width,
                        "height": img.height
                    })
                
                # Save info file
                with open(os.path.join(wad_dir, f"{base_name}_info.json"), 'w') as f:
                    json.dump(info, f, indent=2)
                
                self.log_message(f"WAD layout created: {wad_dir}")
                QMessageBox.information(self, "Success", f"WAD layout created:\n{wad_dir}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {e}")
                self.log_message(f"Export error: {e}")
    
    # Canvas controls
    def zoom_in(self):
        """Zoom in on canvas."""
        self.canvas.zoom_in()
    
    def zoom_out(self):
        """Zoom out on canvas."""
        self.canvas.zoom_out()
    
    def reset_zoom(self):
        """Reset canvas zoom."""
        self.canvas.reset_view()
    
    def on_canvas_clicked(self, x: int, y: int):
        """Handle canvas click events."""
        self.log_message(f"Canvas clicked at ({x}, {y})")
    
    # Plugin management
    def reload_plugins(self):
        """Reload all plugins."""
        self.plugin_manager.load_plugins()
        self.load_plugin_widgets()
        self.log_message("Plugins reloaded")
    
    def show_plugin_manager(self):
        """Show plugin manager dialog."""
        # Simple plugin info dialog
        plugins = self.plugin_manager.list_plugins()
        info_text = "Loaded Plugins:\n\n"
        
        for plugin_info in plugins:
            info_text += f"• {plugin_info.name} v{plugin_info.version}\n"
            info_text += f"  {plugin_info.description}\n"
            info_text += f"  Author: {plugin_info.author}\n"
            info_text += f"  Category: {plugin_info.category}\n\n"
        
        QMessageBox.information(self, "Plugin Manager", info_text)
    
    # Settings
    def load_settings(self):
        """Load application settings."""
        settings = QSettings("SpriteForge", "EnhancedSpriteForge2025")
        
        # Restore window geometry
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restore other settings
        self.name_edit.setText(settings.value("sprite_name", "CSTM"))
        
        sprite_type = settings.value("sprite_type", "Character")
        index = self.type_combo.findText(sprite_type)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)
    
    def save_settings(self):
        """Save application settings."""
        settings = QSettings("SpriteForge", "EnhancedSpriteForge2025")
        
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("sprite_name", self.name_edit.text())
        settings.setValue("sprite_type", self.type_combo.currentText())
    
    def closeEvent(self, event):
        """Handle application close."""
        self.save_settings()
        super().closeEvent(event)


# Application Entry Point
def main():
    """Main application entry point."""
    # Handle command line arguments
    if len(sys.argv) > 1:
        # CLI mode - use original CLI functionality
        print("CLI mode not fully implemented in enhanced version")
        print("Use original sprite_forge_2025.py for CLI features")
        return 1
    
    if not GUI_AVAILABLE:
        print("PyQt6 not available. Please install: pip install PyQt6")
        return 1
    
    app = QApplication(sys.argv)
    app.setApplicationName("Enhanced Sprite Forge 2025")
    app.setOrganizationName("SpriteForge")
    
    # Set application icon (if available)
    # app.setWindowIcon(QIcon("icon.png"))
    
    window = SpriteForgeMainWindow()
    window.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
