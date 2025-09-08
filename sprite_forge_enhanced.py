#!/usr/bin/env python3
"""
Sprite Forge Enhanced - Premium Professional Sprite Creation Tool
A comprehensive sprite editor with advanced features, plugin system, and Doom-specific tools.
Combines all features from both sprite_forge_pro.py and sprite_forge_enhanced.py.
"""

import sys
import os
import json
import logging
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto
import traceback

# Core dependencies
try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageFont
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("PIL/Pillow not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow", "numpy"])
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageFont
    import numpy as np
    HAS_PIL = True

# GUI Framework
try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    print("PyQt6 not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt6"])
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    HAS_PYQT6 = True

# Optional advanced dependencies
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import skimage
    from skimage import filters, morphology, segmentation
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Prevent Qt conflicts
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import scipy
    from scipy import ndimage, signal
    HAS_SCIPY = False
except ImportError:
    HAS_SCIPY = False

# Application constants
__version__ = "3.0.0"
APP_NAME = "Sprite Forge Enhanced"
ORG_NAME = "SpriteForge"
APP_KEY = "sprite_forge_enhanced"
APP_DESCRIPTION = "Premium Professional Sprite Creation Tool"

# Enums
class SpriteType(Enum):
    STATIC = auto()
    ANIMATED = auto()
    ROTATIONAL = auto()
    SPRITE_SHEET = auto()

class ProcessingMode(Enum):
    REAL_TIME = auto()
    BATCH = auto()
    PREVIEW = auto()

class ExportFormat(Enum):
    PNG = auto()
    PK3 = auto()
    WAD = auto()
    GIF = auto()
    ZIP = auto()
    SPRITE_SHEET = auto()

# Data classes
@dataclass
class SpriteFrame:
    image: Image.Image
    delay: int = 100
    offset_x: int = 0
    offset_y: int = 0
    rotation: float = 0.0
    scale: float = 1.0

@dataclass
class SpriteAnimation:
    frames: List[SpriteFrame] = field(default_factory=list)
    loop: bool = True
    speed: float = 1.0
    name: str = ""

@dataclass
class PluginInfo:
    name: str
    version: str
    description: str
    author: str
    category: str
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProjectSettings:
    name: str = "Untitled"
    sprite_type: SpriteType = SpriteType.STATIC
    export_format: ExportFormat = ExportFormat.PNG
    doom_palette: bool = True
    auto_crop: bool = True
    background_removal: bool = False
    pixelation_factor: int = 1
    enhancement_level: float = 1.0

# Core Image Processing
class ImageProcessor:
    """Core image processing functionality for sprites."""
    
    @staticmethod
    def validate_sprite_name(name: str) -> bool:
        """Validate sprite name for Doom compatibility."""
        if not name or len(name) > 8:
            return False
        return all(c.isalnum() or c in '_' for c in name)
    
    @staticmethod
    def pixelate_image(image: Image.Image, factor: int) -> Image.Image:
        """Pixelate image by reducing resolution."""
        if factor <= 1:
            return image
        
        width, height = image.size
        new_width = width // factor
        new_height = height // factor
        
        # Resize down
        small = image.resize((new_width, new_height), Image.Resampling.NEAREST)
        # Resize back up
        return small.resize((width, height), Image.Resampling.NEAREST)
    
    @staticmethod
    def apply_doom_palette(image: Image.Image) -> Image.Image:
        """Apply Doom's color palette to image."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create Doom-like palette (simplified)
        doom_colors = [
            (0, 0, 0), (31, 23, 11), (23, 15, 7), (75, 75, 75),
            (255, 255, 255), (27, 27, 27), (47, 47, 47), (67, 67, 67),
            (87, 87, 87), (107, 107, 107), (127, 127, 127), (147, 147, 147),
            (167, 167, 167), (187, 187, 187), (207, 207, 207), (227, 227, 227)
        ]
        
        # Simple palette mapping
        result = Image.new('RGB', image.size)
        pixels = result.load()
        img_pixels = image.load()
        
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                r, g, b = img_pixels[x, y]
                # Find closest Doom color
                min_dist = float('inf')
                closest_color = (0, 0, 0)
                
                for doom_color in doom_colors:
                    dist = ((r - doom_color[0])**2 + 
                           (g - doom_color[1])**2 + 
                           (b - doom_color[2])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_color = doom_color
                
                pixels[x, y] = closest_color
        
        return result
    
    @staticmethod
    def create_sprite_rotations(image: Image.Image, angles: List[float]) -> List[Image.Image]:
        """Create rotated versions of sprite."""
        rotations = []
        for angle in angles:
            rotated = image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
            rotations.append(rotated)
        return rotations
    
    @staticmethod
    def enhance_sprite(image: Image.Image, brightness: float = 1.0, 
                      contrast: float = 1.0, saturation: float = 1.0) -> Image.Image:
        """Enhance sprite with brightness, contrast, and saturation adjustments."""
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
        
        return image
    
    @staticmethod
    def auto_crop(image: Image.Image, threshold: int = 10) -> Image.Image:
        """Automatically crop transparent/empty borders."""
        if image.mode == 'RGBA':
            # Get alpha channel
            alpha = image.split()[-1]
            bbox = alpha.getbbox()
        else:
            # Convert to RGBA for transparency check
            rgba = image.convert('RGBA')
            alpha = rgba.split()[-1]
            bbox = alpha.getbbox()
        
        if bbox:
            return image.crop(bbox)
        return image
    
    @staticmethod
    def remove_background(image: Image.Image, tolerance: int = 30) -> Image.Image:
        """Remove background based on color similarity."""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get background color (assume top-left pixel)
        bg_color = image.getpixel((0, 0))
        
        # Create mask
        mask = Image.new('L', image.size, 0)
        mask_data = mask.load()
        img_data = image.load()
        
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                pixel = img_data[x, y]
                # Calculate color distance
                dist = sum((a - b) ** 2 for a, b in zip(pixel, bg_color)) ** 0.5
                if dist > tolerance:
                    mask_data[x, y] = 255
        
        # Apply mask
        result = Image.new('RGBA', image.size, (0, 0, 0, 0))
        result.paste(image, mask=mask)
        return result

# Plugin System
class BasePlugin(ABC):
    """Abstract base class for all plugins."""
    
    def __init__(self, info: PluginInfo):
        self.info = info
    
    @abstractmethod
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        """Process image and return result."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get plugin parameters."""
        return self.info.parameters
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate plugin parameters."""
        return True

class JSONConfigPlugin(BasePlugin):
    """Plugin defined by JSON configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        info = PluginInfo(
            name=config.get('name', 'Unknown'),
            version=config.get('version', '1.0.0'),
            description=config.get('description', ''),
            author=config.get('author', 'Unknown'),
            category=config.get('category', 'General'),
            parameters=config.get('parameters', {})
        )
        super().__init__(info)
        self.config = config
        self.operation = config.get('operation', 'none')
    
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        """Process image based on JSON configuration."""
        if self.operation == 'pixelate':
            factor = kwargs.get('factor', 2)
            return ImageProcessor.pixelate_image(image, factor)
        elif self.operation == 'doom_palette':
            return ImageProcessor.apply_doom_palette(image)
        elif self.operation == 'enhance':
            brightness = kwargs.get('brightness', 1.0)
            contrast = kwargs.get('contrast', 1.0)
            saturation = kwargs.get('saturation', 1.0)
            return ImageProcessor.enhance_sprite(image, brightness, contrast, saturation)
        elif self.operation == 'auto_crop':
            threshold = kwargs.get('threshold', 10)
            return ImageProcessor.auto_crop(image, threshold)
        elif self.operation == 'background_removal':
            tolerance = kwargs.get('tolerance', 30)
            return ImageProcessor.remove_background(image, tolerance)
        else:
            return image

class PluginManager:
    """Manages plugin loading and execution."""
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.builtin_plugins = BuiltinPlugins()
        self._load_builtin_plugins()
    
    def _load_builtin_plugins(self):
        """Load built-in plugins."""
        for plugin in self.builtin_plugins.get_all():
            self.register_plugin(plugin)
    
    def register_plugin(self, plugin: BasePlugin):
        """Register a plugin."""
        self.plugins[plugin.info.name] = plugin
        logging.info(f"Registered plugin: {plugin.info.name} v{plugin.info.version}")
    
    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get plugin by name."""
        return self.plugins.get(name)
    
    def get_all_plugins(self) -> List[BasePlugin]:
        """Get all registered plugins."""
        return list(self.plugins.values())
    
    def get_plugins_by_category(self, category: str) -> List[BasePlugin]:
        """Get plugins by category."""
        return [p for p in self.plugins.values() if p.info.category == category]
    
    def load_plugins_from_directory(self, directory: str):
        """Load plugins from directory."""
        plugin_dir = Path(directory)
        if not plugin_dir.exists():
            return
        
        # Load Python plugins
        for py_file in plugin_dir.glob("*.py"):
            try:
                self._load_python_plugin(py_file)
            except Exception as e:
                logging.error(f"Failed to load Python plugin {py_file}: {e}")
        
        # Load JSON plugins
        for json_file in plugin_dir.glob("*.json"):
            try:
                self._load_json_plugin(json_file)
            except Exception as e:
                logging.error(f"Failed to load JSON plugin {json_file}: {e}")
    
    def _load_python_plugin(self, file_path: Path):
        """Load Python plugin from file."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("plugin", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for plugin class
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BasePlugin) and 
                attr != BasePlugin):
                plugin = attr()
                self.register_plugin(plugin)
                break
    
    def _load_json_plugin(self, file_path: Path):
        """Load JSON plugin from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        plugin = JSONConfigPlugin(config)
        self.register_plugin(plugin)

class BuiltinPlugins:
    """Built-in plugin implementations."""
    
    def get_all(self) -> List[BasePlugin]:
        """Get all built-in plugins."""
        return [
            PixelatePlugin(),
            DoomPalettePlugin(),
            EnhancePlugin(),
            AutoCropPlugin(),
            BackgroundRemovalPlugin()
        ]

class PixelatePlugin(BasePlugin):
    """Pixelation plugin."""
    
    def __init__(self):
        info = PluginInfo(
            name="Pixelate",
            version="2.0.0",
            description="Pixelate sprite for retro look",
            author="SpriteForge",
            category="Effects",
            parameters={"factor": {"type": "int", "min": 1, "max": 8, "default": 2}}
        )
        super().__init__(info)
    
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        factor = kwargs.get('factor', 2)
        return ImageProcessor.pixelate_image(image, factor)

class DoomPalettePlugin(BasePlugin):
    """Doom palette plugin."""
    
    def __init__(self):
        info = PluginInfo(
            name="Doom Palette",
            version="2.0.0",
            description="Apply Doom color palette",
            author="SpriteForge",
            category="Palette",
            parameters={}
        )
        super().__init__(info)
    
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        return ImageProcessor.apply_doom_palette(image)

class EnhancePlugin(BasePlugin):
    """Enhancement plugin."""
    
    def __init__(self):
        info = PluginInfo(
            name="Enhance",
            version="2.0.0",
            description="Enhance sprite quality",
            author="SpriteForge",
            category="Enhancement",
            parameters={
                "brightness": {"type": "float", "min": 0.1, "max": 3.0, "default": 1.0},
                "contrast": {"type": "float", "min": 0.1, "max": 3.0, "default": 1.0},
                "saturation": {"type": "float", "min": 0.0, "max": 3.0, "default": 1.0}
            }
        )
        super().__init__(info)
    
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        brightness = kwargs.get('brightness', 1.0)
        contrast = kwargs.get('contrast', 1.0)
        saturation = kwargs.get('saturation', 1.0)
        return ImageProcessor.enhance_sprite(image, brightness, contrast, saturation)

class AutoCropPlugin(BasePlugin):
    """Auto-crop plugin."""
    
    def __init__(self):
        info = PluginInfo(
            name="Auto Crop",
            version="2.0.0",
            description="Automatically crop transparent borders",
            author="SpriteForge",
            category="Utility",
            parameters={"threshold": {"type": "int", "min": 1, "max": 50, "default": 10}}
        )
        super().__init__(info)
    
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        threshold = kwargs.get('threshold', 10)
        return ImageProcessor.auto_crop(image, threshold)

class BackgroundRemovalPlugin(BasePlugin):
    """Background removal plugin."""
    
    def __init__(self):
        info = PluginInfo(
            name="Background Removal",
            version="2.0.0",
            description="Remove background color",
            author="SpriteForge",
            category="Utility",
            parameters={"tolerance": {"type": "int", "min": 10, "max": 100, "default": 30}}
        )
        super().__init__(info)
    
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        tolerance = kwargs.get('tolerance', 30)
        return ImageProcessor.remove_background(image, tolerance)

# GUI Components
class ModernImageCanvas(QWidget):
    """Advanced image display canvas with zoom, pan, and grid features."""
    
    zoomChanged = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.last_pan_pos = None
        self.show_grid = True
        self.show_pixel_grid = True
        self.show_transparency = True
        self.grid_size = 16
        self.pixel_grid_size = 1
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 300)
        
        # Context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    
    def set_image(self, image: Image.Image):
        """Set image to display."""
        if image:
            # Convert PIL image to QPixmap
            if image.mode == 'RGBA':
                qimage = QImage(image.tobytes(), image.width, image.height, 
                               image.width * 4, QImage.Format.Format_RGBA8888)
            else:
                qimage = QImage(image.tobytes(), image.width, image.height, 
                               image.width * 3, QImage.Format.Format_RGB888)
            
            self.image = QPixmap.fromImage(qimage)
            self.pan_offset = QPoint(0, 0)
            self.update()
    
    def set_zoom(self, factor: float):
        """Set zoom factor."""
        self.zoom_factor = max(0.1, min(10.0, factor))
        self.zoomChanged.emit(self.zoom_factor)
        self.update()
    
    def reset_view(self):
        """Reset zoom and pan to default."""
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.zoomChanged.emit(self.zoom_factor)
        self.update()
    
    def paintEvent(self, event: QPaintEvent):
        """Custom paint event."""
        painter = QPainter(self)
        try:
            painter.begin(self)
            
            # Fill background
            if self.show_transparency:
                self._draw_transparency_background(painter)
            else:
                painter.fillRect(self.rect(), QColor(50, 50, 50))
            
            if self.image:
                # Calculate scaled image
                scaled_size = self.image.size() * self.zoom_factor
                scaled_pixmap = self.image.scaled(scaled_size, Qt.AspectRatioMode.KeepAspectRatio, 
                                                Qt.TransformationMode.SmoothTransformation)
                
                # Calculate position
                x = (self.width() - scaled_pixmap.width()) // 2 + self.pan_offset.x()
                y = (self.height() - scaled_pixmap.height()) // 2 + self.pan_offset.y()
                
                # Draw image
                painter.drawPixmap(x, y, scaled_pixmap)
                
                # Draw grids
                if self.show_grid:
                    self._draw_grid(painter, x, y, scaled_pixmap.width(), scaled_pixmap.height())
                
                if self.show_pixel_grid and self.zoom_factor >= 4:
                    self._draw_pixel_grid(painter, x, y, scaled_pixmap.width(), scaled_pixmap.height())
            
            painter.end()
        except Exception as e:
            logging.error(f"Paint error: {e}")
            painter.end()
    
    def _draw_transparency_background(self, painter: QPainter):
        """Draw transparency checkerboard pattern."""
        size = 20
        for y in range(0, self.height(), size):
            for x in range(0, self.width(), size):
                if (x // size + y // size) % 2 == 0:
                    painter.fillRect(x, y, size, size, QColor(255, 255, 255))
                else:
                    painter.fillRect(x, y, size, size, QColor(200, 200, 200))
    
    def _draw_grid(self, painter: QPainter, x: int, y: int, width: int, height: int):
        """Draw grid lines."""
        painter.setPen(QPen(QColor(100, 100, 100, 100), 1, Qt.PenStyle.SolidLine))
        
        grid_size_scaled = int(self.grid_size * self.zoom_factor)
        
        # Vertical lines
        for i in range(0, width, grid_size_scaled):
            painter.drawLine(x + i, y, x + i, y + height)
        
        # Horizontal lines
        for i in range(0, height, grid_size_scaled):
            painter.drawLine(x, y + i, x + width, y + i)
    
    def _draw_pixel_grid(self, painter: QPainter, x: int, y: int, width: int, height: int):
        """Draw pixel grid."""
        painter.setPen(QPen(QColor(150, 150, 150, 80), 1, Qt.PenStyle.SolidLine))
        
        pixel_size_scaled = int(self.pixel_grid_size * self.zoom_factor)
        
        # Vertical lines
        for i in range(0, width, pixel_size_scaled):
            painter.drawLine(x + i, y, x + i, y + height)
        
        # Horizontal lines
        for i in range(0, height, pixel_size_scaled):
            painter.drawLine(x, y + i, x + width, y + i)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.last_pan_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events."""
        if self.last_pan_pos and self.image:
            delta = event.pos() - self.last_pan_pos
            self.pan_offset += delta
            self.last_pan_pos = event.pos()
            self.update()
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for zooming."""
        if self.image:
            delta = event.angleDelta().y()
            zoom_change = 1.1 if delta > 0 else 0.9
            new_zoom = self.zoom_factor * zoom_change
            self.set_zoom(new_zoom)
    
    def show_context_menu(self, pos: QPoint):
        """Show context menu."""
        menu = QMenu(self)
        
        reset_action = menu.addAction("Reset View")
        reset_action.triggered.connect(self.reset_view)
        
        menu.addSeparator()
        
        grid_action = menu.addAction("Show Grid")
        grid_action.setCheckable(True)
        grid_action.setChecked(self.show_grid)
        grid_action.triggered.connect(lambda: self._toggle_grid())
        
        pixel_grid_action = menu.addAction("Show Pixel Grid")
        pixel_grid_action.setCheckable(True)
        pixel_grid_action.setChecked(self.show_pixel_grid)
        pixel_grid_action.triggered.connect(lambda: self._toggle_pixel_grid())
        
        transparency_action = menu.addAction("Show Transparency")
        transparency_action.setCheckable(True)
        transparency_action.setChecked(self.show_transparency)
        transparency_action.triggered.connect(lambda: self._toggle_transparency())
        
        menu.exec(self.mapToGlobal(pos))
    
    def _toggle_grid(self):
        """Toggle grid visibility."""
        self.show_grid = not self.show_grid
        self.update()
    
    def _toggle_pixel_grid(self):
        """Toggle pixel grid visibility."""
        self.show_pixel_grid = not self.show_pixel_grid
        self.update()
    
    def _toggle_transparency(self):
        """Toggle transparency background."""
        self.show_transparency = not self.show_transparency
        self.update()

class AdvancedPluginWidget(QWidget):
    """Advanced widget for plugin interaction."""
    
    pluginApplied = pyqtSignal(str, dict)
    
    def __init__(self, plugin_manager: PluginManager, parent=None):
        super().__init__(parent)
        self.plugin_manager = plugin_manager
        self.current_plugin = None
        self.parameter_widgets = {}
        
        self.setup_ui()
        self.refresh_plugins()
    
    def setup_ui(self):
        """Setup user interface."""
        layout = QVBoxLayout(self)
        
        # Plugin selection
        plugin_group = QGroupBox("Plugin Selection")
        plugin_layout = QVBoxLayout(plugin_group)
        
        self.plugin_combo = QComboBox()
        self.plugin_combo.currentTextChanged.connect(self.on_plugin_changed)
        plugin_layout.addWidget(QLabel("Select Plugin:"))
        plugin_layout.addWidget(self.plugin_combo)
        
        layout.addWidget(plugin_group)
        
        # Parameters
        self.param_group = QGroupBox("Parameters")
        self.param_layout = QFormLayout(self.param_group)
        layout.addWidget(self.param_group)
        
        # Actions
        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout(action_group)
        
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview_plugin)
        action_layout.addWidget(self.preview_btn)
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_plugin)
        action_layout.addWidget(self.apply_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_parameters)
        action_layout.addWidget(self.reset_btn)
        
        layout.addWidget(action_group)
        
        # Plugin info
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        layout.addStretch()
    
    def refresh_plugins(self):
        """Refresh plugin list."""
        self.plugin_combo.clear()
        plugins = self.plugin_manager.get_all_plugins()
        
        for plugin in plugins:
            self.plugin_combo.addItem(plugin.info.name)
        
        if plugins:
            self.on_plugin_changed(plugins[0].info.name)
    
    def on_plugin_changed(self, plugin_name: str):
        """Handle plugin selection change."""
        self.current_plugin = self.plugin_manager.get_plugin(plugin_name)
        self.update_parameter_ui()
        self.update_info()
    
    def update_parameter_ui(self):
        """Update parameter UI for current plugin."""
        # Clear existing parameters
        for widget in self.parameter_widgets.values():
            widget.setParent(None)
        self.parameter_widgets.clear()
        
        if not self.current_plugin:
            return
        
        # Create parameter widgets
        for param_name, param_info in self.current_plugin.get_parameters().items():
            if isinstance(param_info, dict):
                param_type = param_info.get('type', 'string')
                default_value = param_info.get('default', '')
                min_val = param_info.get('min', None)
                max_val = param_info.get('max', None)
                
                if param_type == 'int':
                    widget = QSpinBox()
                    if min_val is not None:
                        widget.setMinimum(min_val)
                    if max_val is not None:
                        widget.setMaximum(max_val)
                    widget.setValue(default_value)
                elif param_type == 'float':
                    widget = QDoubleSpinBox()
                    if min_val is not None:
                        widget.setMinimum(min_val)
                    if max_val is not None:
                        widget.setMaximum(max_val)
                    widget.setValue(default_value)
                elif param_type == 'bool':
                    widget = QCheckBox()
                    widget.setChecked(default_value)
                else:
                    widget = QLineEdit()
                    widget.setText(str(default_value))
                
                self.param_layout.addRow(param_name, widget)
                self.parameter_widgets[param_name] = widget
    
    def update_info(self):
        """Update plugin information display."""
        if self.current_plugin:
            info = self.current_plugin.info
            text = f"<b>{info.name}</b> v{info.version}<br>"
            text += f"<i>{info.description}</i><br>"
            text += f"Author: {info.author}<br>"
            text += f"Category: {info.category}"
            self.info_label.setText(text)
        else:
            self.info_label.setText("No plugin selected")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        params = {}
        for param_name, widget in self.parameter_widgets.items():
            if isinstance(widget, QSpinBox):
                params[param_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[param_name] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[param_name] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                params[param_name] = widget.text()
        return params
    
    def preview_plugin(self):
        """Preview plugin effect."""
        if self.current_plugin:
            params = self.get_parameters()
            self.pluginApplied.emit("preview", params)
    
    def apply_plugin(self):
        """Apply plugin effect."""
        if self.current_plugin:
            params = self.get_parameters()
            self.pluginApplied.emit("apply", params)
    
    def reset_parameters(self):
        """Reset parameters to defaults."""
        self.update_parameter_ui()

class ExportManager:
    """Handles sprite export in various formats."""
    
    def __init__(self):
        self.supported_formats = {
            'PNG': self.export_png,
            'GIF': self.export_gif,
            'PK3': self.export_pk3,
            'WAD': self.export_wad,
            'ZIP': self.export_zip
        }
    
    def export_sprite(self, sprite: Image.Image, filename: str, format: str, **kwargs):
        """Export sprite in specified format."""
        if format.upper() in self.supported_formats:
            return self.supported_formats[format.upper()](sprite, filename, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_png(self, sprite: Image.Image, filename: str, **kwargs):
        """Export as PNG."""
        sprite.save(filename, 'PNG', **kwargs)
    
    def export_gif(self, sprite: Image.Image, filename: str, **kwargs):
        """Export as GIF."""
        sprite.save(filename, 'GIF', **kwargs)
    
    def export_pk3(self, sprite: Image.Image, filename: str, **kwargs):
        """Export as PK3 (ZIP with Doom structure)."""
        sprite_name = kwargs.get('sprite_name', 'SPRITE')
        if not ImageProcessor.validate_sprite_name(sprite_name):
            sprite_name = 'SPRITE'
        
        with zipfile.ZipFile(filename, 'w') as zf:
            # Add sprite to sprites directory
            sprite_path = f"sprites/{sprite_name}.png"
            sprite_bytes = self._image_to_bytes(sprite, 'PNG')
            zf.writestr(sprite_path, sprite_bytes)
            
            # Add metadata
            metadata = {
                'name': sprite_name,
                'width': sprite.width,
                'height': sprite.height,
                'format': 'PNG',
                'created_by': APP_NAME
            }
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))
    
    def export_wad(self, sprite: Image.Image, filename: str, **kwargs):
        """Export as WAD directory structure."""
        sprite_name = kwargs.get('sprite_name', 'SPRITE')
        if not ImageProcessor.validate_sprite_name(sprite_name):
            sprite_name = 'SPRITE'
        
        wad_dir = Path(filename).with_suffix('')
        wad_dir.mkdir(exist_ok=True)
        
        # Create sprites subdirectory
        sprites_dir = wad_dir / 'sprites'
        sprites_dir.mkdir(exist_ok=True)
        
        # Save sprite
        sprite_path = sprites_dir / f"{sprite_name}.png"
        sprite.save(sprite_path, 'PNG')
        
        # Create lump file
        lump_path = wad_dir / f"{sprite_name}.lmp"
        self._create_doom_lump(sprite, lump_path)
        
        # Create metadata
        metadata = {
            'name': sprite_name,
            'width': sprite.width,
            'height': sprite.height,
            'format': 'PNG',
            'created_by': APP_NAME,
            'lump_file': f"{sprite_name}.lmp"
        }
        
        meta_path = wad_dir / 'metadata.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def export_zip(self, sprite: Image.Image, filename: str, **kwargs):
        """Export as ZIP archive."""
        with zipfile.ZipFile(filename, 'w') as zf:
            sprite_bytes = self._image_to_bytes(sprite, 'PNG')
            zf.writestr('sprite.png', sprite_bytes)
            
            # Add metadata
            metadata = {
                'width': sprite.width,
                'height': sprite.height,
                'format': 'PNG',
                'created_by': APP_NAME
            }
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))
    
    def _image_to_bytes(self, image: Image.Image, format: str) -> bytes:
        """Convert PIL image to bytes."""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format)
        return buffer.getvalue()
    
    def _create_doom_lump(self, sprite: Image.Image, lump_path: Path):
        """Create Doom lump file."""
        # Simplified lump creation
        with open(lump_path, 'wb') as f:
            # Write lump header
            f.write(b'SPRITE')  # Magic
            f.write(sprite.width.to_bytes(2, 'little'))  # Width
            f.write(sprite.height.to_bytes(2, 'little'))  # Height
            
            # Write image data (simplified)
            if sprite.mode == 'RGBA':
                data = sprite.tobytes()
            else:
                data = sprite.convert('RGBA').tobytes()
            
            f.write(data)

class SpriteForgeMainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.plugin_manager = PluginManager()
        self.export_manager = ExportManager()
        self.current_image = None
        self.original_image = None
        self.undo_stack = []
        self.redo_stack = []
        self.settings = QSettings(ORG_NAME, APP_KEY)
        
        self.setup_ui()
        self.setup_menus()
        self.setup_toolbars()
        self.setup_statusbar()
        self.load_settings()
        self.load_plugins()
        
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.resize(1200, 800)
    
    def setup_ui(self):
        """Setup main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Tools and plugins
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        self.open_btn = QPushButton("Open Image")
        self.open_btn.clicked.connect(self.open_image)
        file_layout.addWidget(self.open_btn)
        
        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        file_layout.addWidget(self.save_btn)
        
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export_image)
        self.export_btn.setEnabled(False)
        file_layout.addWidget(self.export_btn)
        
        left_layout.addWidget(file_group)
        
        # Plugin panel
        self.plugin_widget = AdvancedPluginWidget(self.plugin_manager)
        self.plugin_widget.pluginApplied.connect(self.handle_plugin)
        left_layout.addWidget(self.plugin_widget)
        
        # Quick tools
        tools_group = QGroupBox("Quick Tools")
        tools_layout = QVBoxLayout(tools_group)
        
        self.pixelate_btn = QPushButton("Pixelate")
        self.pixelate_btn.clicked.connect(lambda: self.apply_quick_tool('pixelate'))
        tools_layout.addWidget(self.pixelate_btn)
        
        self.doom_palette_btn = QPushButton("Doom Palette")
        self.doom_palette_btn.clicked.connect(lambda: self.apply_quick_tool('doom_palette'))
        tools_layout.addWidget(self.doom_palette_btn)
        
        self.enhance_btn = QPushButton("Enhance")
        self.enhance_btn.clicked.connect(lambda: self.apply_quick_tool('enhance'))
        tools_layout.addWidget(self.enhance_btn)
        
        self.auto_crop_btn = QPushButton("Auto Crop")
        self.auto_crop_btn.clicked.connect(lambda: self.apply_quick_tool('auto_crop'))
        tools_layout.addWidget(self.auto_crop_btn)
        
        left_layout.addWidget(tools_group)
        
        # Undo/Redo
        undo_group = QGroupBox("History")
        undo_layout = QHBoxLayout(undo_group)
        
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo)
        self.undo_btn.setEnabled(False)
        undo_layout.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.redo)
        self.redo_btn.setEnabled(False)
        undo_layout.addWidget(self.redo_btn)
        
        left_layout.addWidget(undo_group)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel, 1)
        
        # Center panel - Image canvas
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        
        # Canvas controls
        canvas_controls = QHBoxLayout()
        
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 1000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setSuffix("%")
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        canvas_controls.addWidget(QLabel("Zoom:"))
        canvas_controls.addWidget(self.zoom_slider)
        
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_canvas_view)
        canvas_controls.addWidget(self.reset_view_btn)
        
        center_layout.addLayout(canvas_controls)
        
        # Main canvas
        self.canvas = ModernImageCanvas()
        center_layout.addWidget(self.canvas)
        
        main_layout.addWidget(center_panel, 3)
        
        # Right panel - Properties and preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Image properties
        self.properties_group = QGroupBox("Image Properties")
        self.properties_layout = QFormLayout(self.properties_group)
        
        self.width_label = QLabel("0")
        self.properties_layout.addRow("Width:", self.width_label)
        
        self.height_label = QLabel("0")
        self.properties_layout.addRow("Height:", self.height_label)
        
        self.mode_label = QLabel("None")
        self.properties_layout.addRow("Mode:", self.mode_label)
        
        right_layout.addWidget(self.properties_group)
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QFormLayout(export_group)
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(['PNG', 'GIF', 'PK3', 'WAD', 'ZIP'])
        export_layout.addRow("Format:", self.export_format_combo)
        
        self.sprite_name_edit = QLineEdit("SPRITE")
        export_layout.addRow("Sprite Name:", self.sprite_name_edit)
        
        right_layout.addWidget(export_group)
        
        right_layout.addStretch()
        main_layout.addWidget(right_panel, 1)
    
    def setup_menus(self):
        """Setup application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        
        export_action = QAction("&Export...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        reset_action = QAction("&Reset Image", self)
        reset_action.triggered.connect(self.reset_image)
        edit_menu.addAction(reset_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        preferences_action = QAction("&Preferences...", self)
        preferences_action.triggered.connect(self.show_preferences)
        tools_menu.addAction(preferences_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_toolbars(self):
        """Setup application toolbars."""
        toolbar = self.addToolBar("Main Toolbar")
        
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_image)
        toolbar.addAction(save_action)
        
        export_action = QAction("Export", self)
        export_action.triggered.connect(self.export_image)
        toolbar.addAction(export_action)
    
    def setup_statusbar(self):
        """Setup application status bar."""
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Ready")
    
    def load_settings(self):
        """Load application settings."""
        # Load window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Load other settings
        self.export_format_combo.setCurrentText(
            self.settings.value("export_format", "PNG")
        )
        self.sprite_name_edit.setText(
            self.settings.value("sprite_name", "SPRITE")
        )
    
    def save_settings(self):
        """Save application settings."""
        # Save window geometry
        self.settings.setValue("geometry", self.saveGeometry())
        
        # Save other settings
        self.settings.setValue("export_format", self.export_format_combo.currentText())
        self.settings.setValue("sprite_name", self.sprite_name_edit.text())
    
    def load_plugins(self):
        """Load plugins from directories."""
        # Load from local plugins directory
        local_plugins = Path("plugins")
        if local_plugins.exists():
            self.plugin_manager.load_plugins_from_directory(str(local_plugins))
        
        # Load from user plugins directory
        user_plugins = Path.home() / ".sprite_forge" / "plugins"
        if user_plugins.exists():
            self.plugin_manager.load_plugins_from_directory(str(user_plugins))
    
    def open_image(self):
        """Open image file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
        )
        
        if filename:
            try:
                image = Image.open(filename)
                self.set_current_image(image)
                self.statusbar.showMessage(f"Opened: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open image: {e}")
    
    def save_image(self):
        """Save current image."""
        if not self.current_image:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", 
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if filename:
            try:
                self.current_image.save(filename)
                self.statusbar.showMessage(f"Saved: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {e}")
    
    def export_image(self):
        """Export image in selected format."""
        if not self.current_image:
            return
        
        format_name = self.export_format_combo.currentText()
        sprite_name = self.sprite_name_edit.text()
        
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Export as {format_name}", f"{sprite_name}.{format_name.lower()}",
            f"{format_name} Files (*.{format_name.lower()})"
        )
        
        if filename:
            try:
                self.export_manager.export_sprite(
                    self.current_image, filename, format_name,
                    sprite_name=sprite_name
                )
                self.statusbar.showMessage(f"Exported: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {e}")
    
    def set_current_image(self, image: Image.Image):
        """Set current image and update UI."""
        # Save to undo stack
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            self.undo_btn.setEnabled(True)
        
        self.current_image = image
        self.original_image = image.copy()
        
        # Update canvas
        self.canvas.set_image(image)
        
        # Update properties
        self.width_label.setText(str(image.width))
        self.height_label.setText(str(image.height))
        self.mode_label.setText(image.mode)
        
        # Enable buttons
        self.save_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        # Clear redo stack
        self.redo_stack.clear()
        self.redo_btn.setEnabled(False)
    
    def handle_plugin(self, action: str, params: dict):
        """Handle plugin application."""
        if not self.current_image:
            return
        
        plugin_name = self.plugin_widget.current_plugin.info.name
        plugin = self.plugin_manager.get_plugin(plugin_name)
        
        if not plugin:
            return
        
        try:
            if action == "preview":
                # Create preview
                preview_image = plugin.process(self.current_image, **params)
                self.canvas.set_image(preview_image)
            elif action == "apply":
                # Apply effect
                result_image = plugin.process(self.current_image, **params)
                self.set_current_image(result_image)
                self.statusbar.showMessage(f"Applied {plugin_name}")
        except Exception as e:
            QMessageBox.critical(self, "Plugin Error", f"Plugin failed: {e}")
    
    def apply_quick_tool(self, tool_name: str):
        """Apply quick tool effect."""
        if not self.current_image:
            return
        
        try:
            if tool_name == 'pixelate':
                result = ImageProcessor.pixelate_image(self.current_image, 2)
            elif tool_name == 'doom_palette':
                result = ImageProcessor.apply_doom_palette(self.current_image)
            elif tool_name == 'enhance':
                result = ImageProcessor.enhance_sprite(self.current_image, 1.2, 1.1, 1.0)
            elif tool_name == 'auto_crop':
                result = ImageProcessor.auto_crop(self.current_image)
            else:
                return
            
            self.set_current_image(result)
            self.statusbar.showMessage(f"Applied {tool_name}")
        except Exception as e:
            QMessageBox.critical(self, "Tool Error", f"Tool failed: {e}")
    
    def undo(self):
        """Undo last action."""
        if self.undo_stack:
            # Save current to redo stack
            self.redo_stack.append(self.current_image.copy())
            self.redo_btn.setEnabled(True)
            
            # Restore previous image
            self.current_image = self.undo_stack.pop()
            self.canvas.set_image(self.current_image)
            
            # Update undo button state
            self.undo_btn.setEnabled(len(self.undo_stack) > 0)
    
    def redo(self):
        """Redo last undone action."""
        if self.redo_stack:
            # Save current to undo stack
            self.undo_stack.append(self.current_image.copy())
            self.undo_btn.setEnabled(True)
            
            # Restore redo image
            self.current_image = self.redo_stack.pop()
            self.canvas.set_image(self.current_image)
            
            # Update redo button state
            self.redo_btn.setEnabled(len(self.redo_stack) > 0)
    
    def reset_image(self):
        """Reset image to original."""
        if self.original_image:
            self.set_current_image(self.original_image.copy())
            self.statusbar.showMessage("Image reset to original")
    
    def reset_canvas_view(self):
        """Reset canvas view."""
        self.canvas.reset_view()
        self.zoom_slider.setValue(100)
    
    def on_zoom_changed(self, value: int):
        """Handle zoom slider change."""
        zoom_factor = value / 100.0
        self.canvas.set_zoom(zoom_factor)
    
    def show_preferences(self):
        """Show preferences dialog."""
        QMessageBox.information(self, "Preferences", "Preferences dialog not implemented yet.")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, f"About {APP_NAME}", 
                         f"{APP_NAME} v{__version__}\n\n{APP_DESCRIPTION}")
    
    def closeEvent(self, event):
        """Handle application close event."""
        self.save_settings()
        event.accept()

def main():
    """Main application entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--version':
            print(f"{APP_NAME} v{__version__}")
            return
        elif sys.argv[1] == '--help':
            print(f"Usage: {sys.argv[0]} [--version|--help]")
            print(f"  --version: Show version information")
            print(f"  --help: Show this help message")
            return
    
    # Create and run application
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    
    window = SpriteForgeMainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
