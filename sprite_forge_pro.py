#!/usr/bin/env python3
"""
Sprite Forge Pro 2025 v3.0.0 - The Ultimate Professional Doom Sprite Creation Suite
=====================================================================================

A state-of-the-art sprite and texture creation toolkit designed for professional Doom modding
with enterprise-grade features and user experience that rivals commercial tools.
"""

# Version and metadata
__version__ = "3.0.0"
__author__ = "Sprite Forge Team (Refactored by Jules)"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 Sprite Forge Team"

APP_NAME = "Sprite Forge Pro 2025"
ORG_NAME = "SpriteForge"
APP_KEY = "SpriteForgePro2025"
APP_DESCRIPTION = "The Ultimate Professional Doom Sprite Creation Suite"

# Core imports
import sys, os, json, time, argparse, logging, subprocess, zipfile, shutil, traceback, threading, queue, sqlite3, hashlib, base64, urllib.request, urllib.parse, inspect, yaml
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, NamedTuple, Set
from abc import ABC, abstractmethod
from enum import Enum, auto
import colorsys, math, random, re, tempfile, configparser, pickle, gzip, csv

# Dependency auto-installer
def ensure_dependencies():
    dependencies = {'pillow': 'PIL', 'numpy': 'numpy', 'PyQt6': 'PyQt6', 'requests': 'requests', 'opencv-python': 'cv2', 'scikit-image': 'skimage', 'matplotlib': 'matplotlib', 'scipy': 'scipy', 'PyYAML': 'yaml', 'pillow-avif-plugin': 'pillow_avif'}
    missing = [pkg for pkg, mod in dependencies.items() if not __import__(mod, fromlist=['']) is None]
    if missing:
        print(f"Installing required packages: {', '.join(missing)}")
        for package in missing:
            try: subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--user"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e: print(f"Warning: Failed to install {package}: {e}")

if not os.environ.get('SPRITE_FORGE_NO_AUTO_INSTALL'): ensure_dependencies()

try: import numpy as np
except ImportError: print("NumPy not available."); np = None
try: from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw, ImageFont
except ImportError: print("Pillow not available."); Image = None
try: import requests
except ImportError: print("Requests not available."); requests = None
try: import cv2
except ImportError: print("OpenCV not available."); cv2 = None
try: import skimage; HAS_SKIMAGE = True
except ImportError: HAS_SKIMAGE = False; print("Scikit-image not available.")
try: import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; HAS_MATPLOTLIB = True
except ImportError: HAS_MATPLOTLIB = False; print("Matplotlib not available.")
try: import scipy; from scipy.spatial import KDTree; from scipy.ndimage import distance_transform_edt; HAS_SCIPY = True
except ImportError: HAS_SCIPY = False; print("SciPy not available."); KDTree = None; distance_transform_edt = None
GUI_AVAILABLE = False
try:
    from PyQt6.QtWidgets import *; from PyQt6.QtCore import *; from PyQt6.QtGui import *
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"PyQt6 not fully available: {e}")
    class QObject: pass
    class pyqtSignal:
        def __init__(self, *args): pass
        def connect(self, *args): pass
        def emit(self, *args): pass

def setup_logging(level=logging.INFO, log_file=None):
    log_dir = Path.home() / ".sprite_forge" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if not log_file: log_file = log_dir / f"sprite_forge_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)
logger = setup_logging()

ENHANCED_DOOM_PALETTE = [(0,0,0),(31,23,11),(23,15,7),(75,75,75),(255,255,255),(27,27,27),(19,19,19),(11,11,11),(199,199,199),(119,119,119),(83,83,83),(47,47,47),(255,155,0),(231,119,0),(203,91,0),(175,71,0),(143,59,0),(119,47,0),(91,35,0),(71,27,0),(199,0,0),(167,0,0),(139,0,0),(107,0,0),(75,0,0),(0,255,0),(0,231,0),(0,203,0),(0,175,0),(0,143,0),(0,119,0),(0,91,0)] + [(i*255//224,i*255//224,i*255//224) for i in range(224)]
ENHANCED_DOOM_PALETTE.extend([(139,69,19),(160,82,45),(210,180,140),(222,184,135),(128,0,128),(147,0,211),(138,43,226),(75,0,130),(0,100,0),(34,139,34),(0,128,0),(50,205,50),(0,0,139),(0,0,205),(65,105,225),(70,130,180),(255,20,147),(199,21,133),(219,112,147),(255,182,193)])

class SpriteType(Enum): MONSTER = ("Monster", "Moving enemy sprites"); WEAPON = ("Weapon", "First-person weapon sprites"); PROJECTILE = ("Projectile", "Bullets, rockets, etc."); ITEM = ("Item", "Pickups and powerups"); DECORATION = ("Decoration", "Static objects"); EFFECT = ("Effect", "Explosions, smoke, etc."); PLAYER = ("Player", "Player character sprites"); CUSTOM = ("Custom", "User-defined sprite type")
class ProcessingMode(Enum): FAST, BALANCED, QUALITY, PROFESSIONAL = range(4)
class ExportFormat(Enum): PNG = ("PNG", ".png"); GIF = ("GIF", ".gif"); ZIP = ("ZIP", ".zip"); WAD = ("WAD", ".wad"); PK3 = ("PK3", ".pk3"); WEBP = ("WebP", ".webp"); AVIF = ("AVIF", ".avif"); SPRITE_SHEET_JSON = ("Sprite Sheet (JSON)", ".json"); SPRITE_SHEET_TXT = ("Sprite Sheet (TXT)", ".txt")

@dataclass
class SpriteFrame:
    name: str; image: Image.Image; angle: Optional[float] = None; offset_x: int = 0; offset_y: int = 0; duration: float = 0.1; metadata: Dict[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        if self.image and not self.offset_x and not self.offset_y: self.offset_x, self.offset_y = self.image.width // 2, self.image.height
@dataclass
class Sprite:
    name: str; frames: Dict[str, SpriteFrame] = field(default_factory=dict); metadata: Dict[str, Any] = field(default_factory=dict)
    def add_frame(self, frame: SpriteFrame): self.frames[frame.name] = frame
@dataclass
class SpriteAnimation: name: str; frames: List[SpriteFrame]; loop: bool = True; fps: float = 10.0
@dataclass
class Project:
    settings: 'ProjectSettings'; sprites: Dict[str, Sprite] = field(default_factory=dict); animations: Dict[str, SpriteAnimation] = field(default_factory=dict); metadata: Dict[str, Any] = field(default_factory=dict)
    def add_sprite(self, sprite: Sprite): self.sprites[sprite.name] = sprite
    def add_animation(self, anim: SpriteAnimation): self.animations[anim.name] = anim
@dataclass
class ProjectSettings:
    name: str="New Project"; author: str=""; theme: str="dark"; force_power_of_two: bool = True; sprite_sheet_size: Tuple[int, int] = (1024, 1024); sprite_padding: int = 2; apply_bleed: bool = True; scale_variants: List[float] = field(default_factory=lambda: [1.0])
    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f: json.dump({k: (v.name if isinstance(v, Enum) else v) for k, v in self.__dict__.items()}, f, indent=2)
    @classmethod
    def load(cls) -> 'ProjectSettings':
        s = cls(); s._update_from_file(Path.home()/".sprite_forge"/"settings.json"); s._update_from_file(Path('config.yaml')); s._update_from_env(); return s
    def _update_from_file(self, path: Path):
        if not path.exists(): return
        try:
            with open(path, 'r', encoding='utf-8') as f: data = yaml.safe_load(f) if path.suffix in ['.yaml', '.yml'] else json.load(f)
            if data: self._update_from_dict(data)
        except Exception as e: logger.warning(f"Failed to load settings from {path}: {e}")
    def _update_from_dict(self, data: Dict[str, Any]):
        for k, v in data.items():
            if hasattr(self, k): self._set_value(k, v)
    def _update_from_env(self):
        for k, v in os.environ.items():
            if k.startswith("SPRITEFORGE_"):
                field = k[len("SPRITEFORGE_"):].lower()
                if hasattr(self, field): self._set_value(field, v)
    def _set_value(self, key: str, value: Any):
        try:
            attr_type = type(getattr(self, key));
            if attr_type == bool: value = str(value).lower() in ('true', '1', 'yes')
            elif isinstance(getattr(self, key), Enum): value = type(getattr(self, key))[value.upper()]
            else: value = attr_type(value)
            setattr(self, key, value)
        except (ValueError, KeyError, TypeError) as e: logger.warning(f"Could not set setting '{key}' from config/env: {e}")

class ImageProcessor:
    def __init__(self): self.doom_palette_rgb, self.palette_kdtree = self._get_doom_palette()
    def _get_doom_palette(self):
        if np is None: return None, None
        palette = np.array(ENHANCED_DOOM_PALETTE, dtype=np.uint8); return palette, KDTree(palette) if KDTree else None
    def apply_doom_palette(self, image: Image.Image, perceptual: bool = True) -> Image.Image:
        if not image or not self.palette_kdtree or not perceptual: return image
        src = image.convert('RGBA'); pixels = np.array(src); rgb_pixels = pixels[:, :, :3].reshape(-1, 3);
        _, indices = self.palette_kdtree.query(rgb_pixels); new_rgb = self.doom_palette_rgb[indices].reshape(pixels.shape[:2] + (3,))
        new_pixels = np.concatenate([new_rgb, pixels[:, :, 3:]], axis=2); new_pixels[pixels[:, :, 3] == 0] = [0,0,0,0]
        return Image.fromarray(new_pixels.astype('uint8'), 'RGBA')
    @staticmethod
    def create_nine_patch(image: Image.Image, top: int, left: int, right: int, bottom: int) -> Image.Image:
        w, h = image.size; nine_patch = Image.new('RGBA', (w + 2, h + 2), (0,0,0,0)); nine_patch.paste(image, (1, 1)); draw = ImageDraw.Draw(nine_patch)
        draw.line([(left + 1, 0), (right + 1, 0)], fill=(0,0,0,255)); draw.line([(0, top + 1), (0, bottom + 1)], fill=(0,0,0,255)); return nine_patch
    @staticmethod
    def add_padding(image: Image.Image, padding: int) -> Image.Image:
        return ImageOps.expand(image, border=padding, fill=(0,0,0,0)) if padding > 0 else image
    @staticmethod
    def apply_bleed(image: Image.Image, bleed_size: int) -> Image.Image:
        if bleed_size <= 0 or not np or not HAS_SCIPY: return image
        img_array = np.array(image.convert('RGBA')); alpha = img_array[:,:,3]
        if np.all(alpha > 0): return image
        mask = alpha == 0; indices = distance_transform_edt(mask, return_indices=True, return_distances=False)
        bled_array = img_array[tuple(indices)]; return Image.fromarray(bled_array)
    @staticmethod
    def scale_image(image: Image.Image, scale_factor: float) -> Image.Image:
        if scale_factor == 1.0: return image
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    @staticmethod
    def premultiply_alpha(image: Image.Image) -> Image.Image:
        if not image.mode == 'RGBA' or not np: return image
        arr = np.array(image); alpha = arr[:,:,3] / 255.0
        for i in range(3): arr[:,:,i] = arr[:,:,i] * alpha
        return Image.fromarray(arr)
    @staticmethod
    def flatten_background(image: Image.Image, background_color: Tuple[int,int,int]) -> Image.Image:
        if not image.mode == 'RGBA': return image
        background = Image.new('RGB', image.size, background_color)
        background.paste(image, mask=image.split()[3]); return background
    @staticmethod
    def apply_color_key(image: Image.Image, color_key: Tuple[int,int,int], tolerance: int=10) -> Image.Image:
        img = image.convert("RGBA"); datas = img.getdata(); newData = []
        for item in datas:
            if math.sqrt((item[0]-color_key[0])**2 + (item[1]-color_key[1])**2 + (item[2]-color_key[2])**2) < tolerance:
                newData.append((255, 255, 255, 0))
            else: newData.append(item)
        img.putdata(newData); return img
    @staticmethod
    def adjust_brightness_contrast(image: Image.Image, brightness: float = 1.0, contrast: float = 1.0) -> Image.Image:
        enhancer = ImageEnhance.Brightness(image); image = enhancer.enhance(brightness)
        enhancer = ImageEnhance.Contrast(image); image = enhancer.enhance(contrast)
        return image
    @staticmethod
    def adjust_hsl(image: Image.Image, hue: float=0, saturation: float=1.0, lightness: float=1.0) -> Image.Image:
        if not np: return image
        img = image.convert('RGBA'); arr = np.array(img, dtype=float) / 255.0
        h, l, s = np.vectorize(colorsys.rgb_to_hls)(arr[:,:,0], arr[:,:,1], arr[:,:,2])
        h = (h + hue) % 1.0; s = np.clip(s * saturation, 0, 1); l = np.clip(l * lightness, 0, 1)
        r, g, b = np.vectorize(colorsys.hls_to_rgb)(h, l, s)
        arr[:,:,0]=r; arr[:,:,1]=g; arr[:,:,2]=b
        return Image.fromarray((arr * 255).astype(np.uint8), 'RGBA')
    @staticmethod
    def convert_to_grayscale(image: Image.Image, weights: Tuple[float, float, float] = (0.299, 0.587, 0.114)) -> Image.Image:
        return ImageOps.grayscale(image.convert('RGB')).convert('RGBA')
    @staticmethod
    def replace_color(image: Image.Image, find_color: Tuple[int,int,int], replace_color: Tuple[int,int,int], tolerance: int=10) -> Image.Image:
        img = image.convert("RGBA"); datas = img.getdata(); newData = []
        for item in datas:
            if math.sqrt((item[0]-find_color[0])**2 + (item[1]-find_color[1])**2 + (item[2]-find_color[2])**2) < tolerance:
                newData.append(replace_color + (item[3],))
            else: newData.append(item)
        img.putdata(newData); return img
    @staticmethod
    def create_glow_or_outline(image: Image.Image, color: Tuple[int,int,int], size: int, mode: str='outline') -> Image.Image:
        if not np or not HAS_SCIPY: return image
        img_arr = np.array(image.convert('RGBA')); alpha = img_arr[:,:,3]
        mask = alpha > 0; filled_mask = distance_transform_edt(np.invert(mask)) <= size
        glow_img = Image.new('RGBA', image.size, color + (255,))
        glow_mask = Image.fromarray((filled_mask.astype('uint8') * 255))
        if mode == 'outline':
            final_img = Image.new('RGBA', image.size); final_img.paste(glow_img, (0,0), mask=glow_mask); final_img.paste(image, (0,0), mask=image)
        elif mode == 'glow':
            blurred_mask = glow_mask.filter(ImageFilter.GaussianBlur(size / 2))
            final_img = Image.new('RGBA', image.size); final_img.paste(glow_img, (0,0), mask=blurred_mask); final_img.paste(image, (0,0), mask=image)
        else: # background
            final_img = image.copy(); final_img.paste(glow_img, (0,0), mask=glow_mask); final_img.paste(image, (0,0), mask=image)
        return final_img

class PluginInfo:
    def __init__(self, name: str, **kwargs): self.name=name; self.version=kwargs.get('v','1.0'); self.author=kwargs.get('a','?'); self.description=kwargs.get('d',''); self.category=kwargs.get('c','General')
class BasePlugin(ABC):
    info: PluginInfo
    def __init__(self, core: 'CoreEngine'): self.core = core
    @abstractmethod
    def run(self, **kwargs): pass
    def get_parameters(self) -> Dict[str, Any]: return {}

class PluginManager:
    def __init__(self, core: 'CoreEngine'):
        self.core = core
        self.plugins: Dict[str, BasePlugin] = {}
    def discover_plugins(self, plugin_dir: Path):
        if not plugin_dir.is_dir(): return
        sys.path.insert(0, str(plugin_dir.parent))
        for file in plugin_dir.glob('*.py'):
            module_name = f"{plugin_dir.name}.{file.stem}"
            try:
                module = __import__(module_name, fromlist=['*'])
                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and issubclass(obj, BasePlugin) and obj is not BasePlugin:
                        plugin_instance = obj(self.core)
                        self.plugins[plugin_instance.info.name] = plugin_instance
                        logger.info(f"Discovered plugin: {plugin_instance.info.name}")
            except Exception as e:
                logger.warning(f"Failed to load plugin from {file}: {e}")
        sys.path.pop(0)
    def run_plugin(self, name:str, **kwargs):
        plugin = self.plugins.get(name)
        if plugin:
            try: plugin.run(**kwargs)
            except Exception as e: logger.error(f"Error running plugin '{name}': {e}", exc_info=True)
        else:
            logger.error(f"Plugin '{name}' not found.")

class SimplePackedRect(NamedTuple):
    frame: SpriteFrame
    image: Image.Image
    x: int
    y: int

class SpriteSheetManager:
    def __init__(self, project: Project):
        self.project = project
        self.clear()

    def clear(self):
        self._frames_to_pack: List[Tuple[SpriteFrame, Image.Image]] = []
        self._packed_rects: List[SimplePackedRect] = []
        self._atlas_size = (0, 0)

    def populate_from_project(self):
        self.clear()
        settings = self.project.settings
        for sprite in self.project.sprites.values():
            for frame in sprite.frames.values():
                image = frame.image
                if settings.apply_bleed:
                    image = ImageProcessor.apply_bleed(ImageProcessor.add_padding(image, settings.sprite_padding), settings.sprite_padding)
                elif settings.sprite_padding > 0:
                    image = ImageProcessor.add_padding(image, settings.sprite_padding)
                self._frames_to_pack.append((frame, image))

    def pack(self):
        if not self._frames_to_pack:
            self.populate_from_project()

        max_width = self.project.settings.sprite_sheet_size[0]
        self._packed_rects = []
        x, y, row_height = 0, 0, 0
        total_w, total_h = 0, 0

        for frame, image in self._frames_to_pack:
            if x + image.width > max_width:
                y += row_height
                x = 0
                row_height = 0

            self._packed_rects.append(SimplePackedRect(frame, image, x, y))

            x += image.width
            row_height = max(row_height, image.height)
            total_w = max(total_w, x)
            total_h = y + row_height

        self._atlas_size = (total_w, total_h)
        logger.info(f"Packing complete. Atlas size: {self._atlas_size}")

    def generate_atlases(self) -> List[Image.Image]:
        if not self._packed_rects:
            self.pack()

        if self._atlas_size == (0, 0):
            return []

        atlas = Image.new('RGBA', self._atlas_size, (0,0,0,0))
        for rect in self._packed_rects:
            atlas.paste(rect.image, (rect.x, rect.y))

        return [atlas]

class ExportManager:
    def export_image(self, image: Image.Image, path: str, format: ExportFormat, **kwargs):
        try:
            if format == ExportFormat.PNG: image.save(path, 'PNG', optimize=True)
            elif format == ExportFormat.GIF: image.save(path, 'GIF', save_all=True, **kwargs)
            elif format == ExportFormat.WEBP: image.save(path, 'WEBP', quality=90)
            elif format == ExportFormat.AVIF: image.save(path, 'AVIF', quality=90)
            else: logger.error(f"Unsupported image export format: {format.name}")
        except Exception as e: logger.error(f"Failed to export image to {path}: {e}")
    def export_sprite_sheet(self, sheet_manager: SpriteSheetManager, base_path: str, atlas_format: ExportFormat, meta_format: ExportFormat):
        atlases = sheet_manager.generate_atlases()
        atlas_filenames = [f"{base_path}_atlas_{i}{atlas_format.value[1]}" for i in range(len(atlases))]
        for i, atlas in enumerate(atlases): self.export_image(atlas, atlas_filenames[i], atlas_format)

class Command(ABC):
    @abstractmethod
    def execute(self): pass
    @abstractmethod
    def undo(self): pass
class UndoManager:
    def __init__(self): self.undo_stack: List[Command] = []; self.redo_stack: List[Command] = []
    def execute(self, command: Command):
        command.execute(); self.undo_stack.append(command); self.redo_stack.clear()
    def undo(self):
        if not self.undo_stack: return
        command = self.undo_stack.pop(); command.undo(); self.redo_stack.append(command)
    def redo(self):
        if not self.redo_stack: return
        command = self.redo_stack.pop(); command.execute(); self.undo_stack.append(command)

class PropertyChangeCommand(Command):
    def __init__(self, target: Any, property_name: str, new_value: Any):
        self.target = target; self.property_name = property_name; self.new_value = new_value
        self.old_value = getattr(target, property_name)
    def execute(self): setattr(self.target, self.property_name, self.new_value)
    def undo(self): setattr(self.target, self.property_name, self.old_value)
class AddSpriteCommand(Command):
    def __init__(self, project: Project, sprite: Sprite):
        self.project = project; self.sprite = sprite
    def execute(self): self.project.add_sprite(self.sprite)
    def undo(self): self.project.sprites.pop(self.sprite.name)
class RemoveSpriteCommand(Command):
    def __init__(self, project: Project, sprite_name: str):
        self.project = project; self.sprite_name = sprite_name; self.sprite = project.sprites[sprite_name]
    def execute(self): self.project.sprites.pop(self.sprite_name)
    def undo(self): self.project.add_sprite(self.sprite)

class CoreEngine:
    def __init__(self):
        self.undo_manager = UndoManager()
        self.project = Project(settings=ProjectSettings.load())
        self.image_processor = ImageProcessor()
        self.plugin_manager = PluginManager(self)
        self.sheet_manager = SpriteSheetManager(self.project)
        self.export_manager = ExportManager()

        self.plugin_manager.discover_plugins(Path('plugins'))

    def start_gui(self, app):
        if not GUI_AVAILABLE: return False
        self.gui = SpriteForgeMainWindow(self); self.gui.show(); return True
    def new_project(self): self.project = Project(settings=ProjectSettings.load()); self.undo_manager = UndoManager()
    def save_project(self, path: str):
        path = Path(path)
        if path.suffix != '.sfp': path = path.with_suffix('.sfp')
        logger.info(f"Saving project to {path}...")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                img_dir = tmp_path / 'images'
                img_dir.mkdir()

                project_data = self._serialize_project(self.project, img_dir)

                with open(tmp_path / 'project.json', 'w', encoding='utf-8') as f:
                    json.dump(project_data, f, indent=2)

                with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for file_path in tmp_path.rglob('*'):
                        zf.write(file_path, file_path.relative_to(tmp_path))
            logger.info(f"Project saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save project: {e}", exc_info=True)

    def _serialize_project(self, project: Project, img_dir: Path) -> Dict:

        def frame_to_dict(frame: SpriteFrame, sprite_name: str) -> Dict:
            frame_dict = frame.__dict__.copy()
            img_path = img_dir / f"{sprite_name}_{frame.name}.png"
            frame.image.save(img_path, 'PNG')
            frame_dict['image'] = str(img_path.relative_to(img_dir.parent))
            return frame_dict

        sprites_data = {
            s_name: {
                "name": sprite.name,
                "metadata": sprite.metadata,
                "frames": {f_name: frame_to_dict(frame, s_name) for f_name, frame in sprite.frames.items()}
            } for s_name, sprite in project.sprites.items()
        }

        animations_data = {
            a_name: {
                "name": anim.name,
                "loop": anim.loop,
                "fps": anim.fps,
                "frames": [f.name for f in anim.frames] # Store frame names as references
            } for a_name, anim in project.animations.items()
        }

        return {
            "settings": project.settings.__dict__,
            "sprites": sprites_data,
            "animations": animations_data,
            "metadata": project.metadata,
            "version": __version__
        }

    def load_project(self, path: str):
        path = Path(path)
        if not path.exists():
            logger.error(f"Project file not found: {path}"); return
        logger.info(f"Loading project from {path}...")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                with zipfile.ZipFile(path, 'r') as zf:
                    zf.extractall(tmp_path)

                with open(tmp_path / 'project.json', 'r', encoding='utf-8') as f:
                    project_data = json.load(f)

                new_project = self._deserialize_project(project_data, tmp_path)
                self.project = new_project
                self.sheet_manager.project = new_project # Re-link sheet manager
                self.undo_manager = UndoManager() # Reset undo stack
            logger.info("Project loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load project: {e}", exc_info=True)

    def _deserialize_project(self, data: Dict, project_dir: Path) -> Project:
        settings = ProjectSettings(**data['settings'])
        new_project = Project(settings=settings, metadata=data.get('metadata', {}))

        all_frames: Dict[str, SpriteFrame] = {}

        for s_name, s_data in data.get('sprites', {}).items():
            new_sprite = Sprite(name=s_name, metadata=s_data.get('metadata', {}))
            for f_name, f_data in s_data.get('frames', {}).items():
                f_data['image'] = Image.open(project_dir / f_data['image'])
                new_frame = SpriteFrame(**f_data)
                new_sprite.add_frame(new_frame)
                all_frames[f_name] = new_frame
            new_project.add_sprite(new_sprite)

        for a_name, a_data in data.get('animations', {}).items():
            anim_frames = [all_frames[f_name] for f_name in a_data['frames'] if f_name in all_frames]
            new_anim = SpriteAnimation(name=a_name, frames=anim_frames, loop=a_data.get('loop', True), fps=a_data.get('fps', 10.0))
            new_project.add_animation(new_anim)

        return new_project


if GUI_AVAILABLE:
    class SpriteForgeMainWindow(QMainWindow):
        def __init__(self, core: CoreEngine):
            super().__init__(); self.core = core; self.init_ui()
        def init_ui(self): self.setWindowTitle(APP_NAME); self.setMinimumSize(1200, 800); self.create_menus(); self.create_docks()
        def create_menus(self):
            menubar = self.menuBar(); tools_menu = menubar.addMenu("&Tools")
            sheet_action = QAction('&Sprite Sheet Manager...', self, triggered=self.open_sheet_manager); tools_menu.addAction(sheet_action)
        def create_docks(self): pass
        def open_sheet_manager(self): SpriteSheetManagerDialog(self, self.core).exec()

    class SpriteSheetManagerDialog(QDialog):
        def __init__(self, parent, core: CoreEngine):
            super().__init__(parent); self.core = core; self.manager = core.sheet_manager; self.init_ui()
        def init_ui(self):
            self.setWindowTitle("Sprite Sheet Manager"); self.resize(800, 600); layout = QVBoxLayout(self)
            self.list = QListWidget(); self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            layout.addWidget(self.list)
            btn_layout = QHBoxLayout()
            repopulate_btn = QPushButton("Repopulate from Project"); repopulate_btn.clicked.connect(self.repopulate)
            btn_layout.addWidget(repopulate_btn)
            pack_btn = QPushButton("Pack"); pack_btn.clicked.connect(self.pack); btn_layout.addWidget(pack_btn)
            layout.addLayout(btn_layout)
            self.repopulate() # Initial population
        def repopulate(self):
            # A dummy sprite for testing purposes until project loading is implemented
            if not self.core.project.sprites:
                try:
                    dummy_img = Image.new('RGBA', (64, 64), (255, 0, 0, 255))
                    dummy_frame = SpriteFrame(name="dummy_frame", image=dummy_img)
                    dummy_sprite = Sprite(name="dummy_sprite")
                    dummy_sprite.add_frame(dummy_frame)
                    self.core.project.add_sprite(dummy_sprite)
                except Exception as e:
                    logger.error(f"Could not create dummy sprite for testing: {e}")

            self.manager.populate_from_project()
            self.list.clear()
            # The new manager stores a list of (frame, image) tuples
            for frame, image in self.manager._frames_to_pack:
                self.list.addItem(f"{frame.name} ({image.width}x{image.height})")
        def pack(self):
            self.manager.pack()
            self.show_atlases()
        def show_atlases(self):
            atlases = self.manager.generate_atlases()
            for i, atlas in enumerate(atlases):
                viewer = QDialog(self); viewer.setWindowTitle(f"Atlas {i+1}"); layout = QVBoxLayout(viewer)
                scene = QGraphicsScene(); pixmap = QPixmap.fromImage(QImage(atlas.tobytes("raw", "RGBA"), atlas.width, atlas.height, QImage.Format.Format_RGBA8888))
                scene.addPixmap(pixmap); view = QGraphicsView(scene); layout.addWidget(view); viewer.exec()

def main():
    parser = argparse.ArgumentParser(description=APP_DESCRIPTION)
    parser.add_argument('--headless', action='store_true', help='Run in headless mode without GUI.')
    parser.add_argument('--input', type=str, help='Path to a project file (.sfp) or a directory of images.')
    parser.add_argument('--output', type=str, help='Path for the output file(s).')
    parser.add_argument('--pack', action='store_true', help='Pack a sprite sheet.')
    parser.add_argument('--run-plugin', type=str, help='Name of the plugin to run.')
    args = parser.parse_args()

    core = CoreEngine()

    if args.headless:
        logger.info("Running in headless mode.")
        if not args.input:
            logger.error("--input is required for headless mode.")
            return 1

        input_path = Path(args.input)
        if input_path.is_dir():
            core.new_project()
            new_sprite = Sprite(name=input_path.name)
            for img_file in input_path.glob('*.*'):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                    try:
                        frame = SpriteFrame(name=img_file.stem, image=Image.open(img_file))
                        new_sprite.add_frame(frame)
                    except Exception as e:
                        logger.warning(f"Could not load image {img_file}: {e}")
            core.project.add_sprite(new_sprite)
            logger.info(f"Loaded {len(new_sprite.frames)} frames from directory {args.input}")
        elif input_path.is_file() and input_path.suffix == '.sfp':
            core.load_project(str(input_path))
        else:
            logger.error(f"Invalid input path: {args.input}")
            return 1

        if args.run_plugin:
            core.plugin_manager.run_plugin(args.run_plugin)

        if args.pack:
            if not args.output:
                logger.error("--output is required when packing.")
                return 1
            core.sheet_manager.pack()
            atlases = core.sheet_manager.generate_atlases()
            base_path = Path(args.output).with_suffix('')
            logger.info(f"Generated {len(atlases)} atlas(es). Saving...")
            core.export_manager.export_sprite_sheet(core.sheet_manager, str(base_path), ExportFormat.PNG, ExportFormat.SPRITE_SHEET_JSON)

        logger.info("Headless execution finished.")

    else:
        if not GUI_AVAILABLE:
            logger.error("GUI not available, and --headless not specified. Exiting.")
            return 1
        app = QApplication(sys.argv)
        core.start_gui(app)
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
