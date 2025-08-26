from __future__ import annotations

# Standard lib
import sys, os, json, time, argparse, logging, subprocess, zipfile, shutil, traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, NamedTuple
from abc import ABC, abstractmethod
from datetime import datetime
import colorsys

# Third-party
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw, ImageFont

# ------------------ App constants (make sure these exist only once) ------------------
__version__ = "1.0.0"
APP_NAME = "Sprite Forge Pro 2025"
ORG_NAME = "SpriteForge"
APP_KEY = "SpriteForgePro2025"

# ------------------ Ensure core dependencies (optional safety) ------------------
def ensure_dependencies():
    required = {"pillow": "PIL", "numpy": "numpy", "PyQt6": "PyQt6"}
    missing = []
    for pkg, mod in required.items():
        try:
            __import__(mod if mod != "PIL" else "PIL")
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing required packages: {', '.join(missing)}")
        for pkg in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Comment out if you prefer manual installs:
# ensure_dependencies()

# ------------------ PyQt6 import guard (defines GUI_AVAILABLE) ------------------
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox, QSpinBox, QSlider,
        QTextEdit, QProgressBar, QFileDialog, QMessageBox, QTabWidget, QGroupBox,
        QScrollArea, QSplitter, QStatusBar, QMenuBar, QMenu, QToolBar, QFrame,
        QSpacerItem, QSizePolicy, QListWidget, QListWidgetItem, QTreeWidget,
        QTreeWidgetItem, QDialog, QDialogButtonBox, QFormLayout, QPlainTextEdit
    )
    from PyQt6.QtCore import (
        Qt, QThread, pyqtSignal, QTimer, QSettings, QSize, QPoint, QRect,
        QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
    )
    from PyQt6.QtGui import (
        QPixmap, QPainter, QColor, QFont, QAction, QIcon, QPalette, QTransform,
        QPen, QBrush, QKeySequence, QShortcut, QImage, QFontMetrics, QPainterPath
    )
    GUI_AVAILABLE = True
except Exception as e:
    print(f"PyQt6 not available: {e}")
    GUI_AVAILABLE = False


import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw, ImageFont
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, NamedTuple
from pathlib import Path
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox, QSpinBox, QScrollArea, QGroupBox, QSplitter, QStatusBar, QProgressBar, QFileDialog, QMessageBox, QTabWidget, QPlainTextEdit, QDialog, QDialogButtonBox, QFormLayout
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QPixmap, QImage, QKeySequence
import importlib.util
import colorsys
import traceback
import logging
import shutil
import subprocess
import zipfile
import time
import sys
import os
import json
import argparse

# ------------------ Version and constants ------------------
__version__ = "1.0.0"
APP_NAME = "Sprite Forge Pro 2025"
ORG_NAME = "SpriteForge"
APP_KEY = "SpriteForgePro2025"

# ------------------ GUI availability check ------------------
try:
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
        QPushButton, QLineEdit, QComboBox, QCheckBox, QSpinBox, QScrollArea,
        QGroupBox, QSplitter, QStatusBar, QProgressBar, QFileDialog,
        QMessageBox, QTabWidget, QPlainTextEdit, QDialog, QDialogButtonBox,
        QFormLayout
    )
    from PyQt6.QtCore import Qt, QSettings
    from PyQt6.QtGui import QPixmap, QImage, QKeySequence
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False



def init_ui(self):
        """Initialize the comprehensive user interface."""
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.setMinimumSize(1400, 900)

        # Apply dark theme styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-size: 11px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
                background-color: #353535;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #666;
                padding: 6px 12px;
                border-radius: 4px;
                min-width: 70px;
                font-weight: normal;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:pressed { background-color: #606060; }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
                border-color: #444;
            }
            QLineEdit, QComboBox {
                background-color: #404040;
                border: 1px solid #666;
                padding: 4px;
                border-radius: 3px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #4a90e2;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #353535;
            }
            QTabBar::tab {
                background-color: #404040;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4a90e2;
            }
            QTabBar::tab:hover {
                background-color: #505050;
            }
            QTextEdit, QPlainTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #555;
                color: #ffffff;
            }
            QScrollBar:vertical {
                background-color: #2a2a2a;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
        """)

        # Create menu bar
        self.create_menus()

        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Canvas and tabs
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([450, 950])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Status bar with progress
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.status_bar.showMessage("Ready - Load an image to begin")


def create_menus(self):
        """Create comprehensive application menus."""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #353535;
                color: #ffffff;
                border-bottom: 1px solid #555;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 6px 12px;
            }
            QMenuBar::item:selected {
                background-color: #4a90e2;
            }
            QMenu {
                background-color: #353535;
                color: #ffffff;
                border: 1px solid #555;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #4a90e2;
            }
            QMenu::separator {
                height: 1px;
                background-color: #555;
                margin: 2px;
            }
        """)

        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction(
    "Open Image",
    self.open_image,
     QKeySequence("Ctrl+O"))
        file_menu.addAction(
    "Save Image",
    self.save_image,
     QKeySequence("Ctrl+S"))
        file_menu.addSeparator()

        # Recent files submenu
        self.recent_menu = file_menu.addMenu("Recent Files")
        self.update_recent_menu()

        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, QKeySequence("Ctrl+Q"))

        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("Reload Plugins", self.reload_plugins)
        tools_menu.addAction("Plugin Manager", self.show_plugin_manager)
        tools_menu.addSeparator()
        tools_menu.addAction("External Tools", self.configure_external_tools)
        tools_menu.addSeparator()
        tools_menu.addAction("Test in GZDoom", self.test_in_gzdoom)
        tools_menu.addAction("Test in Zandronum", self.test_in_zandronum)

        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Zoom In", self.zoom_in, QKeySequence("Ctrl++"))
        view_menu.addAction("Zoom Out", self.zoom_out, QKeySequence("Ctrl+-"))
        view_menu.addAction(
    "Reset View",
    self.reset_zoom,
     QKeySequence("Ctrl+0"))
        view_menu.addSeparator()
        view_menu.addAction("Toggle Grid", self.toggle_grid, QKeySequence("G"))
        view_menu.addAction("Toggle Onion Skin", self.toggle_onion_skin)

        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("Help", self.show_help, QKeySequence("F1"))
        help_menu.addAction("About", self.show_about)


def create_left_panel(self) -> QWidget:
        """Create comprehensive left control panel."""
        panel = QWidget()
        panel.setMinimumWidth(420)
        panel.setMaximumWidth(500)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        # File controls
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)

        file_buttons = QHBoxLayout()
        open_btn = QPushButton("Open Image")
        open_btn.clicked.connect(self.open_image)
        file_buttons.addWidget(open_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_image)
        file_buttons.addWidget(save_btn)

        file_layout.addLayout(file_buttons)

        self.file_label = QLabel("No image loaded")
        self.file_label.setStyleSheet(
            "color: #aaa; font-style: italic; padding: 4px;")
        file_layout.addWidget(self.file_label)

        layout.addWidget(file_group)

        # Sprite settings
        sprite_group = QGroupBox("Sprite Settings")
        sprite_layout = QGridLayout(sprite_group)
        sprite_layout.setSpacing(6)

        sprite_layout.addWidget(QLabel("Name:"), 0, 0)
        self.name_edit = QLineEdit("SPRT")
        self.name_edit.setMaxLength(4)
        self.name_edit.textChanged.connect(self.validate_sprite_name)
        sprite_layout.addWidget(self.name_edit, 0, 1)

        sprite_layout.addWidget(QLabel("Type:"), 1, 0)
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "Character", "Weapon", "Item", "Wall", "Flat",
            "Decoration", "Projectile", "Effect"
        ])
        sprite_layout.addWidget(self.type_combo, 1, 1)

        sprite_layout.addWidget(QLabel("Size:"), 2, 0)
        self.size_combo = QComboBox()
        self.size_combo.addItems(["Auto", "32", "64", "128", "256", "512"])
        sprite_layout.addWidget(self.size_combo, 2, 1)

        sprite_layout.addWidget(QLabel("X Offset:"), 3, 0)
        self.offset_x_spin = QSpinBox()
        self.offset_x_spin.setRange(-999, 999)
        self.offset_x_spin.setValue(0)
        sprite_layout.addWidget(self.offset_x_spin, 3, 1)

        sprite_layout.addWidget(QLabel("Y Offset:"), 4, 0)
        self.offset_y_spin = QSpinBox()
        self.offset_y_spin.setRange(-999, 999)
        self.offset_y_spin.setValue(0)
        sprite_layout.addWidget(self.offset_y_spin, 4, 1)

        layout.addWidget(sprite_group)

        # Processing options
        process_group = QGroupBox("Processing Options")
        process_layout = QVBoxLayout(process_group)

        self.pixelate_check = QCheckBox("Pixelate Image")
        self.pixelate_check.setChecked(True)
        process_layout.addWidget(self.pixelate_check)

        self.palette_check = QCheckBox("Apply Doom Palette")
        self.palette_check.setChecked(True)
        process_layout.addWidget(self.palette_check)

        self.transparency_check = QCheckBox("Preserve Transparency")
        self.transparency_check.setChecked(True)
        process_layout.addWidget(self.transparency_check)

        self.sharpen_check = QCheckBox("Sharpen Edges")
        self.sharpen_check.setChecked(False)
        process_layout.addWidget(self.sharpen_check)

        self.denoise_check = QCheckBox("Reduce Noise")
        self.denoise_check.setChecked(False)
        process_layout.addWidget(self.denoise_check)

        # Preview button
        preview_btn = QPushButton("Preview Changes")
        preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover { background-color: #5ba0f2; }
        """)
        preview_btn.clicked.connect(self.preview_changes)
        process_layout.addWidget(preview_btn)

        layout.addWidget(process_group)

        # Plugins section
        plugin_group = QGroupBox("Plugins")
        plugin_layout = QVBoxLayout(plugin_group)

        plugin_toolbar = QHBoxLayout()
        reload_plugins_btn = QPushButton("Reload")
        reload_plugins_btn.clicked.connect(self.reload_plugins)
        plugin_toolbar.addWidget(reload_plugins_btn)
        plugin_toolbar.addStretch()
        plugin_layout.addLayout(plugin_toolbar)

        self.plugin_scroll = QScrollArea()
        self.plugin_scroll.setWidgetResizable(True)
        self.plugin_scroll.setMaximumHeight(250)
        self.plugin_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #555;
                background-color: #2a2a2a;
            }
        """)

        self.plugin_widget = QWidget()
        self.plugin_layout = QVBoxLayout(self.plugin_widget)
        self.plugin_layout.setSpacing(4)
        self.plugin_scroll.setWidget(self.plugin_widget)

        plugin_layout.addWidget(self.plugin_scroll)
        self.load_plugin_widgets()

        layout.addWidget(plugin_group)

        # Export section
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        export_buttons = QHBoxLayout()
        pk3_btn = QPushButton("Export PK3")
        pk3_btn.clicked.connect(self.export_pk3)
        export_buttons.addWidget(pk3_btn)

        wad_btn = QPushButton("WAD Layout")
        wad_btn.clicked.connect(self.export_wad)
        export_buttons.addWidget(wad_btn)
        export_layout.addLayout(export_buttons)

        sheet_btn = QPushButton("Build Sprite Sheet")
        sheet_btn.clicked.connect(self.build_sprite_sheet)
        export_layout.addWidget(sheet_btn)

        layout.addWidget(export_group)

        # External tools section
        external_group = QGroupBox("External Tools")
        external_layout = QVBoxLayout(external_group)

        tool_buttons = QHBoxLayout()
        gzdoom_btn = QPushButton("Test GZDoom")
        gzdoom_btn.clicked.connect(self.test_in_gzdoom)
        tool_buttons.addWidget(gzdoom_btn)

        zandronum_btn = QPushButton("Test Zandronum")
        zandronum_btn.clicked.connect(self.test_in_zandronum)
        tool_buttons.addWidget(zandronum_btn)
        external_layout.addLayout(tool_buttons)

        udb_btn = QPushButton("Launch UDB")
        udb_btn.clicked.connect(self.launch_udb)
        external_layout.addWidget(udb_btn)

        layout.addWidget(external_group)

        # Stretch to push everything to top
        layout.addStretch()

        return panel


def create_right_panel(self) -> QWidget:
        """Create right panel with comprehensive tabs."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget for different views
        self.tab_widget = QTabWidget()

        # Canvas tab
        canvas_tab = self.create_canvas_tab()
        self.tab_widget.addTab(canvas_tab, "Canvas")

        # Sheet tab
        sheet_tab = self.create_sheet_tab()
        self.tab_widget.addTab(sheet_tab, "Sheet")

        # Preview tab
        preview_tab = self.create_preview_tab()
        self.tab_widget.addTab(preview_tab, "Preview")

        # Archive tab
        archive_tab = self.create_archive_tab()
        self.tab_widget.addTab(archive_tab, "Archive")

        # Log tab
        log_tab = self.create_log_tab()
        self.tab_widget.addTab(log_tab, "Log")

        layout.addWidget(self.tab_widget)
        return panel


def create_canvas_tab(self) -> QWidget:
        """Create main canvas tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        # Canvas toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(self.zoom_in)
        toolbar.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(self.zoom_out)
        toolbar.addWidget(zoom_out_btn)

        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_zoom)
        toolbar.addWidget(reset_btn)

        toolbar.addWidget(QFrame())  # Separator

        grid_btn = QPushButton("Toggle Grid")
        grid_btn.clicked.connect(self.toggle_grid)
        toolbar.addWidget(grid_btn)

        onion_btn = QPushButton("Onion Skin")
        onion_btn.clicked.connect(self.toggle_onion_skin)
        toolbar.addWidget(onion_btn)

        toolbar.addStretch()

        # Image info
        self.image_info_label = QLabel("No image")
        self.image_info_label.setStyleSheet("color: #aaa; padding: 4px;")
        toolbar.addWidget(self.image_info_label)

        layout.addLayout(toolbar)

        # Main canvas
        self.canvas = ImageCanvas()
        self.canvas.imageClicked.connect(self.on_canvas_clicked)
        self.canvas.imageChanged.connect(self.update_image_info)
        layout.addWidget(self.canvas)

        # Enable drag and drop
        self.canvas.setAcceptDrops(True)
        self.canvas.dragEnterEvent = self.drag_enter_event
        self.canvas.dropEvent = self.drop_event

        return tab


def create_sheet_tab(self) -> QWidget:
        """Create sprite sheet management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Sheet controls
        controls_group = QGroupBox("Sheet Controls")
        controls_layout = QGridLayout(controls_group)

        controls_layout.addWidget(QLabel("Grid Width:"), 0, 0)
        self.grid_width_spin = QSpinBox()
        self.grid_width_spin.setRange(8, 512)
        self.grid_width_spin.setValue(64)
        controls_layout.addWidget(self.grid_width_spin, 0, 1)

        controls_layout.addWidget(QLabel("Grid Height:"), 0, 2)
        self.grid_height_spin = QSpinBox()
        self.grid_height_spin.setRange(8, 512)
        self.grid_height_spin.setValue(64)
        controls_layout.addWidget(self.grid_height_spin, 0, 3)

        auto_slice_btn = QPushButton("Auto Slice")
        auto_slice_btn.clicked.connect(self.auto_slice_sheet)
        controls_layout.addWidget(auto_slice_btn, 1, 0, 1, 2)

        manual_slice_btn = QPushButton("Manual Slice")
        manual_slice_btn.clicked.connect(self.manual_slice_sheet)
        controls_layout.addWidget(manual_slice_btn, 1, 2, 1, 2)

        layout.addWidget(controls_group)

        # Sheet preview (placeholder)
        sheet_preview = QLabel(
            "Load sprite sheet in Canvas tab to begin slicing")
        sheet_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sheet_preview.setStyleSheet("""
            QLabel {
                border: 2px dashed #555;
                color: #aaa;
                padding: 20px;
                background-color: #2a2a2a;
            }
        """)
        layout.addWidget(sheet_preview)

        return tab


def create_preview_tab(self) -> QWidget:
        """Create animation preview tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Animation controls
        controls_group = QGroupBox("Animation Controls")
        controls_layout = QHBoxLayout(controls_group)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_animation)
        controls_layout.addWidget(self.play_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_animation)
        controls_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_animation)
        controls_layout.addWidget(self.stop_btn)

        controls_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.valueChanged.connect(self.set_animation_fps)
        controls_layout.addWidget(self.fps_spin)

        controls_layout.addStretch()
        layout.addWidget(controls_group)

        # 8-angle rotation preview
        rotations_group = QGroupBox("8-Angle Rotation Preview")
        rotations_layout = QGridLayout(rotations_group)

        self.rotation_previews = []
        for i in range(8):
            preview = QLabel(f"Angle {i + 1}")
            preview.setMinimumSize(80, 80)
            preview.setStyleSheet("""
                QLabel {
                    border: 1px solid #555;
                    background-color: #2a2a2a;
                    color: #aaa;
                }
            """)
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.rotation_previews.append(preview)
            rotations_layout.addWidget(preview, i // 4, i % 4)

        layout.addWidget(rotations_group)

        # Main animation preview
        self.animation_preview = AnimationPreview()
        layout.addWidget(self.animation_preview)

        return tab


def create_archive_tab(self) -> QWidget:
        """Create archive explorer tab."""
        self.archive_explorer = ArchiveExplorer()
        return self.archive_explorer


def create_log_tab(self) -> QWidget:
        """Create log display tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Log controls
        controls = QHBoxLayout()
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.clear_log)
        controls.addWidget(clear_btn)

        save_log_btn = QPushButton("Save Log")
        save_log_btn.clicked.connect(self.save_log)
        controls.addWidget(save_log_btn)

        controls.addStretch()
        layout.addLayout(controls)

        # Log display
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QPlainTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #555;
            }
        """)
        layout.addWidget(self.log_text)

        return tab

    # Validation and utility methods


def validate_sprite_name(self):
        """Validate and format sprite name."""
        current = self.name_edit.text()
        validated = validate_sprite_name(current)
        if current != validated:
            self.name_edit.setText(validated)
        self.update_image_info()


def update_image_info(self):
        """Update image information display."""
        if self.current_image:
            name = validate_sprite_name(self.name_edit.text())
            w, h = self.current_image.size
            info = f"{name} | {w}×{h} pixels"
            if self.processed_image:
                pw, ph = self.processed_image.size
                info += f" → {pw}×{ph}"
            self.image_info_label.setText(info)
        else:
            self.image_info_label.setText("No image")


def log_message(self, message: str, level: str = "INFO"):
        """Add timestamped message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}"

        self.log_text.appendPlainText(formatted_message)

        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

        # Update status bar temporarily
        self.status_bar.showMessage(message, 3000)

        # Log to file as well
        logger.info(message)


def update_recent_menu(self):
        """Update recent files menu."""
        self.recent_menu.clear()
        for file_path in self.recent_files[:10]:  # Show last 10
            if os.path.exists(file_path):
                action = self.recent_menu.addAction(
                    os.path.basename(file_path),
                    lambda fp=file_path: self.load_image_file(fp)
                )


def add_recent_file(self, file_path: str):
        """Add file to recent files list."""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        self.recent_files = self.recent_files[:10]  # Keep only 10 recent
        self.update_recent_menu()

    # File operations


def open_image(self):
        """Open image file dialog."""
        file_path, _ = QFileDialog.getOpenImageFileName(
            self, "Open Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tga);;All Files (*)"
        )

        if file_path:
            self.load_image_file(file_path)


def load_image_file(self, file_path: str):
        """Load image from file path."""
        try:
            self.current_image = Image.open(file_path).convert('RGBA')
            self.original_image = self.current_image.copy()
            self.processed_image = None
            self.canvas.set_image(self.current_image)
            self.file_label.setText(f"Loaded: {os.path.basename(file_path)}")
            self.add_recent_file(file_path)
            self.update_image_info()
            self.log_message(f"Loaded image: {os.path.basename(file_path)}")

            # Update rotation previews
            self.update_rotation_previews()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")
            self.log_message(f"Error loading {file_path}: {e}", "ERROR")


def save_image(self):
        """Save current processed or original image."""
        image_to_save = self.processed_image or self.current_image
        if not image_to_save:
            QMessageBox.warning(self, "Warning", "No image to save")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )

        if file_path:
            try:
                # Convert RGBA to RGB for JPEG
                if file_path.lower().endswith(('.jpg', '.jpeg')):
                    if image_to_save.mode == 'RGBA':
                        # Create white background for JPEG
                        rgb_image = Image.new(
    'RGB', image_to_save.size, (255, 255, 255))
                        rgb_image.paste(
    image_to_save, mask=image_to_save.split()[3])
                        rgb_image.save(file_path, quality=95)
                    else:
                        image_to_save.save(file_path, quality=95)
                else:
                    image_to_save.save(file_path)

                self.log_message(f"Saved image: {os.path.basename(file_path)}")

            except Exception as e:
                QMessageBox.critical(
    self, "Error", f"Failed to save image: {e}")
                self.log_message(f"Error saving {file_path}: {e}", "ERROR")

    # Drag and drop support


def drag_enter_event(self, event):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()


def drop_event(self, event):
        """Handle drop event."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(
    ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tga')):
                self.load_image_file(file_path)
                event.accept()
            else:
                event.ignore()

    # Image processing


def preview_changes(self):
        """Preview processing changes."""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate

            processed = self.apply_basic_processing(self.current_image)
            self.processed_image = processed
            self.canvas.set_image(processed)
            self.update_image_info()
            self.log_message("Preview generated successfully")

            # Update animation frames if applicable
            if hasattr(self, 'animation_frames') and self.animation_frames:
                self.animation_preview.set_animation(
    self.animation_frames, self.fps_spin.value())

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {e}")
            self.log_message(f"Processing error: {e}", "ERROR")
        finally:
            self.progress_bar.setVisible(False)


def apply_basic_processing(self, image: Image.Image) -> Image.Image:
        """Apply basic sprite processing options."""
        img = image.copy()

        # Resize if specified
        size_text = self.size_combo.currentText()
        if size_text != "Auto":
            new_size = int(size_text)
            # Maintain aspect ratio
            w, h = img.size
            if w > h:
                new_h = int((h * new_size) / w)
                new_w = new_size
            else:
                new_w = int((w * new_size) / h)
                new_h = new_size
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Apply effects in order
        if self.denoise_check.isChecked():
            img = img.filter(ImageFilter.MedianFilter(size=3))

        if self.pixelate_check.isChecked():
            img = pixelate_image(img)

        if self.sharpen_check.isChecked():
            img = img.filter(ImageFilter.SHARPEN)

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
        self.processing_thread = ProcessingThread(
    base_image, plugin.process_image, **parameters)
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
        self.update_image_info()
        self.log_message("Plugin processing completed")


def on_plugin_error(self, error_message: str):
        """Handle plugin processing error."""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(
    self,
    "Plugin Error",
     f"Plugin processing failed:\n{error_message}")
        self.log_message(f"Plugin error: {error_message}", "ERROR")

    # Plugin management


def load_plugin_widgets(self):
        """Load plugin widgets into scroll area."""
        # Clear existing widgets
        for i in reversed(range(self.plugin_layout.count())):
            child = self.plugin_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

        # Add plugin widgets
        if not self.plugin_manager.plugins:
            no_plugins = QLabel("No plugins loaded")
            no_plugins.setStyleSheet(
                "color: #666; font-style: italic; padding: 10px;")
            no_plugins.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.plugin_layout.addWidget(no_plugins)
        else:
            for plugin in self.plugin_manager.plugins.values():
                plugin_widget = PluginWidget(plugin)
                plugin_widget.applyRequested.connect(self.apply_plugin)
                self.plugin_layout.addWidget(plugin_widget)

        self.plugin_layout.addStretch()


def reload_plugins(self):
        """Reload all plugins."""
        try:
            self.plugin_manager.load_plugins()
            self.load_plugin_widgets()
            count = len(self.plugin_manager.plugins)
            self.log_message(f"Reloaded {count} plugins")

            if count == 0:
                QMessageBox.information(
                    self, "Plugins",
                    "No plugins found. Check the plugins/ directory."
                )

        except Exception as e:
            QMessageBox.critical(
    self, "Error", f"Failed to reload plugins: {e}")
            self.log_message(f"Plugin reload error: {e}", "ERROR")


def show_plugin_manager(self):
        """Show plugin manager dialog."""
        plugins = self.plugin_manager.list_plugins()

        if not plugins:
            QMessageBox.information(
                self, "Plugin Manager",
                "No plugins loaded.\n\nCreate JSON plugins in the plugins/ directory."
            )
            return

        info_text = f"Loaded Plugins ({len(plugins)}):\n\n"

        for plugin_info in plugins:
            info_text += f"• {plugin_info.name} v{plugin_info.version}\n"
            info_text += f"  {plugin_info.description}\n"
            info_text += f"  Author: {plugin_info.author}\n"
            info_text += f"  Category: {plugin_info.category}\n"
            info_text += f"  File: {
    os.path.basename(
        plugin_info.file_path)}\n\n"

        dialog = QMessageBox(self)
        dialog.setWindowTitle("Plugin Manager")
        dialog.setText(info_text)
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.exec()

    # Canvas controls


def zoom_in(self):
        """Zoom in on canvas."""
        self.canvas.zoom_in()


def zoom_out(self):
        """Zoom out on canvas."""
        self.canvas.zoom_out()


def reset_zoom(self):
        """Reset canvas zoom and pan."""
        self.canvas.reset_view()


def toggle_grid(self):
        """Toggle pixel grid on canvas."""
        self.canvas.toggle_grid()
        self.log_message(
    f"Pixel grid {
        'enabled' if self.canvas.show_grid else 'disabled'}")


def toggle_onion_skin(self):
        """Toggle onion skin overlay."""
        self.canvas.toggle_onion_skin()
        self.log_message(
    f"Onion skin {
        'enabled' if self.canvas.show_onion_skin else 'disabled'}")


def on_canvas_clicked(self, x: int, y: int):
        """Handle canvas pixel click events."""
        if self.current_image:
            try:
                # Get pixel color
                pixel = self.current_image.getpixel((x, y))
                if len(pixel) == 4:  # RGBA
                    color_info = f"RGBA({
    pixel[0]}, {
        pixel[1]}, {
            pixel[2]}, {
                pixel[3]})"
                else:  # RGB
                    color_info = f"RGB({pixel[0]}, {pixel[1]}, {pixel[2]})"

                message = f"Pixel ({x}, {y}): {color_info}"
                self.log_message(message)
                self.status_bar.showMessage(message, 5000)

            except Exception as e:
                self.log_message(f"Error reading pixel: {e}", "ERROR")

    # Sprite sheet operations


def auto_slice_sheet(self):
        """Automatically slice loaded image as sprite sheet."""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "Load an image first")
            return

        try:
            self.sheet_manager.load_sheet(self.current_image)
            grid_w = self.grid_width_spin.value()
            grid_h = self.grid_height_spin.value()

            frames = self.sheet_manager.auto_slice(grid_w, grid_h)
            if frames:
                self.animation_frames = frames
                self.animation_preview.set_animation(
                    frames, self.fps_spin.value())
                self.log_message(
    f"Sliced sheet into {
        len(frames)} frames ({grid_w}x{grid_h})")
            else:
                self.log_message("No frames found in sprite sheet", "WARNING")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Auto slice failed: {e}")
            self.log_message(f"Auto slice error: {e}", "ERROR")


def manual_slice_sheet(self):
        """Manual slice using selected regions."""
        QMessageBox.information(
    self, "TODO", "Manual slicing not yet implemented.\nUse auto slice for now.")


def build_sprite_sheet(self):
        """Build sprite sheet from current frames."""
        if not self.animation_frames:
            QMessageBox.warning(
    self, "Warning", "No frames to build sheet from")
            return

        try:
            cols = 8  # Default column count
            sheet = self.sheet_manager.assemble_sheet(
                self.animation_frames, cols)

            # Save sheet
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Sprite Sheet", "",
                "PNG Files (*.png);;All Files (*)"
            )

            if file_path:
                sheet.save(file_path)
                self.log_message(
    f"Sprite sheet saved: {
        os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(
    self, "Error", f"Failed to build sprite sheet: {e}")
            self.log_message(f"Sheet build error: {e}", "ERROR")

    # Animation preview


def play_animation(self):
        """Start animation playback."""
        if self.animation_frames:
            self.animation_preview.play()
            self.log_message("Animation playback started")
        else:
            QMessageBox.information(self, "Info", "No animation frames loaded")


def pause_animation(self):
        """Pause animation playback."""
        self.animation_preview.pause()
        self.log_message("Animation paused")


def stop_animation(self):
        """Stop animation playback."""
        self.animation_preview.stop()
        self.log_message("Animation stopped")


def set_animation_fps(self, fps: int):
        """Set animation playback FPS."""
        self.animation_preview.set_fps(fps)
        self.log_message(f"Animation FPS set to {fps}")


def update_rotation_previews(self):
        """Update 8-angle rotation previews."""
        if not self.current_image:
            return

        try:
            rotations = create_sprite_rotations(self.current_image, 8)

            for i, (angle, rotation_img) in enumerate(rotations.items()):
                if i < len(self.rotation_previews):
                    # Convert to pixmap and scale
                    img_array = np.array(
    rotation_img.resize(
        (64, 64), Image.Resampling.NEAREST))
                    if len(
    img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
                        h, w, ch = img_array.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(
    img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
                        pixmap = QPixmap.fromImage(qt_image)
                        self.rotation_previews[i].setPixmap(pixmap)
                        self.rotation_previews[i].setText("")

        except Exception as e:
            self.log_message(f"Error updating rotation previews: {e}", "ERROR")

    # Export functions


def export_pk3(self):
        """Export sprite as PK3 package."""
        image_to_export = self.processed_image or self.current_image
        if not image_to_export:
            QMessageBox.warning(self, "Warning", "No image to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export PK3", "",
            "PK3 Files (*.pk3);;All Files (*)"
        )

        if file_path:
            try:
                base_name = validate_sprite_name(self.name_edit.text())
                self.create_pk3_package(file_path, base_name, image_to_export)

                self.log_message(
    f"PK3 exported: {
        os.path.basename(file_path)}")
                QMessageBox.information(
                    self, "Export Complete",
                    f"PK3 package created successfully:\n{file_path}\n\n"
                    f"Load it in GZDoom with:\n"
                    f"gzdoom -file \"{os.path.basename(file_path)}\""
                )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"PK3 export failed: {e}")
                self.log_message(f"PK3 export error: {e}", "ERROR")


def create_pk3_package(
    self,
    file_path: str,
    base_name: str,
     image: Image.Image):
        """Create PK3 package with proper structure."""
        import io

        # Generate rotations
        rotations = create_sprite_rotations(image, 8)

        with zipfile.ZipFile(file_path, 'w', compression=zipfile.ZIP_DEFLATED) as pk3:
            # Add sprite PNGs
            for i, (angle, rot_img) in enumerate(rotations.items(), 1):
                sprite_name = f"{base_name}A{angle}"

                # Save image to memory buffer
                img_buffer = io.BytesIO()
                rot_img.save(img_buffer, format='PNG')
                pk3.writestr(
    f"sprites/{sprite_name}.png",
     img_buffer.getvalue())

            # Create TEXTURES lump
            textures_content = self.generate_textures_lump(
                base_name, rotations)
            pk3.writestr("TEXTURES", textures_content)

            # Create basic DECORATE if requested
            decorate_content = self.generate_decorate_stub(base_name)
            pk3.writestr("DECORATE", decorate_content)

            # Add metadata
            metadata = {
                "generator": f"{APP_NAME} v{__version__}",
                "sprite_name": base_name,
                "sprite_type": self.type_combo.currentText(),
                "frame_count": len(rotations),
                "created": datetime.now().isoformat()
            }
            pk3.writestr("sprite_info.json", json.dumps(metadata, indent=2))


def generate_textures_lump(
    self, base_name: str, rotations: Dict[str, Image.Image]) -> str:
        """Generate TEXTURES lump content."""
        content = f"// Generated by {APP_NAME} v{__version__}\n"
        content += f"// Sprite: {base_name}\n\n"

        for angle, img in rotations.items():
            sprite_name = f"{base_name}A{angle}"
            width, height = img.size

            # Calculate offsets
            offset_x = self.offset_x_spin.value() or width // 2
            offset_y = self.offset_y_spin.value() or height

            content += f"sprite {sprite_name} {width} {height}\n"
            content += "{\n"
            content += f"    XOffset {offset_x}\n"
            content += f"    YOffset {offset_y}\n"
            content += f"    Patch {sprite_name} 0 0\n"
            content += "}\n\n"

        return content


def generate_decorate_stub(self, base_name: str) -> str:
        """Generate basic DECORATE stub."""
        sprite_type = self.type_combo.currentText().lower()

        if sprite_type == "weapon":
            return f"""// {base_name} Weapon - Generated by {APP_NAME}
actor {base_name}Weapon : Weapon
{{
    Weapon.SelectionOrder 100
    States
    {{
    Ready:
        {base_name} A 1 A_WeaponReady
        Loop
    Deselect:
        {base_name} A 1 A_Lower
        Loop
    Select:
        {base_name} A 1 A_Raise
        Loop
    Fire:
        {base_name} A 4 A_FireBullets(1,1,1,5)
        {base_name} A 8
        Goto Ready
    }}
}}
"""
        elif sprite_type == "projectile":
            return f"""// {base_name} Projectile - Generated by {APP_NAME}
actor {base_name}Ball
{{
    Radius 6
    Height 8
    Speed 20
    Damage 8
    Projectile
    +RANDOMIZE
    SeeSound "weapons/rocklf"
    DeathSound "weapons/rocklx"
    States
    {{
    Spawn:
        {base_name} A 1 Bright
        Loop
    Death:
        {base_name} A 8 Bright
        Stop
    }}
}}
"""
        else:
            return f"""// {base_name} Actor - Generated by {APP_NAME}
actor {base_name}Actor
{{
    Radius 16
    Height 56
    States
    {{
    Spawn:
        {base_name} A -1
        Stop
    }}
}}
"""


def export_wad(self):
        """Export WAD layout directory."""
        image_to_export = self.processed_image or self.current_image
        if not image_to_export:
            QMessageBox.warning(self, "Warning", "No image to export")
            return

        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Export Directory")

        if folder_path:
            try:
                base_name = validate_sprite_name(self.name_edit.text())
                self.create_wad_layout(folder_path, base_name, image_to_export)

                self.log_message(
    f"WAD layout created: {folder_path}/{base_name}_wad")
                QMessageBox.information(
                    self, "Export Complete",
                    f"WAD layout directory created:\n{folder_path}/{base_name}_wad"
                )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"WAD export failed: {e}")
                self.log_message(f"WAD export error: {e}", "ERROR")


def create_wad_layout(
    self,
    folder_path: str,
    base_name: str,
     image: Image.Image):
        """Create WAD layout directory structure."""
        wad_dir = os.path.join(folder_path, f"{base_name}_wad")
        os.makedirs(wad_dir, exist_ok=True)

        # Generate rotations
        rotations = create_sprite_rotations(image, 8)

        # Save individual frames
        frame_info = []
        for angle, rot_img in rotations.items():
            sprite_name = f"{base_name}A{angle}"
            filename = f"{sprite_name}.png"
            filepath = os.path.join(wad_dir, filename)

            rot_img.save(filepath)

            frame_info.append({
                "name": sprite_name,
                "filename": filename,
                "angle": int(angle),
                "width": rot_img.width,
                "height": rot_img.height,
                "offset_x": self.offset_x_spin.value() or rot_img.width // 2,
                "offset_y": self.offset_y_spin.value() or rot_img.height
            })

        # Create info JSON
        info = {
            "base_name": base_name,
            "sprite_type": self.type_combo.currentText(),
            "frames": frame_info,
            "generator": f"{APP_NAME} v{__version__}",
            "created": datetime.now().isoformat()
        }

        info_path = os.path.join(wad_dir, f"{base_name}_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

    # External tool integration


def configure_external_tools(self):
        """Configure external tool paths."""
        dialog = QDialog(self)
        dialog.setWindowTitle("External Tools Configuration")
        dialog.setMinimumSize(500, 300)

        layout = QFormLayout(dialog)

        tool_edits = {}
        for tool_name, current_path in self.external_tools.items():
            edit = QLineEdit(current_path)
            browse_btn = QPushButton("Browse")

            tool_layout = QHBoxLayout()
            tool_layout.addWidget(edit)
            tool_layout.addWidget(browse_btn)

            tool_widget = QWidget()
            tool_widget.setLayout(tool_layout)

            # Connect browse button
            browse_btn.clicked.connect(
                lambda checked, e=edit, t=tool_name: self.browse_external_tool(
                    e, t)
            )

            layout.addRow(f"{tool_name.title()}:", tool_widget)
            tool_edits[tool_name] = edit

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            for tool_name, edit in tool_edits.items():
                self.external_tools[tool_name] = edit.text().strip()
            self.log_message("External tool paths updated")


def browse_external_tool(self, line_edit: QLineEdit, tool_name: str):
        """Browse for external tool executable."""
        if sys.platform == "win32":
            filter_text = "Executable Files (*.exe);;All Files (*)"
        else:
            filter_text = "All Files (*)"

        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {tool_name.title()}", "", filter_text
        )

        if file_path:
            line_edit.setText(file_path)


def test_in_gzdoom(self):
        """Test current PK3 in GZDoom."""
        self.test_in_source_port('gzdoom')


def test_in_zandronum(self):
        """Test current PK3 in Zandronum."""
        self.test_in_source_port('zandronum')


def test_in_source_port(self, port_name: str):
        """Test current sprite in specified source port."""
        port_path = self.external_tools.get(port_name, '')

        if not port_path or not os.path.exists(port_path):
            QMessageBox.warning(
                self, "Error",
                f"{port_name.title()} not configured or not found.\n"
                f"Configure path in Tools → External Tools"
            )
            return

        image_to_export = self.processed_image or self.current_image
        if not image_to_export:
            QMessageBox.warning(self, "Warning", "No image to test")
            return

        try:
            # Create temporary PK3
            import tempfile
            temp_dir = tempfile.mkdtemp()
            base_name = validate_sprite_name(self.name_edit.text())
            pk3_path = os.path.join(temp_dir, f"{base_name}_test.pk3")

            self.create_pk3_package(pk3_path, base_name, image_to_export)

            # Launch source port
            cmd = [port_path, "-file", pk3_path]
            subprocess.Popen(cmd, cwd=os.path.dirname(port_path))

            self.log_message(f"Launched {port_name.title()} with test PK3")

        except Exception as e:
            QMessageBox.critical(
    self, "Error", f"Failed to launch {
        port_name.title()}: {e}")
            self.log_message(f"{port_name.title()} launch error: {e}", "ERROR")


def launch_udb(self):
        """Launch Ultimate Doom Builder."""
        udb_path = self.external_tools.get('udb', '')

        if not udb_path or not os.path.exists(udb_path):
            QMessageBox.warning(
                self, "Error",
                "Ultimate Doom Builder not configured.\n"
                "Configure path in Tools → External Tools"
            )
            return

        try:
            subprocess.Popen([udb_path])
            self.log_message("Launched Ultimate Doom Builder")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch UDB: {e}")
            self.log_message(f"UDB launch error: {e}", "ERROR")

    # Log management


def clear_log(self):
        """Clear the log display."""
        self.log_text.clear()
        self.log_message("Log cleared")


def save_log(self):
        """Save log to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.log_message(f"Log saved to {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save log: {e}")

    # Help and about


def show_help(self):
        """Show help dialog."""
        help_dialog = HelpDialog(self)
        help_dialog.exec()


def show_about(self):
        """Show about dialog."""
        about_text = f"""
        <h2>{APP_NAME}</h2>
        <h3>Version {__version__}</h3>

        <p>A professional Doom sprite and texture creation toolkit with advanced features
        for classic Doom modding and modern source port development.</p>

        <p><b>Features:</b></p>
        <ul>
        <li>Professional PyQt6 interface with dark theme</li>
        <li>Advanced plugin system (JSON and Python)</li>
        <li>Real-time canvas with zoom/pan and pixel grid</li>
        <li>Sprite sheet management and animation preview</li>
        <li>PK3/WAD export with proper lump generation</li>
        <li>Integration with GZDoom, Zandronum, UDB</li>
        <li>Archive explorer and texture tools</li>
        </ul>

        <p><b>License:</b> MIT License</p>
        <p><b>Author:</b> Sprite Forge Team</p>

        <p>Built with PyQt6, Pillow, and NumPy</p>
        """

        QMessageBox.about(self, "About", about_text)

    # Settings management


def load_settings(self):
        """Load application settings."""
        settings = QSettings(ORG_NAME, APP_KEY)

        # Window geometry
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # UI settings
        self.name_edit.setText(settings.value("sprite_name", "SPRT"))

        sprite_type = settings.value("sprite_type", "Character")
        index = self.type_combo.findText(sprite_type)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)

        # Recent files
        recent = settings.value("recent_files", [])
        if isinstance(recent, list):
            self.recent_files = recent

        # External tools
        for tool in self.external_tools:
            path = settings.value(f"external_tools/{tool}", "")
            if path:
                self.external_tools[tool] = path

        # Processing options
        self.pixelate_check.setChecked(
            settings.value("pixelate", True, type=bool))
        self.palette_check.setChecked(
    settings.value(
        "doom_palette",
        True,
         type=bool))
        self.transparency_check.setChecked(settings.value(
            "preserve_transparency", True, type=bool))
        self.sharpen_check.setChecked(
            settings.value("sharpen", False, type=bool))
        self.denoise_check.setChecked(
            settings.value("denoise", False, type=bool))

        self.log_message("Settings loaded")


def save_settings(self):
        """Save application settings."""
        settings = QSettings(ORG_NAME, APP_KEY)

        # Window geometry
        settings.setValue("geometry", self.saveGeometry())

        # UI settings
        settings.setValue("sprite_name", self.name_edit.text())
        settings.setValue("sprite_type", self.type_combo.currentText())

        # Recent files
        settings.setValue("recent_files", self.recent_files)

        # External tools
        for tool, path in self.external_tools.items():
            settings.setValue(f"external_tools/{tool}", path)

        # Processing options
        settings.setValue("pixelate", self.pixelate_check.isChecked())
        settings.setValue("doom_palette", self.palette_check.isChecked())
        settings.setValue(
    "preserve_transparency",
     self.transparency_check.isChecked())
        settings.setValue("sharpen", self.sharpen_check.isChecked())
        settings.setValue("denoise", self.denoise_check.isChecked())


def closeEvent(self, event):
        """Handle application close event."""
        self.save_settings()

        # Clean up processing thread
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.terminate()
            self.processing_thread.wait(1000)

        self.log_message(f"{APP_NAME} closing")
        super().closeEvent(event)

# Unit Tests (embedded for --selftest)


# -------------------- SELF-TESTS (drop-in replacement) --------------------

def run_self_tests() -> bool:
    """Run embedded unit tests and return success status."""
    print(f"Running {APP_NAME} self-tests...")

    test_count = 0
    passed_count = 0

    def test_case(name: str, test_func: Callable) -> bool:
        nonlocal test_count, passed_count
        test_count += 1
        try:
            result = test_func()
            if result:
                print(f"✓ {name}")
                passed_count += 1
                return True
            else:
                print(f"✗ {name} - Test returned False")
                return False
        except Exception as e:
            print(f"✗ {name} - Exception: {e}")
            return False

    # Run all tests
    test_case("validate_sprite_name", test_validate_sprite_name)
    test_case("palette_remap_preserves_alpha", test_palette_remap_alpha)
    test_case("pk3_export_contains_textures_lump", test_pk3_export_contains_textures)
    test_case("wad_layout_info_json_is_valid", test_wad_layout_info_json)
    test_case("auto_slice_roundtrip", test_auto_slice_roundtrip)

    # Results
    print(f"\nTest Results: {passed_count}/{test_count} passed")
    if passed_count == test_count:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


# ---- Individual unit tests stay at TOP LEVEL (not nested) ----

def test_validate_sprite_name():
    assert validate_sprite_name("test") == "TEST"
    assert validate_sprite_name("a") == "AXXX"
    assert validate_sprite_name("toolong") == "TOOL"
    assert validate_sprite_name("") == "SPRT"
    assert validate_sprite_name("ABC1") == "ABC1"
    return True


def test_palette_remap_alpha():
    test_img = Image.new('RGBA', (10, 10), (255, 0, 0, 128))
    result = apply_doom_palette(test_img, preserve_transparency=True)
    pixels = list(result.getdata())
    has_alpha = any(p[3] < 255 for p in pixels)
    return result.mode == 'RGBA' and has_alpha


def test_pk3_export_contains_textures():
    import tempfile
    # Create test image
    test_img = Image.new('RGBA', (32, 32), (255, 0, 0, 255))
    # Create temporary PK3
    with tempfile.NamedTemporaryFile(suffix='.pk3', delete=False) as f:
        temp_path = f.name
    try:
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        window = SpriteForgeMainWindow()
        window.create_pk3_package(temp_path, "TEST", test_img)
        with zipfile.ZipFile(temp_path, 'r') as zf:
            files = zf.namelist()
            has_textures = 'TEXTURES' in files
            has_sprites = any(f.startswith('sprites/') for f in files)
            return has_textures and has_sprites
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def test_wad_layout_info_json():
    import tempfile, json
    test_img = Image.new('RGBA', (32, 32), (0, 255, 0, 255))
    with tempfile.TemporaryDirectory() as temp_dir:
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        window = SpriteForgeMainWindow()
        window.create_wad_layout(temp_dir, "TEST", test_img)
        info_path = os.path.join(temp_dir, "TEST_wad", "TEST_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            return (
                info.get("base_name") == "TEST"
                and "frames" in info
                and len(info["frames"]) > 0
            )
        return False


def test_auto_slice_roundtrip():
    # Create a 2x2 sheet with 32x32 tiles
    sheet = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    for i in range(2):
        for j in range(2):
            color = (i * 127 + 64, j * 127 + 64, 128, 255)
            square = Image.new('RGBA', (32, 32), color)
            sheet.paste(square, (j * 32, i * 32))
    manager = SpriteSheetManager()
    manager.load_sheet(sheet)
    frames = manager.auto_slice(32, 32)
    return len(frames) == 4

# ------------------ END SELF-TESTS (replacement block) ------------------



def build_with_pyinstaller():
    """
    Build standalone executable with PyInstaller.

    Run this command to create a standalone executable:

    pyinstaller --onefile --windowed --name="SpriteForgePro" sprite_forge_pro.py

    Options explained:
    --onefile: Create single executable file
    --windowed: Hide console window on Windows
    --name: Set output executable name

    Additional options you may want:
    --icon=icon.ico: Add custom icon (if available)
    --add-data="plugins;plugins": Include plugins folder
    --hidden-import=PIL._tkinter_finder: Ensure PIL works

    The executable will be created in the dist/ folder.

    Note: The first run will be slower as PyInstaller extracts files.
    Subsequent runs will be faster.
    """
    pass


def main():
    """Main application entry point with command line support."""
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} v{__version__} - Professional Doom Sprite Creation Toolkit"
    )

    parser.add_argument(
        '--selftest',
        action='store_true',
        help='Run embedded unit tests and exit'
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'{APP_NAME} v{__version__}'
    )

    parser.add_argument(
        'input_file',
        nargs='?',
        help='Image file to load on startup'
    )

    args = parser.parse_args()

    # Handle self-test mode
    if args.selftest:
        try:
            # Need minimal QApplication for some tests
            app = QApplication([])
            success = run_self_tests()
            return 0 if success else 1
        except Exception as e:
            print(f"Self-test failed: {e}")
            return 1

    # Check GUI availability
    if not GUI_AVAILABLE:
        print("Error: PyQt6 not available")
        print("Install with: pip install PyQt6")
        return 1

    # Create and run main application
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(ORG_NAME)

    # Set application-wide style
    app.setStyle('Fusion')  # Modern cross-platform style

    # Create main window
    window = SpriteForgeMainWindow()

    # Load input file if provided
    if args.input_file and os.path.exists(args.input_file):
        window.load_image_file(args.input_file)

    window.show()

    # Setup graceful shutdown
    import signal

    def signal_handler(sig, frame):
        window.close()
        app.quit()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run application
    try:
        return app.exec()
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())  # !/usr/bin/env python3
"""
Sprite Forge Pro 2025 v1.0.0 - Professional Doom Sprite and Texture Production Suite
====================================================================================

A complete sprite and texture creation toolkit for classic Doom modding with:
- Professional PyQt6 interface with dark theme
- Advanced plugin system supporting JSON and code-based plugins
- Real-time canvas with zoom/pan and pixel grid
- Comprehensive sprite sheet management and animation preview
- PK3/WAD export with proper lump generation
- Integration with GZDoom, Zandronum, and Ultimate Doom Builder
- Archive explorer for WAD/PK3 browsing and editing
- Texture composition and palette management tools
- Batch processing and automation features

MIT License - Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software.

Author: Sprite Forge Team
Version: 1.0.0
Python: 3.10+
Dependencies: PyQt6, Pillow, numpy
"""


# Version and constants
__version__ = "1.0.0"
APP_NAME = "Sprite Forge Pro 2025"
ORG_NAME = "SpriteForge"
APP_KEY = "SpriteForgePro2025"

# Auto-install dependencies


def ensure_dependencies():
    """Auto-install required packages if missing."""
    required = {
        'pillow': 'PIL',
        'numpy': 'numpy',
        'PyQt6': 'PyQt6'
    }

    missing = []
    for package, import_name in required.items():
        try:
            if import_name == 'PIL':
                import PIL
            else:
                __import__(import_name)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Installing required packages: {', '.join(missing)}")
        for package in missing:
            try:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}: {e}")
                sys.exit(1)


ensure_dependencies()

# Import GUI framework
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox,
        QSpinBox, QSlider, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
        QTabWidget, QGroupBox, QScrollArea, QSplitter, QStatusBar,
        QMenuBar, QMenu, QToolBar, QFrame, QSpacerItem, QSizePolicy,
        QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
        QDialog, QDialogButtonBox, QFormLayout, QPlainTextEdit
    )
    from PyQt6.QtCore import (
        Qt, QThread, pyqtSignal, QTimer, QSettings, QSize, QPoint,
        QRect, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
    )
    from PyQt6.QtGui import (
        QPixmap, QPainter, QColor, QFont, QAction, QIcon, QPalette,
        QTransform, QPen, QBrush, QKeySequence, QShortcut, QImage,
        QFontMetrics, QPainterPath
    )
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"PyQt6 not available: {e}")
    GUI_AVAILABLE = False
    sys.exit(1)


# Logging setup

def setup_logging():
    """Setup logging to file and console."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "sprite_forge.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


logger = setup_logging()

# Doom palette constants
DOOM_PALETTE = [
    (0, 0, 0), (31, 23, 11), (23, 15, 7), (75, 75, 75), (255, 255, 255),
    (27, 27, 27), (19, 19, 19), (11, 11, 11), (199, 199, 199), (119, 119, 119),
    (83, 83, 83), (47, 47, 47), (255, 155, 0), (231, 119, 0), (203, 91, 0),
    (175, 71, 0), (143, 59, 0), (119, 47, 0), (91, 35, 0), (71, 27, 0),
    (199, 0, 0), (167, 0, 0), (139, 0, 0), (107, 0, 0), (75, 75, 0),
    (0, 255, 0), (0, 231, 0), (0, 203, 0), (0, 175, 0), (0, 143, 0),
    (0, 119, 0), (0, 91, 0), (0, 71, 0)
] + [(i, i, i) for i in range(32, 256, 8)]  # Fill to 256 colors

# Core utility functions


def validate_sprite_name(name: str) -> str:
    """Ensure sprite name is exactly 4 uppercase characters."""
    if not name:
        return "SPRT"

    cleaned = ''.join(c for c in name.upper() if c.isalnum())
    if len(cleaned) < 4:
        cleaned = cleaned.ljust(4, 'X')
    elif len(cleaned) > 4:
        cleaned = cleaned[:4]
    return cleaned


def pixelate_image(image: Image.Image, factor: int = 4) -> Image.Image:
    """Apply pixelation effect."""
    if factor <= 1:
        return image.copy()

    w, h = image.size
    small_w, small_h = max(1, w // factor), max(1, h // factor)
    small = image.resize((small_w, small_h), Image.Resampling.NEAREST)
    return small.resize((w, h), Image.Resampling.NEAREST)


def apply_doom_palette(
    image: Image.Image,
     preserve_transparency: bool = True) -> Image.Image:
    """Apply Doom-style palette quantization with proper alpha handling."""
    src = image.convert('RGBA')
    pixels = list(src.getdata())
    new_pixels = []

    for r, g, b, a in pixels:
        if a == 0 and preserve_transparency:
            new_pixels.append((0, 0, 0, 0))
            continue

        # Find closest doom palette color
        best_color = DOOM_PALETTE[0]
        best_distance = float('inf')

        # Use first 32 colors for better results
        for doom_color in DOOM_PALETTE[:32]:
            dr, dg, db = r - doom_color[0], g - \
                doom_color[1], b - doom_color[2]
            distance = dr * dr + dg * dg + db * db
            if distance < best_distance:
                best_distance = distance
                best_color = doom_color

        new_pixels.append((*best_color, a))

    result = Image.new('RGBA', src.size)
    result.putdata(new_pixels)
    return result


def create_sprite_rotations(
    image: Image.Image, num_rotations: int = 8) -> Dict[str, Image.Image]:
    """Generate sprite rotations from base image."""
    rotations = {}

    for i in range(1, num_rotations + 1):
        if i <= 4:
            # Front and side views
            rotations[str(i)] = image.copy()
        else:
            # Mirror for back views
            rotations[str(i)] = ImageOps.mirror(image)

    return rotations

# Frame and animation structures


@dataclass
class SpriteFrame:
    """Represents a single sprite frame."""
    name: str
    image: Image.Image
    offset_x: int = 0
    offset_y: int = 0
    duration: int = 100  # milliseconds


@dataclass
class SpriteAnimation:
    """Represents a sprite animation sequence."""
    name: str
    frames: List[SpriteFrame]
    loop: bool = True
    fps: float = 10.0

# Plugin system


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

class JSONConfigPlugin(BasePlugin):
    """Plugin created from JSON configuration."""
    
def __init__(self, config: Dict, file_path: str):
        self.config = config
        self.file_path = file_path
    
def get_info(self) -> PluginInfo:
        return PluginInfo(
            name=self.config["name"],
            version=self.config["version"],
            description=self.config["description"],
            author=self.config["author"],
            category=self.config["category"],
            file_path=self.file_path
        )
    
def get_parameters(self) -> Dict[str, Any]:
        return self.config.get("parameters", {})
    
def process_image(self, image: Image.Image, **kwargs) -> Image.Image:
        """Process image according to JSON configuration."""
        processing = self.config.get("processing", {})
        proc_type = processing.get("type", "")
        
        if proc_type == "hsv_transform":
            return self._process_hsv_transform(image, kwargs)
        elif proc_type == "edge_filter":
            return self._process_edge_filter(image, kwargs)
        elif proc_type == "isometric_transform":
            return self._process_isometric(image, kwargs)
        
        return image
    
def _process_hsv_transform(self, image: Image.Image, params: Dict) -> Image.Image:
        """Process HSV color transformation."""
        img = image.convert('RGBA')
        pixels = list(img.getdata())
        new_pixels = []
        
        hue_shift = params.get('hue_shift', 0) / 360.0
        saturation = params.get('saturation', 1.0)
        lightness = params.get('lightness', 1.0)
        preserve_skin = params.get('preserve_skin', False)
        
        for r, g, b, a in pixels:
            if a == 0:
                new_pixels.append((r, g, b, a))
                continue
            
            # Skip skin tones if requested (rough approximation)
            if preserve_skin and self._is_skin_tone(r, g, b):
                new_pixels.append((r, g, b, a))
                continue
                
            h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            h = (h + hue_shift) % 1.0
            s = min(1.0, s * saturation)
            v = min(1.0, v * lightness)
            
            nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
            new_pixels.append((int(nr*255), int(ng*255), int(nb*255), a))
        
        result = Image.new('RGBA', img.size)
        result.putdata(new_pixels)
        return result
    
def _is_skin_tone(self, r: int, g: int, b: int) -> bool:
        """Rough skin tone detection."""
        # Simple heuristic for skin tones
        return (r > g > b) and (r - g < 50) and (g - b < 50) and r > 100
    
def _process_edge_filter(self, image: Image.Image, params: Dict) -> Image.Image:
        """Apply edge enhancement filter."""
        strength = params.get('strength', 1.0)
        threshold = params.get('threshold', 50)
        
        img = image.convert('RGBA')
        
        # Apply edge enhancement
        enhanced = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        if strength != 1.0:
            enhanced = Image.blend(img, enhanced, min(1.0, strength))
        
        return enhanced
    
def _process_isometric(self, image: Image.Image, params: Dict) -> Image.Image:
        """Create simple isometric preview."""
        angle = params.get('angle', 45)
        height_factor = params.get('height', 1.0)
        
        # Simple perspective transformation
        img = image.convert('RGBA')
        w, h = img.size
        
        # Create a slightly larger canvas
        new_w, new_h = int(w * 1.2), int(h * 1.2)
        result = Image.new('RGBA', (new_w, new_h), (0, 0, 0, 0))
        
        # Apply simple brightness adjustment for depth effect
        brightened = ImageEnhance.Brightness(img).enhance(0.8 + height_factor * 0.2)
        
        # Center the image
        offset_x = (new_w - w) // 2
        offset_y = (new_h - h) // 2
        result.paste(brightened, (offset_x, offset_y))
        
        return result

class PluginManager:
    """Manages plugin loading and execution."""
    
def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = plugin_dir
        self.plugins: Dict[str, BasePlugin] = {}
        self.load_plugins()
    
def load_plugins(self):
        """Load all plugins from the plugins directory."""
        plugin_path = Path(self.plugin_dir)
        if not plugin_path.exists():
            plugin_path.mkdir(exist_ok=True)
            self.create_example_plugins()
        
        self.plugins.clear()
        
        # Load JSON plugins
        for json_file in plugin_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                plugin = JSONConfigPlugin(config, str(json_file))
                info = plugin.get_info()
                self.plugins[info.name] = plugin
                logger.info(f"Loaded JSON plugin: {info.name}")
                
            except Exception as e:
                logger.error(f"Failed to load plugin {json_file}: {e}")
        
        # Load Python plugins (for future extensibility)
        for py_file in plugin_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    if hasattr(module, 'create_plugin'):
                        plugin = module.create_plugin()
                        if isinstance(plugin, BasePlugin):
                            info = plugin.get_info()
                            self.plugins[info.name] = plugin
                            logger.info(f"Loaded Python plugin: {info.name}")
                            
            except Exception as e:
                logger.error(f"Failed to load Python plugin {py_file}: {e}")
    
def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get plugin by name."""
        return self.plugins.get(name)
    
def list_plugins(self) -> List[PluginInfo]:
        """List all loaded plugins."""
        return [plugin.get_info() for plugin in self.plugins.values()]
    
def create_example_plugins(self):
        """Create example plugin configurations on first run."""
        # AI Recolor Plugin
        ai_recolor = {
            "name": "AI Recolor",
            "version": "1.0.0",
            "description": "Intelligent color transformation using HSV shifting",
            "author": "Sprite Forge Team",
            "category": "Color",
            "parameters": {
                "hue_shift": {"type": "slider", "min": -180, "max": 180, "default": 0, "label": "Hue Shift"},
                "saturation": {"type": "slider", "min": 0.0, "max": 2.0, "default": 1.0, "step": 0.1, "label": "Saturation"},
                "lightness": {"type": "slider", "min": 0.0, "max": 2.0, "default": 1.0, "step": 0.1, "label": "Lightness"},
                "preserve_skin": {"type": "checkbox", "default": True, "label": "Preserve Skin Tones"}
            },
            "processing": {
                "type": "hsv_transform"
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
                "strength": {"type": "slider", "min": 0.0, "max": 5.0, "default": 1.0, "step": 0.1, "label": "Enhancement Strength"},
                "threshold": {"type": "slider", "min": 0, "max": 255, "default": 50, "label": "Edge Threshold"}
            },
            "processing": {
                "type": "edge_filter"
            }
        }
        
        # Isometric Preview Plugin
        iso_preview = {
            "name": "Isometric Preview",
            "version": "1.0.0",
            "description": "Generate simple isometric preview",
            "author": "Sprite Forge Team",
            "category": "Preview",
            "parameters": {
                "angle": {"type": "slider", "min": 0, "max": 360, "default": 45, "label": "View Angle"},
                "height": {"type": "slider", "min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1, "label": "Height Factor"}
            },
            "processing": {
                "type": "isometric_transform"
            }
        }
        
        plugins = [ai_recolor, edge_enhance, iso_preview]
        for plugin_config in plugins:
            filename = f"{plugin_config['name'].lower().replace(' ', '_')}.json"
            file_path = Path(self.plugin_dir) / filename
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(plugin_config, f, indent=2)
                logger.info(f"Created example plugin: {filename}")
            except Exception as e:
                logger.error(f"Failed to create example plugin {filename}: {e}")

# Enhanced Canvas Widget
class ImageCanvas(QWidget):
    """Professional image display widget with zoom/pan/grid capabilities."""
    
    imageClicked = pyqtSignal(int, int)
    imageChanged = pyqtSignal()
    
def __init__(self):
        super().__init__()
        self.image = None
        self.pixmap = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.last_pan_point = None
        self.show_grid = False
        self.show_onion_skin = False
        self.onion_image = None
        self.grid_size = 16
        
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Setup styling
        self.setStyleSheet("""
            ImageCanvas {
                background-color: #2d2d2d;
                border: 1px solid #555;
            }
        """)
        
        # Keyboard shortcuts
        self.setup_shortcuts()
    
def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Grid toggle
        grid_shortcut = QShortcut(QKeySequence("G"), self)
        grid_shortcut.activated.connect(self.toggle_grid)
        
        # Zoom shortcuts
        zoom_in_shortcut = QShortcut(QKeySequence("Ctrl++"), self)
        zoom_in_shortcut.activated.connect(self.zoom_in)
        
        zoom_out_shortcut = QShortcut(QKeySequence("Ctrl+-"), self)
        zoom_out_shortcut.activated.connect(self.zoom_out)
        
        reset_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        reset_shortcut.activated.connect(self.reset_view)
    
def set_image(self, image: Image.Image):
        """Set the image to display."""
        self.image = image
        self.update_pixmap()
        self.reset_view()
        self.update()
        self.imageChanged.emit()
    
def set_onion_skin_image(self, image: Optional[Image.Image]):
        """Set onion skin overlay image."""
        self.onion_image = image
        self.update()
    
def toggle_grid(self):
        """Toggle pixel grid display."""
        self.show_grid = not self.show_grid
        self.update()
    
def toggle_onion_skin(self):
        """Toggle onion skin display."""
        self.show_onion_skin = not self.show_onion_skin
        self.update()
    
def update_pixmap(self):
        """Convert PIL image to QPixmap."""
        if not self.image:
            self.pixmap = None
            return
        
        # Convert PIL to QPixmap
        img_array = np.array(self.image)
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                h, w, ch = img_array.shape
                bytes_per_line = ch * w
                qt_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
            else:  # RGB
                h, w, ch = img_array.shape
                bytes_per_line = ch * w
                qt_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:  # Grayscale
            h, w = img_array.shape
            bytes_per_line = w
            qt_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        self.pixmap = QPixmap.fromImage(qt_image)
    
def reset_view(self):
        """Reset zoom and pan to fit image."""
        if not self.pixmap:
            return
        
        widget_size = self.size()
        image_size = self.pixmap.size()
        
        if image_size.width() == 0 or image_size.height() == 0:
            return
        
        scale_x = widget_size.width() / image_size.width()
        scale_y = widget_size.height() / image_size.height()
        self.scale_factor = min(scale_x, scale_y, 1.0) * 0.9  # Leave some margin
        
        scaled_w = image_size.width() * self.scale_factor
        scaled_h = image_size.height() * self.scale_factor
        
        self.offset_x = (widget_size.width() - scaled_w) / 2
        self.offset_y = (widget_size.height() - scaled_h) / 2
        
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
        
        # Fill background
        painter.fillRect(self.rect(), QColor(45, 45, 45))
        
        if self.pixmap:
            # Draw checkerboard background for transparency
            self.draw_transparency_background(painter)
            
            # Apply transform
            painter.save()
            painter.translate(self.offset_x, self.offset_y)
            painter.scale(self.scale_factor, self.scale_factor)
            
            # Draw onion skin first (if enabled)
            if self.show_onion_skin and self.onion_image:
                onion_pixmap = self.image_to_pixmap(self.onion_image)
                if onion_pixmap:
                    painter.setOpacity(0.3)
                    painter.drawPixmap(0, 0, onion_pixmap)
                    painter.setOpacity(1.0)
            
            # Draw main image
            painter.drawPixmap(0, 0, self.pixmap)
            
            # Draw grid if enabled
            if self.show_grid and self.scale_factor > 4:
                self.draw_grid(painter)
            
            painter.restore()
        else:
            # Draw placeholder text
            painter.setPen(QColor(128, 128, 128))
            font = QFont("Arial", 14)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded\nDrag & Drop or use File > Open")
    
def draw_transparency_background(self, painter):
        """Draw checkerboard pattern for transparent areas."""
        if not self.pixmap:
            return
        
        # Calculate visible image area
        img_rect = QRect(
            int(self.offset_x),
            int(self.offset_y),
            int(self.pixmap.width() * self.scale_factor),
            int(self.pixmap.height() * self.scale_factor)
        )
        
        # Checkerboard pattern
        tile_size = max(8, int(16 * self.scale_factor))
        painter.save()
        
        # Light tiles
        painter.fillRect(img_rect, QColor(220, 220, 220))
        
        # Dark tiles
        painter.setBrush(QBrush(QColor(200, 200, 200)))
        for y in range(img_rect.top(), img_rect.bottom(), tile_size):
            for x in range(img_rect.left(), img_rect.right(), tile_size):
                tile_x = (x - img_rect.left()) // tile_size
                tile_y = (y - img_rect.top()) // tile_size
                if (tile_x + tile_y) % 2:
                    painter.fillRect(x, y, tile_size, tile_size, QColor(200, 200, 200))
        
        painter.restore()
    
def draw_grid(self, painter):
        """Draw pixel grid overlay."""
        if not self.pixmap:
            return
        
        painter.save()
        painter.setPen(QPen(QColor(100, 100, 100, 128), 1))
        
        w, h = self.pixmap.width(), self.pixmap.height()
        
        # Draw vertical lines
        for x in range(0, w + 1, self.grid_size):
            painter.drawLine(x, 0, x, h)
        
        # Draw horizontal lines
        for y in range(0, h + 1, self.grid_size):
            painter.drawLine(0, y, w, y)
        
        painter.restore()
    
def image_to_pixmap(self, image: Image.Image) -> Optional[QPixmap]:
        """Convert PIL Image to QPixmap."""
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
                h, w, ch = img_array.shape
                bytes_per_line = ch * w
                qt_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
            elif len(img_array.shape) == 3:  # RGB
                h, w, ch = img_array.shape
                bytes_per_line = ch * w
                qt_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                return None
            
            return QPixmap.fromImage(qt_image)
        except Exception:
            return None
    
def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_pan_point = event.position()
        elif event.button() == Qt.MouseButton.RightButton:
            # Convert to image coordinates and emit click
            img_pos = self.widget_to_image_coords(event.position())
            if img_pos:
                self.imageClicked.emit(int(img_pos.x()), int(img_pos.y()))
    
def mouseMoveEvent(self, event):
        """Handle mouse move events for panning."""
        if self.dragging and self.last_pan_point:
            delta = event.position() - self.last_pan_point
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_pan_point = event.position()
            self.update()
    
def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.last_pan_point = None
    
def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        
        # Zoom towards mouse cursor
        mouse_pos = event.position()
        old_scale = self.scale_factor
        self.scale_factor *= zoom_factor
        self.scale_factor = max(0.1, min(20.0, self.scale_factor))  # Clamp zoom
        
        # Adjust offset to zoom towards cursor
        scale_change = self.scale_factor / old_scale - 1
        self.offset_x -= (mouse_pos.x() - self.offset_x) * scale_change
        self.offset_y -= (mouse_pos.y() - self.offset_y) * scale_change
        
        self.update()
    
def widget_to_image_coords(self, widget_pos):
        """Convert widget coordinates to image coordinates."""
        if not self.pixmap:
            return None
        
        # Apply inverse transform
        img_x = (widget_pos.x() - self.offset_x) / self.scale_factor
        img_y = (widget_pos.y() - self.offset_y) / self.scale_factor
        
        # Check bounds
        if 0 <= img_x < self.pixmap.width() and 0 <= img_y < self.pixmap.height():
            return QPoint(int(img_x), int(img_y))
        
        return None

# Sprite Sheet Manager
class SpriteSheetManager:
    """Manages sprite sheet operations like slicing and assembly."""
    
def __init__(self):
        self.sheet_image = None
        self.frames = []
        self.grid_size = (64, 64)
        self.auto_crop = True
    
def load_sheet(self, image: Image.Image):
        """Load sprite sheet image."""
        self.sheet_image = image
        self.frames = []
    
def auto_slice(self, grid_width: int, grid_height: int) -> List[Image.Image]:
        """Automatically slice sprite sheet into frames."""
        if not self.sheet_image:
            return []
        
        frames = []
        sheet_w, sheet_h = self.sheet_image.size
        
        for y in range(0, sheet_h, grid_height):
            for x in range(0, sheet_w, grid_width):
                # Extract frame
                frame = self.sheet_image.crop((x, y, x + grid_width, y + grid_height))
                
                # Auto-crop if enabled
                if self.auto_crop:
                    bbox = frame.getbbox()
                    if bbox:
                        frame = frame.crop(bbox)
                
                # Skip empty frames
                if frame.getbbox():
                    frames.append(frame)
        
        self.frames = frames
        return frames
    
def manual_slice(self, regions: List[Tuple[int, int, int, int]]) -> List[Image.Image]:
        """Manually slice sprite sheet using provided regions."""
        if not self.sheet_image:
            return []
        
        frames = []
        for x, y, w, h in regions:
            frame = self.sheet_image.crop((x, y, x + w, y + h))
            if self.auto_crop:
                bbox = frame.getbbox()
                if bbox:
                    frame = frame.crop(bbox)
            frames.append(frame)
        
        self.frames = frames
        return frames
    
def assemble_sheet(self, frames: List[Image.Image], cols: int = 8) -> Image.Image:
        """Assemble frames into a sprite sheet."""
        if not frames:
            return Image.new('RGBA', (64, 64), (0, 0, 0, 0))
        
        # Find maximum frame size
        max_w = max(frame.width for frame in frames)
        max_h = max(frame.height for frame in frames)
        
        # Calculate sheet dimensions
        rows = (len(frames) + cols - 1) // cols
        sheet_w = cols * max_w
        sheet_h = rows * max_h
        
        # Create sheet
        sheet = Image.new('RGBA', (sheet_w, sheet_h), (0, 0, 0, 0))
        
        # Place frames
        for i, frame in enumerate(frames):
            col = i % cols
            row = i // cols
            
            x = col * max_w + (max_w - frame.width) // 2  # Center frame
            y = row * max_h + (max_h - frame.height) // 2
            
            sheet.paste(frame, (x, y))
        
        return sheet

# Animation Preview Widget
class AnimationPreview(QWidget):
    """Widget for previewing sprite animations."""
    
def __init__(self):
        super().__init__()
        self.animation = None
        self.current_frame = 0
        self.playing = False
        self.fps = 10.0
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        self.setMinimumSize(200, 200)
        self.setStyleSheet("background-color: #3a3a3a; border: 1px solid #555;")
    
def set_animation(self, frames: List[Image.Image], fps: float = 10.0):
        """Set animation frames."""
        self.frames = frames
        self.fps = fps
        self.current_frame = 0
        self.update_timer()
        self.update()
    
def play(self):
        """Start animation playback."""
        if self.frames and len(self.frames) > 1:
            self.playing = True
            self.timer.start()
    
def pause(self):
        """Pause animation playback."""
        self.playing = False
        self.timer.stop()
    
def stop(self):
        """Stop animation and reset to first frame."""
        self.playing = False
        self.timer.stop()
        self.current_frame = 0
        self.update()
    
def next_frame(self):
        """Advance to next frame."""
        if self.frames and len(self.frames) > 0:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.update()
    
def prev_frame(self):
        """Go to previous frame."""
        if self.frames and len(self.frames) > 0:
            self.current_frame = (self.current_frame - 1) % len(self.frames)
            self.update()
    
def set_fps(self, fps: float):
        """Set animation FPS."""
        self.fps = max(1.0, fps)
        self.update_timer()
    
def update_timer(self):
        """Update timer interval based on FPS."""
        if self.fps > 0:
            interval = int(1000 / self.fps)
            self.timer.setInterval(interval)
    
def paintEvent(self, event):
        """Paint current animation frame."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(58, 58, 58))
        
        if hasattr(self, 'frames') and self.frames and 0 <= self.current_frame < len(self.frames):
            frame = self.frames[self.current_frame]
            
            # Convert PIL to QPixmap
            try:
                img_array = np.array(frame)
                if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
                    h, w, ch = img_array.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
                elif len(img_array.shape) == 3:  # RGB
                    h, w, ch = img_array.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                else:
                    return
                
                pixmap = QPixmap.fromImage(qt_image)
                
                # Scale to fit widget
                widget_size = self.size()
                pixmap_size = pixmap.size()
                
                if pixmap_size.width() > 0 and pixmap_size.height() > 0:
                    scale_x = widget_size.width() / pixmap_size.width()
                    scale_y = widget_size.height() / pixmap_size.height()
                    scale = min(scale_x, scale_y) * 0.8  # Leave margin
                    
                    scaled_pixmap = pixmap.scaled(
                        int(pixmap_size.width() * scale),
                        int(pixmap_size.height() * scale),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.FastTransformation
                    )
                    
                    # Center the pixmap
                    x = (widget_size.width() - scaled_pixmap.width()) // 2
                    y = (widget_size.height() - scaled_pixmap.height()) // 2
                    painter.drawPixmap(x, y, scaled_pixmap)
                
            except Exception as e:
                logger.error(f"Error painting animation frame: {e}")
        else:
            # Draw placeholder
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Animation")

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
            logger.error(f"Processing thread error: {e}")
            self.error.emit(str(e))

# Plugin Widget
class PluginWidget(QGroupBox):
    """Widget for plugin parameters and execution."""
    
    applyRequested = pyqtSignal(BasePlugin, dict)
    
def __init__(self, plugin: BasePlugin):
        super().__init__(plugin.get_info().name)
        self.plugin = plugin
        self.param_widgets = {}
        self.setup_ui()
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 10px;
                background-color: #353535;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #ffffff;
            }
        """)
    
def setup_ui(self):
        """Setup plugin parameter UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Plugin info
        info = self.plugin.get_info()
        info_text = f"v{info.version} - {info.description}"
        if len(info_text) > 40:
            info_text = info_text[:40] + "..."
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: #aaa; font-size: 10px; font-weight: normal;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Parameters
        params = self.plugin.get_parameters()
        if params:
            for param_name, param_config in params.items():
                param_widget = self.create_parameter_widget(param_name, param_config)
                if param_widget:
                    layout.addWidget(param_widget)
        
        # Apply button
        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover { background-color: #5ba0f2; }
            QPushButton:pressed { background-color: #3a80d2; }
        """)
        apply_btn.clicked.connect(self.apply_plugin)
        layout.addWidget(apply_btn)
    
def create_parameter_widget(self, name: str, config: Dict) -> Optional[QWidget]:
        """Create widget for parameter based on config."""
        param_type = config.get("type", "")
        
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        label = QLabel(config.get("label", name))
        label.setStyleSheet("color: #ffffff; font-weight: normal;")
        label.setMinimumWidth(80)
        layout.addWidget(label)
        
        if param_type == "slider":
            widget = QSlider(Qt.Orientation.Horizontal)
            min_val = config.get("min", 0)
            max_val = config.get("max", 100)
            default_val = config.get("default", min_val)
            step = config.get("step", 1)
            
            # Handle float sliders by scaling
            if isinstance(step, float) or isinstance(min_val, float) or isinstance(max_val, float):
                self.is_float_slider = True
                self.float_scale = 100
                widget.setMinimum(int(min_val * self.float_scale))
                widget.setMaximum(int(max_val * self.float_scale))
                widget.setValue(int(default_val * self.float_scale))
            else:
                self.is_float_slider = False
                self.float_scale = 1
                widget.setMinimum(min_val)
                widget.setMaximum(max_val)
                widget.setValue(default_val)
            
            self.param_widgets[name] = widget
            
            value_label = QLabel()
            value_label.setStyleSheet("color: #ffffff; font-weight: normal; min-width: 40px;")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            
            def update_label(value):
                if hasattr(self, 'is_float_slider') and self.is_float_slider:
                    display_val = value / self.float_scale
                    value_label.setText(f"{display_val:.2f}")
                else:
                    value_label.setText(str(value))
            
            update_label(widget.value())
            widget.valueChanged.connect(update_label)
            
            layout.addWidget(widget, 1)
            layout.addWidget(value_label)
            
        elif param_type == "checkbox":
            widget = QCheckBox()
            widget.setChecked(config.get("default", False))
            widget.setStyleSheet("color: #ffffff;")
            self.param_widgets[name] = widget
            layout.addWidget(widget)
            
        elif param_type == "combo":
            widget = QComboBox()
            options = config.get("options", [])
            widget.addItems(options)
            default_idx = config.get("default", 0)
            if isinstance(default_idx, str):
                default_idx = options.index(default_idx) if default_idx in options else 0
            widget.setCurrentIndex(default_idx)
            widget.setStyleSheet("""
                QComboBox { color: #ffffff; background-color: #404040; border: 1px solid #666; }
                QComboBox::drop-down { border: none; }
                QComboBox::down-arrow { color: #ffffff; }
            """)
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
                value = widget.value()
                if hasattr(self, 'is_float_slider') and self.is_float_slider:
                    value = value / self.float_scale
                params[name] = value
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
        return params
    
def apply_plugin(self):
        """Emit apply signal with plugin and parameters."""
        params = self.get_parameters()
        self.applyRequested.emit(self.plugin, params)

# Archive Explorer
class ArchiveExplorer(QWidget):
    """Widget for exploring WAD and PK3 archives."""
    
def __init__(self):
        super().__init__()
        self.current_archive = None
        self.setup_ui()
    
def setup_ui(self):
        """Setup archive explorer UI."""
        layout = QVBoxLayout(self)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        open_btn = QPushButton("Open Archive")
        open_btn.clicked.connect(self.open_archive)
        toolbar.addWidget(open_btn)
        
        extract_btn = QPushButton("Extract")
        extract_btn.clicked.connect(self.extract_selected)
        toolbar.addWidget(extract_btn)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # File tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type", "Size"])
        self.tree.setStyleSheet("""
            QTreeWidget { 
                background-color: #2d2d2d; 
                color: #ffffff;
                border: 1px solid #555;
            }
            QTreeWidget::item:selected { background-color: #4a90e2; }
        """)
        layout.addWidget(self.tree)
    
def open_archive(self):
        """Open WAD or PK3 archive."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Archive", "",
            "Archive Files (*.wad *.pk3 *.zip);;All Files (*)"
        )
        
        if file_path:
            self.load_archive(file_path)
    
def load_archive(self, file_path: str):
        """Load archive contents into tree."""
        self.current_archive = file_path
        self.tree.clear()
        
        try:
            if file_path.lower().endswith(('.pk3', '.zip')):
                self.load_zip_archive(file_path)
            elif file_path.lower().endswith('.wad'):
                self.load_wad_archive(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load archive: {e}")
    
def load_zip_archive(self, file_path: str):
        """Load PK3/ZIP archive contents."""
        with zipfile.ZipFile(file_path, 'r') as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                
                parts = info.filename.split('/')
                current_item = None
                current_parent = None
                
                # Build tree structure
                for part in parts[:-1]:  # Directories
                    found = False
                    parent = current_parent or self.tree.invisibleRootItem()
                    
                    for i in range(parent.childCount()):
                        child = parent.child(i)
                        if child.text(0) == part:
                            current_parent = child
                            found = True
                            break
                    
                    if not found:
                        item = QTreeWidgetItem([part, "Directory", ""])
                        if current_parent:
                            current_parent.addChild(item)
                        else:
                            self.tree.addTopLevelItem(item)
                        current_parent = item
                
                # Add file
                filename = parts[-1]
                file_type = self.get_file_type(filename)
                size_str = f"{info.file_size} bytes"
                
                file_item = QTreeWidgetItem([filename, file_type, size_str])
                if current_parent:
                    current_parent.addChild(file_item)
                else:
                    self.tree.addTopLevelItem(file_item)
    
def load_wad_archive(self, file_path: str):
        """Load WAD archive contents (simplified)."""
        # This is a placeholder - real WAD parsing would be more complex
        item = QTreeWidgetItem(["WAD Parsing", "TODO", "Implement full WAD support"])
        self.tree.addTopLevelItem(item)
    
def get_file_type(self, filename: str) -> str:
        """Determine file type from extension."""
        ext = Path(filename).suffix.lower()
        type_map = {
            '.png': 'Image',
            '.jpg': 'Image',
            '.jpeg': 'Image',
            '.bmp': 'Image',
            '.wav': 'Audio',
            '.ogg': 'Audio',
            '.mp3': 'Audio',
            '.txt': 'Text',
            '.cfg': 'Config',
            '.acs': 'Script',
            '.wad': 'Archive'
        }
        return type_map.get(ext, 'Unknown')
    
def extract_selected(self):
        """Extract selected files."""
        selected = self.tree.selectedItems()
        if not selected or not self.current_archive:
            QMessageBox.information(self, "Info", "Select files to extract")
            return
        
        # TODO: Implement extraction
        QMessageBox.information(self, "TODO", "Extraction not yet implemented")

# Help Dialog
class HelpDialog(QDialog):
    """Help dialog with quickstart and shortcuts."""
    
def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sprite Forge Pro - Help")
        self.setMinimumSize(600, 500)
        self.setup_ui()
    
def setup_ui(self):
        """Setup help dialog UI."""
        layout = QVBoxLayout(self)
        
        # Tab widget
        tabs = QTabWidget()
        
        # Quickstart tab
        quickstart = QTextEdit()
        quickstart.setReadOnly(True)
        quickstart.setHtml("""
        <h2>Quickstart Guide</h2>
        <h3>Basic Workflow</h3>
        <ol>
        <li><b>Load Image:</b> File → Open Image (Ctrl+O) or drag & drop</li>
        <li><b>Configure Sprite:</b> Set 4-character name, type, and size</li>
        <li><b>Apply Processing:</b> Enable pixelation, Doom palette, etc.</li>
        <li><b>Preview:</b> Click "Preview Changes" to see results</li>
        <li><b>Use Plugins:</b> Apply color transforms, effects, filters</li>
        <li><b>Export:</b> Create PK3 package or WAD layout (Ctrl+E)</li>
        </ol>
        
        <h3>Canvas Controls</h3>
        <ul>
        <li><b>Pan:</b> Left-click and drag</li>
        <li><b>Zoom:</b> Mouse wheel or Ctrl +/-</li>
        <li><b>Reset View:</b> Ctrl+0</li>
        <li><b>Toggle Grid:</b> G key</li>
        <li><b>Pixel Coordinates:</b> Right-click on image</li>
        </ul>
        
        <h3>Sprite Sheets</h3>
        <ul>
        <li>Load sheet in Canvas tab</li>
        <li>Switch to Sheet tab for slicing tools</li>
        <li>Use auto-slice with grid size or manual regions</li>
        <li>Preview animations in Preview tab</li>
        </ul>
        """)
        tabs.addTab(quickstart, "Quickstart")
        
        # Shortcuts tab
        shortcuts = QTextEdit()
        shortcuts.setReadOnly(True)
        shortcuts.setHtml("""
        <h2>Keyboard Shortcuts</h2>
        
        <h3>File Operations</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr><td><b>Ctrl+O</b></td><td>Open Image</td></tr>
        <tr><td><b>Ctrl+S</b></td><td>Save Image</td></tr>
        <tr><td><b>Ctrl+E</b></td><td>Export PK3</td></tr>
        </table>
        
        <h3>Canvas Controls</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr><td><b>Ctrl + Plus</b></td><td>Zoom In</td></tr>
        <tr><td><b>Ctrl + Minus</b></td><td>Zoom Out</td></tr>
        <tr><td><b>Ctrl+0</b></td><td>Reset Zoom</td></tr>
        <tr><td><b>G</b></td><td>Toggle Pixel Grid</td></tr>
        </table>
        
        <h3>Processing</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr><td><b>Ctrl+Enter</b></td><td>Apply Selected Plugin</td></tr>
        </table>
        """)
        tabs.addTab(shortcuts, "Shortcuts")
        
        # Export guide tab
        export_guide = QTextEdit()
        export_guide.setReadOnly(True)
        export_guide.setHtml("""
        <h2>Export Guide</h2>
        
        <h3>PK3 Package</h3>
        <p>Creates a complete mod package with:</p>
        <ul>
        <li>Sprite PNG files with proper naming (NAMEA1.png, etc.)</li>
        <li>TEXTURES lump with sprite definitions</li>
        <li>Proper directory structure for GZDoom/Zandronum</li>
        </ul>
        
        <h3>WAD Layout</h3>
        <p>Creates a directory structure for WAD building:</p>
        <ul>
        <li>Individual sprite frame PNGs</li>
        <li>JSON metadata file with frame information</li>
        <li>Ready for importing into WAD editors</li>
        </ul>
        
        <h3>Testing Your Sprites</h3>
        <ol>
        <li>Set paths to GZDoom/Zandronum in Tools menu</li>
        <li>Export PK3 package</li>
        <li>Use "Test in GZDoom" button to launch with your mod</li>
        <li>Use console commands like 'give YOURSPRITE' to test</li>
        </ol>
        """)
        tabs.addTab(export_guide, "Export")
        
        layout.addWidget(tabs)
        
        # OK button
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

# Main Application Window
class SpriteForgeMainWindow(QMainWindow):
    """Enhanced main application window with full feature set."""
    
def __init__(self):
        super().__init__()
        self.current_image = None
        self.processed_image = None
        self.original_image = None  # Keep original for reset
        self.plugin_manager = PluginManager()
        self.processing_thread = None
        self.sheet_manager = SpriteSheetManager()
        self.animation_frames = []
        self.recent_files = []
        
        # External tool paths
        self.external_tools = {
            'gzdoom': '',
            'zandronum': '',
            'udb': '',
            'blender': ''
        }
        
        self.init_ui()
        self.load_settings()
        self.check_first_run()
        
        logger.info(f"{APP_NAME} v{__version__} started")
    
def check_first_run(self):
        """Check if this is first run and create example files."""
        readme_path = Path("README_SpriteForge.txt")
        if not readme_path.exists():
            self.create_readme()
    
def create_readme(self):
    """Create README file on first run."""
    readme_content = f"""
{APP_NAME} v{__version__}
{'=' * (len(APP_NAME) + 10)}

This is the first run of {APP_NAME}. 
A README file has been created with usage information.
"""
    with open("README.txt", "w", encoding="utf-8") as f:
        f.write(readme_content)

