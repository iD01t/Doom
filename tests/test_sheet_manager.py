import pytest
from pathlib import Path
from PIL import Image

# Add the root directory to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sprite_forge_pro import CoreEngine, Sprite, SpriteFrame

@pytest.fixture
def core_with_sprites():
    """Returns a CoreEngine instance with a project containing a couple of sprites."""
    core = CoreEngine()
    # Disable plugin discovery for testing
    core.plugin_manager.plugins = {}

    project = core.project
    # Use a sheet width that forces a new row
    project.settings.sprite_sheet_size = (100, 128)
    project.settings.sprite_padding = 0 # Disable padding for predictable coordinates
    project.settings.apply_bleed = False

    # Create frames
    frame1 = SpriteFrame(name="FrameA1", image=Image.new('RGBA', (30, 40), (0,0,0,255)))
    frame2 = SpriteFrame(name="FrameA2", image=Image.new('RGBA', (30, 40), (255,0,0,255)))
    frame3 = SpriteFrame(name="FrameB1", image=Image.new('RGBA', (50, 50), (0,255,0,255))) # This one will wrap to the next line

    sprite1 = Sprite(name="SpriteA")
    sprite1.add_frame(frame1)
    sprite1.add_frame(frame2)

    sprite2 = Sprite(name="SpriteB")
    sprite2.add_frame(frame3)

    project.add_sprite(sprite1)
    project.add_sprite(sprite2)

    return core

def test_populate_from_project(core_with_sprites):
    """Test that the manager's list is populated from the project's sprites."""
    manager = core_with_sprites.sheet_manager
    manager.populate_from_project()

    # Expecting 3 frames to be added to the internal list
    assert len(manager._frames_to_pack) == 3
    assert manager._frames_to_pack[0][0].name == "FrameA1"

def test_simple_packing_logic(core_with_sprites):
    """Test the custom packing logic with predictable coordinates."""
    manager = core_with_sprites.sheet_manager
    manager.pack()

    packed_rects = manager._packed_rects
    assert len(packed_rects) == 3

    # Frame 1: at origin
    assert packed_rects[0].x == 0
    assert packed_rects[0].y == 0

    # Frame 2: next to frame 1
    assert packed_rects[1].x == 30
    assert packed_rects[1].y == 0

    # Frame 3: wraps to a new row
    # The new row starts at y = 40 (the height of the first row)
    assert packed_rects[2].x == 0
    assert packed_rects[2].y == 40

def test_generate_atlas_with_simple_packer(core_with_sprites):
    """Test generating an atlas with the new simple packer."""
    manager = core_with_sprites.sheet_manager
    atlases = manager.generate_atlases()

    assert len(atlases) == 1
    atlas = atlases[0]

    # Expected size:
    # Width = max(30+30, 50) = 60
    # Height = 40 (row 1) + 50 (row 2) = 90
    assert atlas.size == (60, 90)

    # Check a pixel from the third sprite (green) to ensure it was pasted correctly
    pixel = atlas.getpixel((10, 50)) # A pixel inside the third sprite's area
    assert pixel == (0,255,0,255)
