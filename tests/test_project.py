import pytest
from pathlib import Path
from PIL import Image

# Add the root directory to the Python path to allow importing sprite_forge_pro
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sprite_forge_pro import Project, ProjectSettings, Sprite, SpriteFrame, SpriteAnimation

@pytest.fixture
def dummy_image():
    """Creates a dummy 10x10 red image for testing."""
    return Image.new('RGBA', (10, 10), (255, 0, 0, 255))

@pytest.fixture
def basic_project():
    """Returns a Project with default settings."""
    return Project(settings=ProjectSettings())

def test_project_creation(basic_project):
    """Test that a project is created with default settings."""
    assert basic_project is not None
    assert basic_project.settings.name == "New Project"
    assert len(basic_project.sprites) == 0
    assert len(basic_project.animations) == 0

def test_add_sprite(basic_project):
    """Test adding a sprite to a project."""
    sprite = Sprite(name="test_sprite")
    basic_project.add_sprite(sprite)
    assert len(basic_project.sprites) == 1
    assert "test_sprite" in basic_project.sprites
    assert basic_project.sprites["test_sprite"] == sprite

def test_add_sprite_frame(dummy_image):
    """Test adding a frame to a sprite."""
    sprite = Sprite(name="test_sprite")
    frame = SpriteFrame(name="frame1", image=dummy_image)
    sprite.add_frame(frame)
    assert len(sprite.frames) == 1
    assert "frame1" in sprite.frames
    assert sprite.frames["frame1"].image.size == (10, 10)

def test_add_animation(basic_project, dummy_image):
    """Test adding an animation to a project."""
    frame1 = SpriteFrame(name="anim_frame1", image=dummy_image)
    frame2 = SpriteFrame(name="anim_frame2", image=dummy_image)
    anim = SpriteAnimation(name="test_anim", frames=[frame1, frame2])

    basic_project.add_animation(anim)
    assert len(basic_project.animations) == 1
    assert "test_anim" in basic_project.animations
    assert len(basic_project.animations["test_anim"].frames) == 2
    assert basic_project.animations["test_anim"].frames[0].name == "anim_frame1"
