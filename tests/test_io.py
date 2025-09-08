import pytest
import tempfile
from pathlib import Path
from PIL import Image, ImageChops

# Add the root directory to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sprite_forge_pro import CoreEngine, Sprite, SpriteFrame, SpriteAnimation

@pytest.fixture
def core_engine():
    """Returns a clean CoreEngine instance."""
    # Disable plugin discovery for testing to avoid side effects
    core = CoreEngine()
    core.plugin_manager.plugins = {}
    return core

@pytest.fixture
def sample_project(core_engine):
    """Creates a sample project with various components."""
    project = core_engine.project
    project.settings.author = "Test Author"

    # Create sprite 1
    sprite1 = Sprite(name="Player")
    frame1_img = Image.new('RGBA', (10, 10), (255, 0, 0, 255))
    frame1 = SpriteFrame(name="PLAYER_A1", image=frame1_img)
    sprite1.add_frame(frame1)

    # Create sprite 2
    sprite2 = Sprite(name="Monster")
    frame2_img = Image.new('RGBA', (20, 20), (0, 255, 0, 255))
    frame2 = SpriteFrame(name="MONSTER_A1", image=frame2_img)
    sprite2.add_frame(frame2)

    project.add_sprite(sprite1)
    project.add_sprite(sprite2)

    # Create animation
    anim = SpriteAnimation(name="player_walk", frames=[frame1])
    project.add_animation(anim)

    return project

def test_project_save_and_load(core_engine, sample_project):
    """
    Tests the full cycle of saving a project to a file and loading it back.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test_project.sfp"

        # Save the project
        core_engine.save_project(str(file_path))
        assert file_path.exists()

        # Create a new core engine and load the project
        new_core = CoreEngine()
        new_core.load_project(str(file_path))

        loaded_project = new_core.project

        # --- Assertions ---
        # Compare settings
        assert loaded_project.settings.author == "Test Author"

        # Compare sprites and frames
        assert len(loaded_project.sprites) == 2
        assert "Player" in loaded_project.sprites
        assert "Monster" in loaded_project.sprites

        loaded_player_sprite = loaded_project.sprites["Player"]
        assert len(loaded_player_sprite.frames) == 1
        assert "PLAYER_A1" in loaded_player_sprite.frames

        original_img = sample_project.sprites["Player"].frames["PLAYER_A1"].image
        loaded_img = loaded_player_sprite.frames["PLAYER_A1"].image

        assert original_img.size == loaded_img.size
        assert ImageChops.difference(original_img, loaded_img).getbbox() is None

        # Compare animations
        assert len(loaded_project.animations) == 1
        assert "player_walk" in loaded_project.animations

        loaded_anim = loaded_project.animations["player_walk"]
        assert len(loaded_anim.frames) == 1
        # Check that the frame in the animation is the correct one from the loaded sprites
        assert loaded_anim.frames[0].name == "PLAYER_A1"
        assert loaded_anim.frames[0] is loaded_project.sprites["Player"].frames["PLAYER_A1"]
