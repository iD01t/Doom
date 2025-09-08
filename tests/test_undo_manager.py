import pytest
from pathlib import Path

# Add the root directory to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sprite_forge_pro import CoreEngine, Project, Sprite, PropertyChangeCommand, AddSpriteCommand, RemoveSpriteCommand

@pytest.fixture
def core_engine():
    """Returns a clean CoreEngine instance."""
    core = CoreEngine()
    core.plugin_manager.plugins = {} # Disable plugins
    return core

def test_property_change_undo_redo(core_engine):
    """Tests the undo and redo of a simple property change."""
    project = core_engine.project
    undo_manager = core_engine.undo_manager

    original_name = project.settings.name
    new_name = "Modified Project Name"

    # Create and execute the command
    cmd = PropertyChangeCommand(project.settings, 'name', new_name)
    undo_manager.execute(cmd)

    assert project.settings.name == new_name

    # Undo the command
    undo_manager.undo()
    assert project.settings.name == original_name

    # Redo the command
    undo_manager.redo()
    assert project.settings.name == new_name

def test_add_sprite_undo_redo(core_engine):
    """Tests the undo and redo of adding a sprite."""
    project = core_engine.project
    undo_manager = core_engine.undo_manager

    assert len(project.sprites) == 0

    sprite = Sprite(name="new_sprite")
    cmd = AddSpriteCommand(project, sprite)

    # Execute
    undo_manager.execute(cmd)
    assert len(project.sprites) == 1
    assert "new_sprite" in project.sprites

    # Undo
    undo_manager.undo()
    assert len(project.sprites) == 0

    # Redo
    undo_manager.redo()
    assert len(project.sprites) == 1
    assert "new_sprite" in project.sprites

def test_remove_sprite_undo_redo(core_engine):
    """Tests the undo and redo of removing a sprite."""
    project = core_engine.project
    undo_manager = core_engine.undo_manager

    # First, add a sprite directly to set up the state
    sprite = Sprite(name="sprite_to_delete")
    project.add_sprite(sprite)
    assert len(project.sprites) == 1

    # Create and execute the remove command
    cmd = RemoveSpriteCommand(project, "sprite_to_delete")
    undo_manager.execute(cmd)
    assert len(project.sprites) == 0

    # Undo
    undo_manager.undo()
    assert len(project.sprites) == 1
    assert "sprite_to_delete" in project.sprites

    # Redo
    undo_manager.redo()
    assert len(project.sprites) == 0
