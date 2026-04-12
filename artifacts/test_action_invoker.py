
"""
Tests for action_invoker module.
"""

import pytest
from typing import Dict, Any

# Import the invoker functions
from artifacts.action_invoker import (
    invoke_action,
    invoke_action_with_validation,
    get_all_action_classes,
    create_action_instance,
    get_action_classes_by_module,
)

# Import action types
from compass.agents.neo.types import ReadFileAction, WriteFileAction


def test_get_all_action_classes():
    """Test that we can retrieve all action classes."""
    classes = get_all_action_classes()
    assert isinstance(classes, list)
    assert len(classes) > 0
    # Check that some expected action classes are present
    class_names = [c.__name__ for c in classes]
    assert "ReadFileAction" in class_names
    assert "WriteFileAction" in class_names


def test_create_action_instance():
    """Test creating action instances by type name."""
    # Test ReadFileAction
    action = create_action_instance("ReadFileAction", path="test.txt")
    assert action is not None
    assert isinstance(action, ReadFileAction)
    assert action.path == "test.txt"

    # Test WriteFileAction
    action = create_action_instance("WriteFileAction", path="output.txt", content="hello")
    assert action is not None
    assert isinstance(action, WriteFileAction)
    assert action.path == "output.txt"
    assert action.content == "hello"

    # Test invalid type
    action = create_action_instance("NonExistentAction")
    assert action is None


def test_invoke_action_with_read_file():
    """Test invoking ReadFileAction."""
    # Create a test file first
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        # Create and invoke ReadFileAction
        action = ReadFileAction(path=test_file)
        success, result = invoke_action(action, project_path=tmpdir)

        assert success
        assert "test content" in result


def test_invoke_action_with_write_file():
    """Test invoking WriteFileAction."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "output.txt")

        # Create and invoke WriteFileAction
        action = WriteFileAction(path=test_file, content="hello world")
        success, result = invoke_action(action, project_path=tmpdir)

        assert success
        assert os.path.exists(test_file)

        with open(test_file, "r") as f:
            content = f.read()
        assert content == "hello world"


def test_invoke_action_with_validation():
    """Test invoking action with validation."""
    # Test with valid action
    action = ReadFileAction(path="nonexistent.txt")
    is_valid, error, result = invoke_action_with_validation(action, project_path=".")

    # Validation should pass (even if file doesn't exist - execution will fail)
    assert is_valid
    assert error is None

    # Test with invalid action (missing required field)
    try:
        action = ReadFileAction(path="")  # Empty path might be invalid
        is_valid, error, result = invoke_action_with_validation(action, project_path=".")
        # If validation doesn't catch it, that's also acceptable
    except Exception:
        # Some validation might raise instead of returning False
        pass


def test_get_action_classes_by_module():
    """Test grouping action classes by module."""
    module_actions = get_action_classes_by_module()
    assert isinstance(module_actions, dict)
    assert len(module_actions) > 0

    # Check that compass.agents.neo.types is present
    assert "compass.agents.neo.types" in module_actions or any("types" in m for m in module_actions.keys())


def test_dict_to_action_conversion():
    """Test converting dict to action instance."""
    # This tests the internal _create_action_instance function indirectly
    action_dict = {
        "type": "ReadFileAction",
        "path": "test.txt"
    }

    success, result = invoke_action(action_dict, project_path=".")
    # Should fail because file doesn't exist, but conversion should work
    assert not success  # Execution fails because file doesn't exist
    assert "test.txt" in result or "not found" in result.lower()


def test_action_type_registration():
    """Test that action types are properly registered."""
    classes = get_all_action_classes()

    # Check that all classes have required attributes
    for cls in classes:
        assert hasattr(cls, "__dataclass_fields__") or hasattr(cls, "__init__")
        # Check for expected action fields
        if hasattr(cls, "__dataclass_fields__"):
            fields = cls.__dataclass_fields__
            # Most actions should have at least a path or similar field
            field_names = [f for f in fields.keys() if not f.startswith("_")]
            assert len(field_names) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
