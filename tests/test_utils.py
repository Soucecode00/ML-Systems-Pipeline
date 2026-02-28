"""
Unit tests for utility functions
"""

import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import setup_logger, get_project_root, create_directory_if_not_exists
from src.utils.config_loader import load_config


def test_setup_logger():
    """Test logger setup."""
    logger = setup_logger("test_logger")
    assert logger is not None
    assert logger.name == "test_logger"


def test_get_project_root():
    """Test getting project root."""
    root = get_project_root()
    assert root.exists()
    assert root.is_dir()


def test_load_config():
    """Test loading configuration."""
    config = load_config("model_config")
    assert config is not None
    assert 'model' in config
    assert 'features' in config


def test_load_config_not_found():
    """Test loading non-existent configuration."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config")


def test_create_directory():
    """Test directory creation."""
    test_dir = Path("tests/temp_test_dir")
    create_directory_if_not_exists(str(test_dir))
    assert test_dir.exists()
    test_dir.rmdir()
