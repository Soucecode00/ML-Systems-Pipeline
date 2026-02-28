import json
from pathlib import Path
from typing import Dict, Any


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_name: Name of config file (without .json extension)
        
    Returns:
        Dict containing configuration
    """
    config_path = Path(__file__).parent.parent.parent / "config" / f"{config_name}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def save_config(config: Dict[str, Any], config_name: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_name: Name of config file (without .json extension)
    """
    config_path = Path(__file__).parent.parent.parent / "config" / f"{config_name}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
