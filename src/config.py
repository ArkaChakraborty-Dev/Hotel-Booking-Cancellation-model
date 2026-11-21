"""Configuration loader and manager."""
import os
from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Loads configuration from YAML file.
    
    Args:
        config_path: Optional path to the configuration file.
                     Defaults to config/config.yaml at the project root.
                     
    Returns:
        Dictionary containing configuration parameters.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config
