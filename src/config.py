import yaml
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    """Loads the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config

# Global config instance for convenient importing
# Note: For tests or custom runs, it's better to pass config explicitly,
# but providing a global instance helps with simple scripts.
try:
    CONFIG = load_config()
except FileNotFoundError:
    CONFIG = {}
