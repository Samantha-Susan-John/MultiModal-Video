"""Configuration loading utilities."""
import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_all_configs(config_dir: str = './configs') -> Dict[str, Dict]:
    """
    Load all configuration files from directory.
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        Dictionary of all configs
    """
    config_dir = Path(config_dir)
    
    configs = {}
    
    # Load each config file
    for config_file in config_dir.glob('*.yaml'):
        config_name = config_file.stem
        configs[config_name] = load_config(config_file)
    
    return configs


def merge_configs(*configs: Dict) -> Dict:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration
    """
    merged = {}
    
    for config in configs:
        merged.update(config)
    
    return merged


def save_config(config: Dict, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save file
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


class ConfigManager:
    """Configuration manager."""
    
    def __init__(self, config_dir: str = './configs'):
        """
        Initialize config manager.
        
        Args:
            config_dir: Directory containing configs
        """
        self.config_dir = Path(config_dir)
        self.configs = load_all_configs(config_dir)
    
    def get(self, config_name: str) -> Dict:
        """Get specific configuration."""
        return self.configs.get(config_name, {})
    
    def get_all(self) -> Dict:
        """Get all configurations."""
        return self.configs
    
    def update(self, config_name: str, updates: Dict):
        """Update specific configuration."""
        if config_name in self.configs:
            self.configs[config_name].update(updates)
    
    def save_all(self):
        """Save all configurations."""
        for config_name, config in self.configs.items():
            save_path = self.config_dir / f"{config_name}.yaml"
            save_config(config, save_path)
