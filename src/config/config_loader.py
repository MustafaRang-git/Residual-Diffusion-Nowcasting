import os
import yaml
from pathlib import Path

def load_config(config_path: str = None) -> dict:
    """
    Loads the YAML configuration file and returns it as a dictionary.
    
    Args:
        config_path (str, optional): Path to the yaml config file. 
            Defaults to 'default.yaml' in the same directory as this script.
            
    Returns:
        dict: Parsed configuration dictionary.
    """
    if config_path is None:
        # Resolve path relative to this script
        current_dir = Path(__file__).parent
        config_path = current_dir / "default.yaml"
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

# Example usage/test
if __name__ == "__main__":
    cfg = load_config()
    print("=== Configuration Loaded Successfully ===")
    print(f"Project Name: {cfg['project']['name']}")
    print(f"Data Directory: {cfg['paths']['data_dir']}")
    print(f"Model Image Size: {cfg['data']['img_size']}x{cfg['data']['img_size']}")
