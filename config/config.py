# config.py
import yaml
import os

def load_config(config_file="config.yaml"):
    """
    Load configuration settings from a YAML file.
    
    Args:
        config_file (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration settings as a dictionary.
    """
    # Check if the configuration file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    # Load the YAML file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    
    return config

