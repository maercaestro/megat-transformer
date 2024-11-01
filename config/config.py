import os
import yaml

def load_config():
    # Set the path to config.yaml relative to the config.py file
    config_file = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    
    return config

