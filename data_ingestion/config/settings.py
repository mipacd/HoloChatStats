import configparser
import os

def get_config(key1, key2):
    """
    Reads a value from config.ini, reliably finding the file from the project root.
    """
    config = configparser.ConfigParser()
    
    # 1. Get the directory of this file (config/settings.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. Go up one level to find the project's root directory
    project_root = os.path.dirname(current_dir)
    # 3. Construct the definitive path to config.ini
    ini_file = os.path.join(project_root, 'config.ini')
    
    if not config.read(ini_file):
        raise FileNotFoundError(f"config.ini not found. Looked for it at: {ini_file}")
        
    try:
        return config[key1][key2]
    except KeyError as e:
        raise KeyError(f"Key not found in config.ini: {e}")