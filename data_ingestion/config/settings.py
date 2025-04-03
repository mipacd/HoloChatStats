import configparser
import os
import sys

def get_config(key1, key2):
    """
    Reads a value from config.ini.

    Args:
        key1 (str): Section name in config.ini.
        key2 (str): Option name in config.ini.

    Returns:
        str: Value associated with key1 and key2 in config.ini.

    Raises:
        FileNotFoundError: If config.ini is not found.
        KeyError: If key1 or key2 are not found in config.ini.
    """
    config = configparser.ConfigParser()
    caller_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    ini_file = os.path.join(caller_script_dir, 'config.ini')
    if not config.read(ini_file):
        raise FileNotFoundError("config.ini not found.")
    try:
        return config[key1][key2]
    except KeyError as e:
        raise KeyError(f"Key not found in config.ini: {e}")