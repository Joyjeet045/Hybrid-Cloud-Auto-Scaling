import os
import yaml

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config(path=None):
    if path is None:
        path = os.path.join(CONFIG_DIR, "default.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)
