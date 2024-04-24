import sys
import os

# Inside test_utils.py
from src.utils import get_config

def test_get_config(path="configs/config.toml"):
    config = get_config(path)
    assert config["data"]["cora_folder_path"] == "cora"
    
 

 