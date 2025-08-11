import sys
import os

def setup_src_path():
    """Add the src directory to Python path for imports"""
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

# Automatically setup path when module is imported
setup_src_path()
