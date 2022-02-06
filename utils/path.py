import importlib
import os.path
import sys
from types import ModuleType


def get_module(path: str) -> ModuleType:
    path = os.path.normpath(path)
    sys.path.insert(0, os.path.dirname(path))
    return importlib.import_module(os.path.basename(path))
