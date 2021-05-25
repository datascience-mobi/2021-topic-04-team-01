import pathlib
import os

def list_dirs(dir):
    """Returns all directories in a given directory
    """
    return [f for f in pathlib.Path(dir).iterdir() if f.is_dir()]

print(list_dirs('CroppedYale'))

def list_files(directory):
    """Returns all files in a given directory
    """
    return [
        f
        for f in pathlib.Path(directory).iterdir()
        if f.is_file() and not f.name.startswith(".")
    ]

print(list_files('CroppedYale/yaleB01'))




