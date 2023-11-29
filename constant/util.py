# Copyright 2023 O1 Software Network. MIT licensed.
import sys
from pathlib import Path


def repo_top() -> Path:
    """Returns the top directory of the handson-ml3 repository in a portable way."""
    # Result is same as `git rev-parse --show-toplevel`, without forking a child.
    return Path(__file__).parent.parent.resolve()


def constant() -> Path:
    """Returns base directory for our project files."""
    return repo_top() / "constant"


def temp_dir() -> Path:
    """Returns a writable directory for temporary files."""
    temporary = {
        "cygwin": "/tmp",
        "darwin": "/tmp",
        "linux": "/tmp",
        "win32": r"C:\temp",
    }
    folder = Path(temporary[sys.platform])
    assert folder.is_dir()
    return folder
