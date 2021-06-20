import errno
import logging
import os
from typing import Union

from py._path.local import LocalPath

log = logging.getLogger(__name__)


def is_dir(dir_path: Union[str, LocalPath], exception: bool = False) -> bool:
    dir_exists = os.path.isdir(dir_path)

    if exception and not dir_exists:
        raise NotADirectoryError(dir_path)

    return dir_exists

def create_dir(dir_path: Union[str, LocalPath], exception: bool = True) -> None:
    """Creates a directory and its super paths.
    Succeeds even if the path already exists."""
    if not is_dir(dir_path):
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except OSError as e:
            if exception and e.errno != errno.EEXIST:
                raise

    # check that the dir_path now indeed exists as a directory
    is_dir(dir_path, exception=exception)


def is_file(file_path: Union[str, LocalPath], exception: bool = False) -> bool:
    file_exists = os.path.isfile(file_path)

    if exception and not file_exists:
        raise FileNotFoundError(file_path)

    return file_exists