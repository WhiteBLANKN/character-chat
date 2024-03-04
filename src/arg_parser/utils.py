import importlib.util
import importlib.metadata
from typing import TYPE_CHECKING, Union, Tuple
import shutil

def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists
    
def special_print(message: str) -> None:
    terminal_length, terminal_height = shutil.get_terminal_size()
    print('*' * terminal_length)
    print(message.center(terminal_length-len(message)))
    print('*' * terminal_length)