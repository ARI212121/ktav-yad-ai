import os
import sys


_DLL_DIRECTORY_HANDLES = []

if sys.platform == "win32" and getattr(sys, "frozen", False):
    _bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    _dll_dirs = (
        _bundle_dir,
        os.path.join(_bundle_dir, "torch", "lib"),
        os.path.join(_bundle_dir, "numpy.libs"),
    )
    for _dll_dir in _dll_dirs:
        if os.path.isdir(_dll_dir):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(_dll_dir))
    os.environ["PATH"] = os.pathsep.join(
        (*_dll_dirs, os.environ.get("PATH", "")))
