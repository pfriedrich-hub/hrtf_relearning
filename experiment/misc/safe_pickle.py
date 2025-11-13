import io
import pickle
import copyreg
import pathlib
from typing import Any, BinaryIO

# ---- 1) Make future pickles portable ---------------------------------------
def _reduce_path(p: pathlib.Path):
    # Store as a simple string; reconstruct with pathlib.Path on load
    # Use .as_posix() to be resilient to backslashes in legacy data
    return (pathlib.Path, (p.as_posix(),))

# Register for all relevant pathlib classes (exist on all interpreters)
for cls in (
    pathlib.Path,
    pathlib.PosixPath,
    pathlib.WindowsPath,
    pathlib.PurePath,
    pathlib.PurePosixPath,
    pathlib.PureWindowsPath,
):
    try:
        copyreg.pickle(cls, _reduce_path)
    except Exception:
        # Some classes may not be instantiable on this OS, but type objects still exist.
        pass

# ---- 2) Read legacy pickles created on a different OS ----------------------
class _CrossPlatformUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Remap OS-specific concrete classes to plain Path on load
        if module == "pathlib" and name in {"WindowsPath", "PosixPath"}:
            return pathlib.Path
        return super().find_class(module, name)

def loads(data: bytes) -> Any:
    return _CrossPlatformUnpickler(io.BytesIO(data)).load()

def load(file: BinaryIO) -> Any:
    return _CrossPlatformUnpickler(file).load()

def dumps(obj: Any, protocol: int | None = None) -> bytes:
    return pickle.dumps(obj, protocol=protocol or pickle.HIGHEST_PROTOCOL)

def dump(obj: Any, file: BinaryIO, protocol: int | None = None) -> None:
    pickle.dump(obj, file, protocol=protocol or pickle.HIGHEST_PROTOCOL)