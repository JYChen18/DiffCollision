import os
import ctypes
import platform
import coal

def load_dependencies():
    """Manually load shared libraries from cmeel.prefix/lib if needed."""
    if platform.system() == "Windows":
        return # Not implemented for Windows yet

    coal_path = os.path.dirname(coal.__file__)
    if "cmeel.prefix" in coal_path:
        parts = coal_path.split(os.sep)
        if "cmeel.prefix" in parts:
            idx = parts.index("cmeel.prefix")
            prefix = os.sep.join(parts[:idx+1])
            lib_dir = os.path.join(prefix, "lib")
            
            # Load dependencies in order
            ext = ".dylib" if platform.system() == "Darwin" else ".so"
            libs = ["libcoal", "libeigenpy", "libomp"]
            
            for lib in libs:
                lib_path = os.path.join(lib_dir, lib + ext)
                if os.path.exists(lib_path):
                    try:
                        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    except Exception as e:
                        print(f"Warning: Failed to load {lib_path}: {e}")

# Run loader before any other imports in this package
load_dependencies()
