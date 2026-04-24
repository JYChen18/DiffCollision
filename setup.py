from setuptools import setup, Extension
import os
import platform
import pybind11
import coal

def get_include_paths():
    """Locate include and library directories for coal and dependencies."""
    include_dirs = [pybind11.get_include()]
    lib_dirs = []

    # Try to find via coal python package (cmeel style)
    coal_path = os.path.dirname(coal.__file__)
    # cmeel structure: .../cmeel.prefix/lib/python3.x/site-packages/coal
    if "cmeel.prefix" in coal_path:
        parts = coal_path.split(os.sep)
        if "cmeel.prefix" in parts:
            idx = parts.index("cmeel.prefix")
            prefix = os.sep.join(parts[:idx+1])
            
            include_base = os.path.join(prefix, "include")
            lib_base = os.path.join(prefix, "lib")
            
            if os.path.exists(include_base):
                include_dirs.append(include_base)
                # Add common subdirectories if they exist
                for sub in ["coal", "eigen3", "eigenpy/eigen"]:
                    sub_path = os.path.join(include_base, sub)
                    if os.path.exists(sub_path):
                        include_dirs.append(sub_path)
            
            if os.path.exists(lib_base):
                lib_dirs.append(lib_base)

    return list(set(include_dirs)), list(set(lib_dirs))

include_dirs, library_dirs = get_include_paths()

# Essential libraries for coal and its dependencies
libraries = ["coal", "boost_filesystem", "qhull_r", "octomap", "octomath", "assimp"]
extra_compile_args = []
extra_link_args = []

system = platform.system()

if system == "Darwin":  # macOS
    extra_compile_args += ["-std=c++17", "-Xpreprocessor", "-fopenmp"]
    extra_link_args += ["-lomp"]
elif system == "Linux":
    extra_compile_args += ["-std=c++17", "-fopenmp"]
    extra_link_args += ["-fopenmp"]
    libraries += ["pthread", "m", "c", "stdc++"]
elif system == "Windows":
    extra_compile_args += ["/std:c++17", "/openmp"]
else:
    print(f"Warning: Unsupported OS ({system}). Build may fail.")

# --- Extension Definition ---
extension_mod = Extension(
    "diffcollision.cpp._coal_openmp",
    sources=["src/diffcollision/cpp/coal_openmp.cpp"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    runtime_library_dirs=library_dirs, # Important for finding shared libs at runtime
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

# --- Setup ---
setup(ext_modules=[extension_mod])
