from setuptools import setup, Extension
import os
import platform
import pybind11

# --- Check Conda Environment ---
conda_prefix = os.environ.get("CONDA_PREFIX")
if not conda_prefix:
    raise RuntimeError(
        "Please activate a Conda environment containing 'coal' before installing."
    )

# --- Locate coal library ---
coal_include = os.path.join(conda_prefix, "include", "coal")
lib_dir = os.path.join(conda_prefix, "lib")

if not os.path.exists(coal_include):
    raise RuntimeError(
        f"Error: 'coal' not found in Conda environment ({conda_prefix}).\n"
        f"Please run:\n  conda install coal -c conda-forge"
    )

include_dirs = [
    pybind11.get_include(),
    os.path.join(conda_prefix, "include"),
    os.path.join(conda_prefix, "include", "eigen3"),
    coal_include,
]

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
    library_dirs=[lib_dir],
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

# --- Setup ---
setup(ext_modules=[extension_mod])
