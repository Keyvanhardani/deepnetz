"""
DeepNetz build script — compiles Python to C extensions via Cython.

Usage:
    python setup.py build_ext --inplace   # compile locally
    python -m build                        # build wheel for PyPI
"""

import os
import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension

# Try Cython, fall back to pure Python
USE_CYTHON = True
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
    cythonize = None
    build_ext = None
    print("WARNING: Cython not found. Building pure Python package.")

# Files to compile with Cython (the actual logic)
CYTHON_MODULES = [
    "deepnetz/cli.py",
    "deepnetz/server.py",
    "deepnetz/engine/model.py",
    "deepnetz/engine/backend.py",
    "deepnetz/engine/hardware.py",
    "deepnetz/engine/planner.py",
    "deepnetz/engine/gguf_reader.py",
    "deepnetz/engine/downloader.py",
]

# __init__.py files stay as Python (minimal, just imports)
# They get distributed as .py but contain no logic


def get_extensions():
    if not USE_CYTHON:
        return []

    extensions = []
    for path in CYTHON_MODULES:
        module_name = path.replace("/", ".").replace(".py", "")
        extensions.append(
            Extension(
                module_name,
                [path],
                extra_compile_args=["-O2"],
            )
        )
    return cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
        # Don't include .py source in distribution
        nthreads=1,  # avoid macOS multiprocessing issues in CI
    )


# Clean .py files from build to prevent source leaking
class CleanSourceBuildExt(build_ext if build_ext else object):
    def run(self):
        super().run()
        # After compilation, remove .c files (Cython intermediate)
        for path in CYTHON_MODULES:
            c_file = path.replace(".py", ".c")
            if os.path.exists(c_file):
                os.remove(c_file)
                print(f"  Removed intermediate: {c_file}")


if __name__ == "__main__":
  setup(
    name="deepnetz",
    version="0.2.0",
    description="Run massive models on minimal hardware",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Keyvan Hardani",
    author_email="hardani@hotmail.de",
    url="https://deepnetz.com",
    project_urls={
        "Homepage": "https://deepnetz.com",
        "Repository": "https://github.com/Keyvanhardani/deepnetz",
    },
    license="Proprietary",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": CleanSourceBuildExt} if USE_CYTHON else {},
    exclude_package_data={"": ["*.py"], "deepnetz": ["cli.py", "server.py"],
                          "deepnetz.engine": ["model.py", "backend.py", "hardware.py",
                                               "planner.py", "gguf_reader.py", "downloader.py",
                                               "resolver.py"]},
    python_requires=">=3.9",
    install_requires=[
        "llama-cpp-python>=0.3.0",
        "huggingface_hub",
    ],
    extras_require={
        "server": ["fastapi", "uvicorn"],
    },
    entry_points={
        "console_scripts": [
            "deepnetz=deepnetz.cli:main",
        ],
    },
    # Don't include .py source files for compiled modules
    package_data={
        "deepnetz": ["*.so", "*.pyd"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
  )
