"""Deepfake module package initialization."""

import os
import platform

# Work around duplicated OpenMP runtime on some Windows Conda setups.
if platform.system() == "Windows":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
