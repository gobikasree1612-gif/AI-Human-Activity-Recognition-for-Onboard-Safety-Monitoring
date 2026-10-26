"""
split_dataset.py (root shim)
============================
Redirects to dataset/split_dataset.py.
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "dataset"))

from dataset.split_dataset import main

if __name__ == "__main__":
    main()
