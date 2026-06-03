#!/usr/bin/env python3
"""Legacy compatibility shim for match_lm.

The maintained implementation now lives in find_pairs.match_pipeline.
"""

from __future__ import annotations

import os
import sys

if __package__ is None or __package__ == "":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from find_pairs.match_pipeline import main


if __name__ == "__main__":
    main()
