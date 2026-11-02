"""Make `pytest` test THIS working tree, not an installed `mpmf` elsewhere.

Without this, `pytest` picked up whatever `mpmf` was on sys.path (e.g. an
editable install of another checkout) and the suite silently tested the wrong
code. Round 77 hit exactly that. Subprocess-based tests inherit PYTHONPATH.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_pp = os.environ.get("PYTHONPATH", "")
if _SRC not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = _SRC + (os.pathsep + _pp if _pp else "")
