"""Import helpers for the archived v2 scheduling framework.

The new v1 implementation treats ``rl_sensor_scheduling_framework`` as an
archive. This shim only exposes its Python modules; it does not mutate archived
source files or result artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = ROOT / "rl_sensor_scheduling_framework"
ARCHIVE_SRC = ARCHIVE_ROOT / "src"


def ensure_archive_src() -> Path:
    if not ARCHIVE_SRC.exists():
        raise FileNotFoundError(f"Archived framework src not found: {ARCHIVE_SRC}")
    src = str(ARCHIVE_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    return ARCHIVE_SRC


ensure_archive_src()
