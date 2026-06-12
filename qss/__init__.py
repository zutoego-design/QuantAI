from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_SRC_QSS = Path(__file__).resolve().parent.parent / "src" / "qss"
if _SRC_QSS.exists():
    __path__.append(str(_SRC_QSS))
