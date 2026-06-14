from __future__ import annotations

import re


def normalize_symbol(symbol: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9.^-]", "", symbol.upper().strip())
    return cleaned.replace("-", ".")
