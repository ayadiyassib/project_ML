"""
Tiny in-process cache for parsed DataFrames.

Re-reading a 20 000-row CSV on every request (index, preview, preprocessing)
would add ~100 ms per page.  Files are immutable once imported, so a cache keyed
by ``(path, mtime, size)`` is safe; mutating callers always receive a copy.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import pandas as pd

_MAX_ENTRIES = 4
_cache: "OrderedDict[tuple[str, float, int], pd.DataFrame]" = OrderedDict()
_lock = threading.Lock()


def _key(path: Path) -> tuple[str, float, int]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_mtime, stat.st_size)


def get_dataframe(path: str | Path, loader: Callable[[Path], pd.DataFrame]) -> pd.DataFrame:
    """Return a copy of the parsed file, using the cache when possible."""
    resolved = Path(path)
    key = _key(resolved)

    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached.copy()

    frame = loader(resolved)

    with _lock:
        _cache[key] = frame
        _cache.move_to_end(key)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)

    return frame.copy()


def invalidate(path: str | Path | None = None) -> None:
    """Drop one file from the cache, or the whole cache when ``path`` is None."""
    with _lock:
        if path is None:
            _cache.clear()
            return
        target = str(Path(path).resolve())
        for key in [k for k in _cache if k[0] == target]:
            _cache.pop(key, None)
