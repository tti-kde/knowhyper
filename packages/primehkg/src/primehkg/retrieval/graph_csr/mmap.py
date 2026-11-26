from __future__ import annotations

import mmap
import struct
from pathlib import Path

_SLICE_STEP_MSG = "only step=1 slices are supported"


class MmapU32Array:
    """Read-only `uint32` array backed by an mmap."""

    def __init__(self, mm: mmap.mmap) -> None:
        """Create from a read-only mmap handle."""
        self._mm = mm

    def __len__(self) -> int:
        """Return the number of items."""
        return self._mm.size() // 4

    def __getitem__(self, key: int | slice) -> int | list[int]:
        """Return one item or a slice (step must be 1)."""
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                raise IndexError(_SLICE_STEP_MSG)
            count = stop - start
            if count <= 0:
                return []
            off = start * 4
            return list(struct.unpack_from(f"<{count}I", self._mm, off))
        i = key
        n = len(self)
        if i < 0:
            i = n + i
        if i < 0 or i >= n:
            raise IndexError(i)
        off = i * 4
        return struct.unpack_from("<I", self._mm, off)[0]

    def close(self) -> None:
        """Close the underlying mmap handle."""
        self._mm.close()


class MmapU64Array:
    """Read-only `uint64` array backed by an mmap."""

    def __init__(self, mm: mmap.mmap) -> None:
        """Create from a read-only mmap handle."""
        self._mm = mm

    def __len__(self) -> int:
        """Return the number of items."""
        return self._mm.size() // 8

    def __getitem__(self, key: int | slice) -> int | list[int]:
        """Return one item or a slice (step must be 1)."""
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                raise IndexError(_SLICE_STEP_MSG)
            count = stop - start
            if count <= 0:
                return []
            off = start * 8
            return list(struct.unpack_from(f"<{count}Q", self._mm, off))
        i = key
        n = len(self)
        if i < 0:
            i = n + i
        if i < 0 or i >= n:
            raise IndexError(i)
        off = i * 8
        return struct.unpack_from("<Q", self._mm, off)[0]

    def close(self) -> None:
        """Close the underlying mmap handle."""
        self._mm.close()


def _mmap_ro(path: Path) -> mmap.mmap:
    f = path.open("rb")
    try:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    finally:
        f.close()


def mmap_u32_array(path: str | Path) -> MmapU32Array:
    return MmapU32Array(_mmap_ro(Path(path)))


def mmap_u64_array(path: str | Path) -> MmapU64Array:
    return MmapU64Array(_mmap_ro(Path(path)))
