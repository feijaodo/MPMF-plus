"""Float32 vector storage for MPMF (Round 77).

Vectors live in ONE contiguous float32 matrix per open store, addressed by a
``node_id -> row`` index.  Nodes hold no Python float lists in memory: a read
returns a row view straight into the matrix.

The on-disk format is untouched.  The vector block is still ``dim``
little-endian float32 values per node, each row padded to a 32-byte boundary.
A float32 read back and re-packed with ``'<f'`` is bit-exact, so
``save(load(f))`` produces an identical vector block.

numpy is the fast path.  Without numpy the same layout is kept in a flat
``array('f')`` (still 4 bytes per component, still float32) so memory stays
compact and vectors are never expanded into Python float lists.
"""

from __future__ import annotations

import math
from array import array as _array
from typing import Dict, Iterator, List, Optional, Tuple

try:  # pragma: no cover - exercised only when numpy is missing
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

HAVE_NUMPY = _np is not None


class VectorStore:
    """One float32 matrix + a node_id -> row index for a single open store."""

    __slots__ = ("dim", "_mat", "_ids", "_row", "_n", "_cap", "_norms",
                 "_norms_dirty", "_version", "_numpy")

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._ids: List[int] = []          # row -> node_id
        self._row: Dict[int, int] = {}     # node_id -> row
        self._n = 0                        # live rows
        self._cap = 0
        self._numpy = HAVE_NUMPY
        self._mat = None                   # (cap, dim) float32 | array('f')
        self._norms = None                 # (cap,) float32 | array('f')
        self._norms_dirty = True
        self._version = 0                  # bumps on every row add/update/drop
        self._reserve(256)

    # ------------------------------------------------------------------ setup
    def _reserve(self, cap: int) -> None:
        cap = max(int(cap), 1)
        if cap <= self._cap:
            return
        dim = max(self.dim, 1)
        n = self._n
        if self._numpy:
            mat = _np.zeros((cap, dim), dtype=_np.float32)
            norms = _np.zeros(cap, dtype=_np.float32)
            if n:
                mat[:n] = self._mat[:n]
                norms[:n] = self._norms[:n]
        else:  # pragma: no cover - live env has numpy
            mat = _array("f", bytes(4 * cap * dim))
            norms = _array("f", bytes(4 * cap))
            if n:
                mat[: n * dim] = self._mat[: n * dim]
                norms[:n] = self._norms[:n]
        self._mat, self._norms, self._cap = mat, norms, cap

    def reserve(self, cap: int) -> None:
        """Pre-size the matrix (used when the node count is known up front)."""
        self._reserve(int(cap))

    # --------------------------------------------------------------- coercion
    def _coerce(self, vec, validate: bool = True):
        dim = self.dim
        if dim <= 0:
            raise ValueError(
                "VectorStore: refusing to store a zero-length vector "
                "(vector_dimensions=%d)" % dim)
        if self._numpy:
            a = vec if isinstance(vec, _np.ndarray) else _np.asarray(vec)
            if a.dtype != _np.float32:
                a = a.astype(_np.float32, copy=False)
            if a.ndim != 1 or a.shape[0] != dim:
                raise ValueError(
                    "vector must be a flat array of %d floats, got shape %r"
                    % (dim, getattr(a, "shape", None)))
            if validate:
                if not bool(_np.all(_np.isfinite(a))):
                    raise ValueError("vector contains NaN or Inf")
                if float(_np.dot(a, a)) == 0.0:
                    raise ValueError(
                        "refusing to store an all-zero vector (node would "
                        "have no direction)")
            return a
        row = [float(x) for x in vec]  # pragma: no cover - no-numpy fallback
        if len(row) != dim:
            raise ValueError(
                "vector must have %d floats, got %d" % (dim, len(row)))
        if validate:
            if not all(math.isfinite(x) for x in row):
                raise ValueError("vector contains NaN or Inf")
            if not any(row):
                raise ValueError("refusing to store an all-zero vector")
        return row

    # ----------------------------------------------------------------- writes
    def append(self, node_id: int, vec, validate: bool = True) -> int:
        """Insert (or update) a row. Returns the row index."""
        node_id = int(node_id)
        r = self._row.get(node_id)
        if r is not None:
            self.update(node_id, vec, validate=validate)
            return r
        a = self._coerce(vec, validate=validate)
        if self._n >= self._cap:
            self._reserve(max(64, self._cap * 2))
        r = self._n
        if self._numpy:
            self._mat[r] = a
        else:  # pragma: no cover
            self._mat[r * self.dim:(r + 1) * self.dim] = _array("f", a)
        self._ids.append(node_id)
        self._row[node_id] = r
        self._n = r + 1
        self._norms_dirty = True
        self._version += 1
        return r

    def update(self, node_id: int, vec, validate: bool = True) -> None:
        r = self._row.get(int(node_id))
        if r is None:
            raise KeyError("no such node in VectorStore: %r" % (node_id,))
        a = self._coerce(vec, validate=validate)
        if self._numpy:
            self._mat[r] = a
        else:  # pragma: no cover
            self._mat[r * self.dim:(r + 1) * self.dim] = _array("f", a)
        self._norms_dirty = True
        self._version += 1

    def drop(self, node_id: int) -> bool:
        """Remove a row. The last row is swapped into the hole (O(1))."""
        r = self._row.pop(int(node_id), None)
        if r is None:
            return False
        last = self._n - 1
        if r != last:
            if self._numpy:
                self._mat[r] = self._mat[last]
            else:  # pragma: no cover
                self._mat[r * self.dim:(r + 1) * self.dim] = \
                    self._mat[last * self.dim:(last + 1) * self.dim]
            moved = self._ids[last]
            self._ids[r] = moved
            self._row[moved] = r
        self._ids.pop()
        self._n = last
        self._norms_dirty = True
        self._version += 1
        return True

    # ------------------------------------------------------------------ reads
    def get(self, node_id: int):
        """Row view (zero copy) or None. Read-only for numpy callers."""
        r = self._row.get(int(node_id))
        if r is None:
            return None
        if self._numpy:
            v = self._mat[r]
            v.flags.writeable = False
            return v
        return memoryview(self._mat)[r * self.dim:(r + 1) * self.dim]  # pragma: no cover

    def get_list(self, node_id: int) -> Optional[List[float]]:
        v = self.get(node_id)
        return None if v is None else [float(x) for x in v]

    def has(self, node_id: int) -> bool:
        return int(node_id) in self._row

    def ids(self) -> List[int]:
        return list(self._ids)

    def __len__(self) -> int:
        return self._n

    def __contains__(self, node_id) -> bool:
        return int(node_id) in self._row

    def __repr__(self) -> str:
        return "<VectorStore dim=%d rows=%d cap=%d v=%d>" % (
            self.dim, self._n, self._cap, self._version)

    @property
    def version(self) -> int:
        return self._version

    def _recompute_norms(self) -> None:
        n = self._n
        if self._numpy:
            if n:
                self._norms[:n] = _np.linalg.norm(self._mat[:n], axis=1)
        else:  # pragma: no cover
            for i in range(n):
                row = self._mat[i * self.dim:(i + 1) * self.dim]
                self._norms[i] = math.sqrt(math.fsum(float(x) * float(x) for x in row))
        self._norms_dirty = False

    def snapshot(self) -> Tuple[Tuple[int, ...], object, object, int]:
        """(ids, matrix, norms, version).

        ``matrix`` is a zero-copy view of shape (len(ids), dim) with numpy, so
        callers score the store in place instead of rebuilding a second matrix.
        The version bumps on every add/update/drop; cache on it, not on node
        count.
        """
        if self._norms_dirty or self._norms is None:
            self._recompute_norms()
        n = self._n
        if self._numpy:
            return (tuple(self._ids), self._mat[:n], self._norms[:n], self._version)
        return (tuple(self._ids),                        # pragma: no cover
                [self.get(i) for i in self._ids],
                list(self._norms[:n]), self._version)

    def matrix(self):
        """Zero-copy (rows, dim) view of the live rows."""
        if self._numpy:
            return self._mat[: self._n]
        return memoryview(self._mat)[: self._n * self.dim]  # pragma: no cover

    def iter_rows(self) -> Iterator[Tuple[int, object]]:
        for nid in self._ids:
            yield nid, self.get(nid)

    # ------------------------------------------------------------- validation
    def validate_all(self) -> None:
        """Bulk sanity check of a freshly loaded matrix (fast, one pass)."""
        n = self._n
        if not n:
            return
        if self._numpy:
            m = self._mat[:n]
            if not bool(_np.all(_np.isfinite(m))):
                bad = int(_np.nonzero(~_np.isfinite(m).all(axis=1))[0][0])
                raise ValueError(
                    "store contains a non-finite vector (node_id=%d)"
                    % self._ids[bad])
            norms = _np.linalg.norm(m, axis=1)
            bad = _np.nonzero(norms == 0)[0]
            if len(bad):
                raise ValueError(
                    "store contains %d zero-length vector(s); first node_id=%d"
                    % (len(bad), self._ids[int(bad[0])]))
        else:  # pragma: no cover
            for i in range(n):
                row = self._mat[i * self.dim:(i + 1) * self.dim]
                if not all(math.isfinite(float(x)) for x in row):
                    raise ValueError("store contains a non-finite vector "
                                     "(node_id=%d)" % self._ids[i])
                if not any(row):
                    raise ValueError("store contains a zero-length vector "
                                     "(node_id=%d)" % self._ids[i])
