"""
axiom_neuro/geometry/manifold_mapper.py
=========================================
Minkowski–Convex-Hull Information Manifold Mapper
===================================================

At each timestep t, the 'firing population' F(t) = { i : spike_i(t) = 1 }
is mapped to a 3-D point cloud via a learned or random embedding:

    p_i = Φ(i) ∈ ℝ³

The convex hull of { p_i | i ∈ F(t) } is the 'Information Manifold':
    M(t) = conv({ Φ(i) | i ∈ F(t) })

Its geometric properties (volume, surface area, shape index) quantify
the 'information state' of the network at time t.

Minkowski Sum of consecutive manifolds:
    M(t) ⊕ M(t+1) = { a + b | a ∈ M(t), b ∈ M(t+1) }

encodes the geometric transition of network state.

The module also computes:
  - Manifold Volume V(t)
  - Manifold Surface Area A(t)
  - Isoperimetric ratio I(t) = 36π V² / A³  (sphere = 1)
  - Convex hull facets for 3D rendering
  - Minkowski Sum of two consecutive hulls
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Neuron Embedding
# ─────────────────────────────────────────────────────────────────────────────

class NeuronEmbedding:
    """
    Maps neuron indices to 3-D coordinates Φ: {0..N-1} → ℝ³.

    Strategies:
      'random_sphere'  : uniformly on unit sphere
      'toroidal'       : wrap-around 3-D torus
      'pca'            : project weight matrix onto first 3 PCs (learned)
    """

    def __init__(self, n_neurons: int, strategy: str = 'random_sphere', seed: int = 0):
        self.n  = n_neurons
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)
        self.coords = self._build_embedding()   # (N, 3)

    def _build_embedding(self) -> np.ndarray:
        if self.strategy == 'random_sphere':
            pts = self.rng.standard_normal((self.n, 3))
            pts /= (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12)
            return pts

        elif self.strategy == 'toroidal':
            idx = np.arange(self.n)
            side = int(np.ceil(self.n ** (1/3)))
            x = (idx % side) / side * 2 * np.pi
            y = ((idx // side) % side) / side * 2 * np.pi
            z = (idx // (side ** 2)) / side * 2 * np.pi
            pts = np.column_stack([
                np.cos(x), np.cos(y), np.cos(z)
            ])
            return pts.astype(np.float64)

        elif self.strategy == 'grid':
            side = int(np.ceil(self.n ** (1/3)))
            xs = np.linspace(-1, 1, side)
            grid = np.array([(x,y,z) for x in xs for y in xs for z in xs])
            return grid[:self.n]

        else:
            raise ValueError(f"Unknown embedding strategy: {self.strategy}")

    def from_weight_pca(self, W_dense: np.ndarray) -> None:
        """Update embedding via PCA of weight matrix rows."""
        W_c = W_dense - W_dense.mean(axis=0)
        _, _, Vt = np.linalg.svd(W_c, full_matrices=False)
        self.coords = (W_c @ Vt[:3].T)
        # Normalise to unit sphere
        r = np.linalg.norm(self.coords, axis=1, keepdims=True)
        self.coords /= (r + 1e-12)
        self.strategy = 'pca'

    def get(self, indices: np.ndarray) -> np.ndarray:
        """Return (k, 3) point cloud for given neuron indices."""
        return self.coords[indices]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Convex Hull Geometry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ManifoldSnapshot:
    """
    Geometric snapshot of the firing-population manifold at one timestep.
    """
    t:              float
    n_firing:       int
    hull_vertices:  Optional[np.ndarray]   = None   # (V, 3)
    hull_simplices: Optional[np.ndarray]   = None   # (F, 3) triangle indices
    volume:         float                  = 0.0
    area:           float                  = 0.0
    isoperimetric:  float                  = 0.0
    centroid:       Optional[np.ndarray]   = None   # (3,)
    valid:          bool                   = False


def _hull_mesh_arrays(points: np.ndarray, hull: ConvexHull):
    """Return (x,y,z, i,j,k) for Plotly Mesh3d."""
    verts = points[hull.vertices]
    g2l   = {g: l for l, g in enumerate(hull.vertices)}
    tri   = np.array([[g2l[s] for s in face] for face in hull.simplices])
    return verts, tri


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Manifold Mapper
# ─────────────────────────────────────────────────────────────────────────────

class ManifoldMapper:
    """
    Computes the Information Manifold from firing patterns.

    Usage
    -----
    >>> emb    = NeuronEmbedding(n_neurons=1000)
    >>> mapper = ManifoldMapper(emb, min_firing=10)
    >>> snap   = mapper.update(t=10.0, spike_indices=np.array([3,17,42,...]))
    >>> print(snap.volume, snap.area)
    """

    def __init__(
        self,
        embedding:   NeuronEmbedding,
        min_firing:  int   = 4,     # minimum spikes to attempt hull
        history_len: int   = 500,   # max snapshots to keep
    ):
        self.emb         = embedding
        self.min_firing  = min_firing
        self.history_len = history_len
        self.history:    List[ManifoldSnapshot] = []

    def update(self, t: float, spike_indices: np.ndarray) -> ManifoldSnapshot:
        """
        Compute manifold for current timestep.

        Parameters
        ----------
        t             : simulation time (ms)
        spike_indices : (k,) int array of firing neuron indices

        Returns
        -------
        ManifoldSnapshot with hull geometry.
        """
        snap = ManifoldSnapshot(t=t, n_firing=len(spike_indices))

        if len(spike_indices) >= self.min_firing:
            pts = self.emb.get(spike_indices)  # (k, 3)
            pts = np.unique(pts, axis=0)        # deduplicate

            if len(pts) >= 4:
                try:
                    hull = ConvexHull(pts)
                    verts, tri = _hull_mesh_arrays(pts, hull)

                    snap.hull_vertices  = verts
                    snap.hull_simplices = tri
                    snap.volume         = hull.volume
                    snap.area           = hull.area
                    snap.centroid       = pts[hull.vertices].mean(axis=0)
                    snap.isoperimetric  = self._isoperimetric_ratio(hull.volume, hull.area)
                    snap.valid          = True
                except Exception:
                    pass   # degenerate point set

        # Append to history (bounded)
        self.history.append(snap)
        if len(self.history) > self.history_len:
            self.history.pop(0)

        return snap

    @staticmethod
    def _isoperimetric_ratio(V: float, A: float) -> float:
        """
        Isoperimetric ratio I = 36π V² / A³
        I = 1 for a sphere (maximum), I < 1 for all other shapes.
        """
        if A < 1e-12:
            return 0.0
        return 36.0 * np.pi * V ** 2 / (A ** 3)

    # ── Minkowski Sum of consecutive manifolds ────────────────────────────────

    def minkowski_sum_consecutive(
        self,
        idx1: int = -2,
        idx2: int = -1,
    ) -> Optional[np.ndarray]:
        """
        Compute Minkowski Sum of two consecutive manifold hulls.
        Returns vertex set of M(t1) ⊕ M(t2), or None if not enough history.
        """
        if len(self.history) < 2:
            return None

        s1 = self.history[idx1]
        s2 = self.history[idx2]

        if not (s1.valid and s2.valid):
            return None

        V1 = s1.hull_vertices
        V2 = s2.hull_vertices

        # Minkowski Sum: all pairwise sums, then convex hull
        sums = (V1[:, None, :] + V2[None, :, :]).reshape(-1, 3)
        try:
            hull = ConvexHull(sums)
            return sums[hull.vertices]
        except Exception:
            return sums

    # ── Time-series extraction ────────────────────────────────────────────────

    def get_volume_trace(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (times_ms, volumes) for valid snapshots."""
        valid = [s for s in self.history if s.valid]
        t = np.array([s.t for s in valid])
        v = np.array([s.volume for s in valid])
        return t, v

    def get_area_trace(self) -> Tuple[np.ndarray, np.ndarray]:
        valid = [s for s in self.history if s.valid]
        t = np.array([s.t for s in valid])
        a = np.array([s.area for s in valid])
        return t, a

    def get_isoperimetric_trace(self) -> Tuple[np.ndarray, np.ndarray]:
        valid = [s for s in self.history if s.valid]
        t = np.array([s.t for s in valid])
        iso = np.array([s.isoperimetric for s in valid])
        return t, iso

    def latest_valid(self) -> Optional[ManifoldSnapshot]:
        for s in reversed(self.history):
            if s.valid:
                return s
        return None
