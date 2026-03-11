"""
axiom_neuro/core/synaptic_matrix.py
=====================================
Sparse Synaptic Weight Matrix
==============================

Memory-efficient representation of the N×N connectivity matrix using
scipy sparse CSR format. For N=50,000 at 1% connectivity, this reduces
memory from 50000² × 8 bytes = 20 GB → ~20 MB.

Also provides fast synaptic current injection:
    I_syn[i] = Σ_j  W[i,j] * spike[j] * g(t - t_j)   (conductance model)

Simplified instantaneous version used here:
    I_syn = W @ spikes   (current-based synapse)
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse import random as sparse_random
from dataclasses import dataclass
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Synapse Parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SynapseParams:
    """
    Parameters for the synaptic weight matrix.

    Attributes
    ----------
    n_pre       : int    Pre-synaptic population size
    n_post      : int    Post-synaptic population size
    density     : float  Connection probability (0..1)
    w_init_mean : float  Initial weight mean (nA / spike)
    w_init_std  : float  Initial weight std
    w_min       : float  Weight lower bound (hard clip)
    w_max       : float  Weight upper bound (hard clip)
    exc_fraction: float  Fraction of excitatory synapses (Dale's law)
    """
    n_pre:        int   = 1000
    n_post:       int   = 1000
    density:      float = 0.05
    w_init_mean:  float = 0.5
    w_init_std:   float = 0.2
    w_min:        float = 0.0
    w_max:        float = 5.0
    exc_fraction: float = 0.8    # 80% excitatory, 20% inhibitory (Dale's law)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Sparse Weight Matrix
# ─────────────────────────────────────────────────────────────────────────────

class SparseWeightMatrix:
    """
    Sparse (CSR) representation of the synaptic weight matrix W[post, pre].

    Current injection:
        I_syn[post] = W @ spikes_pre   (shape: n_post,)

    Supports in-place STDP updates via COO access.

    Usage
    -----
    >>> W = SparseWeightMatrix(SynapseParams(n_pre=1000, n_post=1000, density=0.05))
    >>> I = W.compute_current(spikes_pre)
    """

    def __init__(self, params: SynapseParams, seed: int = 0):
        self.p    = params
        self.rng  = np.random.default_rng(seed)
        self._build_matrix()

    def _build_matrix(self) -> None:
        p   = self.p
        rng = self.rng

        # Generate sparse mask
        W_lil = lil_matrix((p.n_post, p.n_pre), dtype=np.float32)

        # Efficient batch construction
        n_synapses = int(p.density * p.n_pre * p.n_post)
        rows = rng.integers(0, p.n_post, n_synapses)
        cols = rng.integers(0, p.n_pre,  n_synapses)

        # Weights: excitatory (+) or inhibitory (-)
        weights = rng.normal(p.w_init_mean, p.w_init_std, n_synapses).astype(np.float32)
        weights = np.clip(weights, p.w_min, p.w_max)

        # Dale's law: last exc_fraction% of neurons are excitatory
        n_inh_pre = int((1.0 - p.exc_fraction) * p.n_pre)
        inh_mask  = cols < n_inh_pre
        weights[inh_mask] *= -1.0  # inhibitory → negative

        W_lil[rows, cols] = weights.reshape(-1)  # lil supports this via index

        # Convert to CSR for fast matrix-vector multiply
        self._W_csr: csr_matrix = W_lil.tocsr()

        # Cache COO arrays for fast STDP updates
        self._W_coo = self._W_csr.tocoo()

    # ── Current injection ─────────────────────────────────────────────────────

    def compute_current(self, spikes_pre: np.ndarray) -> np.ndarray:
        """
        Compute post-synaptic current.

        Parameters
        ----------
        spikes_pre : (n_pre,) bool or float array

        Returns
        -------
        I_syn : (n_post,) float64 array (nA)
        """
        return self._W_csr.dot(spikes_pre.astype(np.float32)).astype(np.float64)

    # ── Weight access ─────────────────────────────────────────────────────────

    def get_weights_dense(self) -> np.ndarray:
        """Return dense (n_post, n_pre) weight matrix. Use only for small N."""
        return self._W_csr.toarray()

    def get_weight(self, post: int, pre: int) -> float:
        return float(self._W_csr[post, pre])

    def set_weight(self, post: int, pre: int, value: float) -> None:
        self._W_csr[post, pre] = np.clip(value, self.p.w_min, self.p.w_max)

    def apply_weight_delta(self, dW_rows: np.ndarray, dW_cols: np.ndarray, dW_vals: np.ndarray) -> None:
        """
        Batch in-place weight update for STDP.

        Parameters
        ----------
        dW_rows, dW_cols : index arrays (post, pre)
        dW_vals          : delta weight values
        """
        # Work in LIL format for random access, then convert back
        W_lil = self._W_csr.tolil()
        for r, c, dw in zip(dW_rows, dW_cols, dW_vals):
            old = W_lil[r, c]
            W_lil[r, c] = float(np.clip(old + dw, self.p.w_min, self.p.w_max))
        self._W_csr = W_lil.tocsr()

    def apply_weight_matrix_delta(self, dW: csr_matrix) -> None:
        """Add a sparse dW matrix directly (fast path for STDP)."""
        self._W_csr = self._W_csr + dW
        # Clip weights
        self._W_csr.data[:] = np.clip(self._W_csr.data, self.p.w_min, self.p.w_max)

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def n_synapses(self) -> int:
        return self._W_csr.nnz

    @property
    def mean_weight(self) -> float:
        return float(self._W_csr.data.mean()) if self._W_csr.nnz > 0 else 0.0

    @property
    def weight_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (hist_counts, bin_edges) of weight distribution."""
        return np.histogram(self._W_csr.data, bins=50)

    @property
    def memory_bytes(self) -> int:
        """Memory footprint of CSR arrays."""
        W = self._W_csr
        return (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes)

    def density_actual(self) -> float:
        return self._W_csr.nnz / (self.p.n_post * self.p.n_pre)

    def __repr__(self) -> str:
        return (
            f"SparseWeightMatrix("
            f"shape=({self.p.n_post}×{self.p.n_pre}), "
            f"nnz={self.n_synapses:,}, "
            f"density={self.density_actual():.4f}, "
            f"mem={self.memory_bytes/1024:.1f} KB)"
        )
