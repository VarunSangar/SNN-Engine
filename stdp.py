"""
axiom_neuro/learning/stdp.py
==============================
Spike-Timing-Dependent Plasticity (STDP)
==========================================

Double-exponential STDP kernel:
    LTP (pre before post, Δt > 0):  ΔW = A_+ * exp(-Δt / τ_+)
    LTD (post before pre, Δt < 0):  ΔW = -A_- * exp( Δt / τ_-)

Implemented via eligibility traces (online, O(N) per step):
    x_pre[j]  += 1 on pre-spike,  decays as exp(-dt/τ_+)
    x_post[i] += 1 on post-spike, decays as exp(-dt/τ_-)

Weight update at each spike:
    pre  fires j: ΔW[i,j] = -A_- * x_post[i]   (LTD of all post→pre pairs)
    post fires i: ΔW[i,j] =  A_+ * x_pre[j]    (LTP of all pre→post pairs)

Homeostatic Plasticity (Turrigiano 2004):
    Synaptic scaling: W → W * (r_target / r_actual)^β
    Applied every homeostasis_interval steps.

References
----------
- Bi & Poo (1998) J. Neurosci.
- Song, Miller & Abbott (2000) Nature Neuroscience.
- Turrigiano (2004) Nature Reviews Neuroscience.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.sparse import csr_matrix, coo_matrix
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  STDP Parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class STDPParams:
    """
    Parameters for the double-exponential STDP rule.

    Attributes
    ----------
    A_plus              : float  LTP amplitude
    A_minus             : float  LTD amplitude
    tau_plus            : float  LTP time constant (ms)
    tau_minus           : float  LTD time constant (ms)
    w_min, w_max        : float  Weight bounds
    learning_rate       : float  Global scaling factor
    homeostasis_enabled : bool   Apply synaptic scaling
    r_target            : float  Target firing rate for homeostasis (Hz)
    homeostasis_beta    : float  Homeostasis gain
    homeostasis_interval: int    Apply every N steps
    """
    A_plus:               float = 0.01
    A_minus:              float = 0.0105   # Slight LTD dominance → stability
    tau_plus:             float = 20.0     # ms
    tau_minus:            float = 20.0     # ms
    w_min:                float = 0.0
    w_max:                float = 5.0
    learning_rate:        float = 1.0
    homeostasis_enabled:  bool  = True
    r_target:             float = 5.0      # Hz
    homeostasis_beta:     float = 0.1
    homeostasis_interval: int   = 1000     # steps


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Online STDP Engine
# ─────────────────────────────────────────────────────────────────────────────

class STDPEngine:
    """
    Online STDP via eligibility traces.

    Supports recurrent (n_pre == n_post) and feedforward connections.

    Usage
    -----
    >>> stdp = STDPEngine(STDPParams(), W_sparse, n_pre=1000, n_post=1000)
    >>> dW = stdp.step(spikes_pre, spikes_post, dt=0.1)
    >>> W_sparse.apply_weight_matrix_delta(dW)
    """

    def __init__(
        self,
        params:   STDPParams,
        W:        "SparseWeightMatrix",  # forward ref
        n_pre:    int,
        n_post:   int,
        dt:       float = 0.1,
    ):
        self.p      = params
        self.W      = W
        self.n_pre  = n_pre
        self.n_post = n_post
        self.dt     = dt

        # Eligibility traces
        self.x_pre  = np.zeros(n_pre,  dtype=np.float64)   # LTP trace
        self.x_post = np.zeros(n_post, dtype=np.float64)   # LTD trace

        # Exponential decay factors (precomputed)
        self._decay_plus  = np.exp(-dt / params.tau_plus)
        self._decay_minus = np.exp(-dt / params.tau_minus)

        self._step_count = 0

        # History for analysis
        self.dW_history: list[float] = []   # mean |ΔW| per step

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(
        self,
        spikes_pre:  np.ndarray,   # (n_pre,)  bool
        spikes_post: np.ndarray,   # (n_post,) bool
        firing_rates: Optional[np.ndarray] = None,  # (n_post,) Hz — for homeostasis
    ) -> csr_matrix:
        """
        Compute ΔW for this timestep and optionally apply homeostasis.

        Returns
        -------
        dW : sparse (n_post × n_pre) weight-change matrix
        """
        p = self.p

        # 1. Decay traces
        self.x_pre  *= self._decay_plus
        self.x_post *= self._decay_minus

        # 2. Update traces at spike sites
        self.x_pre[spikes_pre]   += 1.0
        self.x_post[spikes_post] += 1.0

        # 3. Compute ΔW via outer products on sparse support
        #    Only update existing synapses (sparse structure preserved)
        dW = self._compute_sparse_dW(spikes_pre, spikes_post)

        # 4. Apply to weight matrix
        self.W.apply_weight_matrix_delta(dW)

        # 5. Homeostatic plasticity
        self._step_count += 1
        if p.homeostasis_enabled and (self._step_count % p.homeostasis_interval == 0):
            if firing_rates is not None:
                self._apply_homeostasis(firing_rates)

        # Track mean update magnitude
        if dW.nnz > 0:
            self.dW_history.append(float(np.mean(np.abs(dW.data))))

        return dW

    def _compute_sparse_dW(
        self,
        spikes_pre:  np.ndarray,
        spikes_post: np.ndarray,
    ) -> csr_matrix:
        """
        Efficiently compute ΔW only at existing synaptic connections.

        For each nonzero (i,j) in W:
            If post i fired:  ΔW[i,j] += A_+ * lr * x_pre[j]   (LTP)
            If pre  j fired:  ΔW[i,j] -= A_- * lr * x_post[i]  (LTD)
        """
        W_csr  = self.W._W_csr
        p      = self.p
        lr     = p.learning_rate

        # Get COO indices of existing synapses
        W_coo  = W_csr.tocoo()
        rows   = W_coo.row    # post indices
        cols   = W_coo.col    # pre  indices
        n_syn  = len(rows)

        dw = np.zeros(n_syn, dtype=np.float64)

        # LTP: post fired — potentiate based on pre trace
        fired_post_mask = spikes_post[rows]
        dw[fired_post_mask] += (
            lr * p.A_plus * self.x_pre[cols[fired_post_mask]]
        )

        # LTD: pre fired — depress based on post trace
        fired_pre_mask = spikes_pre[cols]
        dw[fired_pre_mask] -= (
            lr * p.A_minus * self.x_post[rows[fired_pre_mask]]
        )

        return coo_matrix(
            (dw.astype(np.float32), (rows, cols)),
            shape=(self.n_post, self.n_pre),
        ).tocsr()

    # ── Homeostatic Plasticity ─────────────────────────────────────────────────

    def _apply_homeostasis(self, firing_rates: np.ndarray) -> None:
        """
        Synaptic scaling (Turrigiano 2004):
            W_i → W_i * (r_target / r_i)^β

        Applied row-by-row (per post-synaptic neuron).
        """
        p     = self.p
        W_csr = self.W._W_csr

        # Avoid division by zero
        r_safe = np.maximum(firing_rates, 0.1)   # min 0.1 Hz
        scale  = (p.r_target / r_safe) ** p.homeostasis_beta  # (n_post,)

        # Multiply each row i by scale[i]
        # Efficient: W.data[W.indptr[i]:W.indptr[i+1]] *= scale[i]
        for i in range(self.n_post):
            start = W_csr.indptr[i]
            end   = W_csr.indptr[i + 1]
            if start < end:
                W_csr.data[start:end] *= scale[i]

        # Clip to bounds
        W_csr.data[:] = np.clip(W_csr.data, p.w_min, p.w_max)

    # ── Analytics ─────────────────────────────────────────────────────────────

    def reset_traces(self) -> None:
        self.x_pre[:]  = 0.0
        self.x_post[:] = 0.0

    @property
    def mean_trace_pre(self) -> float:
        return float(self.x_pre.mean())

    @property
    def mean_trace_post(self) -> float:
        return float(self.x_post.mean())


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BCM (Bienenstock-Cooper-Munro) — bonus learning rule
# ─────────────────────────────────────────────────────────────────────────────

class BCMRule:
    """
    BCM sliding threshold plasticity.

    ΔW[i,j] = η * y_i * (y_i - θ_i) * x_j

    where θ_i = <y_i²> / <y_i>  is the sliding threshold.

    Reference: Bienenstock, Cooper & Munro (1982).
    """

    def __init__(self, n_pre: int, n_post: int, eta: float = 0.001, tau_theta: float = 1000.0, dt: float = 0.1):
        self.n_pre     = n_pre
        self.n_post    = n_post
        self.eta       = eta
        self.tau_theta = tau_theta
        self.dt        = dt

        self.theta = np.ones(n_post, dtype=np.float64) * 0.5
        self._decay_theta = np.exp(-dt / tau_theta)

    def step(
        self,
        activity_pre:  np.ndarray,   # (n_pre,)
        activity_post: np.ndarray,   # (n_post,)
    ) -> np.ndarray:
        """Returns dense ΔW (n_post × n_pre). Use for small populations."""
        y = activity_post  # (n_post,)
        x = activity_pre   # (n_pre,)

        # Update threshold
        self.theta *= self._decay_theta
        self.theta += (1.0 - self._decay_theta) * (y ** 2)

        # BCM update: outer product
        mod = y * (y - self.theta)   # (n_post,)
        dW  = self.eta * np.outer(mod, x)   # (n_post, n_pre)
        return dW
