"""
axiom_neuro/core/lif_model.py
==============================
Leaky Integrate-and-Fire (LIF) Neuron Population Model
=======================================================

Membrane dynamics (Euler integration):
    τ_m dV/dt = -(V - V_rest) + R_m * I(t)

Discrete update (dt step):
    V[t+1] = V[t] + (dt/τ_m) * (-(V[t] - V_rest) + R_m * I[t])

After spike:
    V → V_reset,  refractory counter set

All operations are fully vectorized over the population using NumPy.
Numba @njit kernels are provided for the inner-loop hot path.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Optional Numba JIT ────────────────────────────────────────────────────────
try:
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    # Graceful fallback — define identity decorator
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator
    prange = range
    _NUMBA = False


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Parameter Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LIFParams:
    """
    Biophysical parameters for the LIF population.

    Attributes
    ----------
    tau_m       : float   Membrane time constant (ms)
    V_rest      : float   Resting potential (mV)
    V_thresh    : float   Spike threshold (mV)
    V_reset     : float   Reset potential post-spike (mV)
    R_m         : float   Membrane resistance (MΩ)
    t_refrac    : float   Absolute refractory period (ms)
    dt          : float   Euler integration step (ms)
    n_neurons   : int     Population size
    noise_sigma : float   Gaussian current noise std (nA) — biological variability
    """
    tau_m:       float = 20.0     # ms
    V_rest:      float = -65.0    # mV
    V_thresh:    float = -50.0    # mV
    V_reset:     float = -70.0    # mV
    R_m:         float = 10.0     # MΩ
    t_refrac:    float = 2.0      # ms
    dt:          float = 0.1      # ms
    n_neurons:   int   = 1000
    noise_sigma: float = 0.5      # nA

    @property
    def refrac_steps(self) -> int:
        return max(1, int(self.t_refrac / self.dt))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Numba JIT kernels
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _lif_euler_kernel(
    V:             np.ndarray,   # (N,)  membrane potentials
    refrac_count:  np.ndarray,   # (N,)  refractory counters (integer steps)
    I_ext:         np.ndarray,   # (N,)  external + synaptic current (nA)
    spikes:        np.ndarray,   # (N,)  bool output array
    dt:            float,
    tau_m:         float,
    V_rest:        float,
    V_thresh:      float,
    V_reset:       float,
    R_m:           float,
    noise_sigma:   float,
    noise:         np.ndarray,   # (N,)  pre-drawn noise
) -> None:
    """
    In-place LIF Euler step for an entire neuron population.
    Parallelised with prange (Numba parallel=True).

    Parameters mutated in-place: V, refrac_count, spikes.
    """
    N = V.shape[0]
    dt_over_tau = dt / tau_m

    for i in prange(N):
        spikes[i] = False

        if refrac_count[i] > 0:
            refrac_count[i] -= 1
            V[i] = V_reset
            continue

        # Euler: dV = (dt/τ) * (-(V-V_rest) + R_m*I + noise)
        dV = dt_over_tau * (
            -(V[i] - V_rest)
            + R_m * (I_ext[i] + noise_sigma * noise[i])
        )
        V[i] += dV

        if V[i] >= V_thresh:
            spikes[i] = True
            V[i]      = V_reset
            refrac_count[i] = int(2.0 / dt)   # refrac_steps (passed as scalar ok)


@njit(cache=True)
def _lif_euler_kernel_serial(
    V, refrac_count, I_ext, spikes,
    dt, tau_m, V_rest, V_thresh, V_reset, R_m,
    noise_sigma, noise
) -> None:
    """Serial fallback (same logic, no prange)."""
    N = V.shape[0]
    dt_over_tau = dt / tau_m
    for i in range(N):
        spikes[i] = False
        if refrac_count[i] > 0:
            refrac_count[i] -= 1
            V[i] = V_reset
            continue
        dV = dt_over_tau * (
            -(V[i] - V_rest) + R_m * (I_ext[i] + noise_sigma * noise[i])
        )
        V[i] += dV
        if V[i] >= V_thresh:
            spikes[i] = True
            V[i] = V_reset
            refrac_count[i] = int(2.0 / dt)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Population State
# ─────────────────────────────────────────────────────────────────────────────

class LIFPopulation:
    """
    Vectorized LIF population.

    State arrays (all shape (N,)):
        V             : membrane potential (mV)
        refrac_count  : refractory countdown (integer steps)
        spikes        : boolean spike indicator (current step)
        spike_times   : list of (t, neuron_idx) — spike history

    Usage
    -----
    >>> pop = LIFPopulation(LIFParams(n_neurons=50_000))
    >>> I = np.ones(50_000) * 2.0   # nA
    >>> spikes = pop.step(t=0.0, I_ext=I)
    """

    def __init__(self, params: LIFParams, seed: int = 42):
        self.p    = params
        self.N    = params.n_neurons
        self.rng  = np.random.default_rng(seed)
        self._use_parallel = _NUMBA

        # State
        self.V            = np.full(self.N, params.V_rest, dtype=np.float64)
        self.refrac_count = np.zeros(self.N, dtype=np.int32)
        self.spikes       = np.zeros(self.N, dtype=np.bool_)

        # Slight initial dispersion (biological variability)
        self.V += self.rng.uniform(-2.0, 2.0, self.N)

        # Spike history: list of arrays, one per timestep
        self.spike_history: list[np.ndarray] = []
        self.t_history:     list[float]       = []

        # Running firing-rate estimate per neuron (for homeostasis)
        self.firing_rate_estimate = np.zeros(self.N, dtype=np.float64)
        self._fr_tau = 100.0  # ms — homeostatic time constant

    # ── Core step ─────────────────────────────────────────────────────────────

    def step(self, t: float, I_ext: np.ndarray) -> np.ndarray:
        """
        Advance population by one dt.

        Parameters
        ----------
        t     : current simulation time (ms)
        I_ext : (N,) external + synaptic current array (nA)

        Returns
        -------
        spikes : (N,) bool array — True where neuron fired
        """
        noise = self.rng.standard_normal(self.N)
        p = self.p

        if self._use_parallel:
            _lif_euler_kernel(
                self.V, self.refrac_count, I_ext, self.spikes,
                p.dt, p.tau_m, p.V_rest, p.V_thresh, p.V_reset, p.R_m,
                p.noise_sigma, noise,
            )
        else:
            self._step_numpy(I_ext, noise)

        # Update running firing rate (exponential moving average)
        alpha = p.dt / self._fr_tau
        self.firing_rate_estimate *= (1.0 - alpha)
        self.firing_rate_estimate[self.spikes] += alpha / p.dt * 1000.0  # Hz

        # Record
        fired_idx = np.where(self.spikes)[0]
        self.spike_history.append(fired_idx)
        self.t_history.append(t)

        return self.spikes.copy()

    def _step_numpy(self, I_ext: np.ndarray, noise: np.ndarray) -> None:
        """Pure-NumPy fallback (no Numba required)."""
        p = self.p
        dt_over_tau = p.dt / p.tau_m

        # Refractory mask
        in_refrac = self.refrac_count > 0
        self.refrac_count[in_refrac] -= 1
        self.V[in_refrac] = p.V_reset

        # Free neurons
        free = ~in_refrac
        dV = dt_over_tau * (
            -(self.V[free] - p.V_rest)
            + p.R_m * (I_ext[free] + p.noise_sigma * noise[free])
        )
        self.V[free] += dV

        # Threshold crossing
        fired = free & (self.V >= p.V_thresh)
        self.spikes[:] = False
        self.spikes[fired] = True
        self.V[fired] = p.V_reset
        self.refrac_count[fired] = p.refrac_steps

    # ── Utilities ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all state to initial conditions."""
        self.V[:] = self.p.V_rest + self.rng.uniform(-2.0, 2.0, self.N)
        self.refrac_count[:] = 0
        self.spikes[:] = False
        self.spike_history.clear()
        self.t_history.clear()
        self.firing_rate_estimate[:] = 0.0

    def mean_firing_rate(self) -> float:
        """Population-mean firing rate (Hz) over full recorded history."""
        if not self.t_history:
            return 0.0
        total_spikes  = sum(len(s) for s in self.spike_history)
        total_time_s  = (self.t_history[-1] - self.t_history[0] + self.p.dt) * 1e-3
        return total_spikes / (self.N * total_time_s + 1e-12)

    @property
    def V_mean(self) -> float:
        return float(np.mean(self.V))

    @property
    def n_firing(self) -> int:
        return int(np.sum(self.spikes))

    def get_raster(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (times_ms, neuron_indices) arrays for raster plotting.
        """
        times, neurons = [], []
        for t, idx in zip(self.t_history, self.spike_history):
            times.extend([t] * len(idx))
            neurons.extend(idx.tolist())
        return np.array(times), np.array(neurons, dtype=np.int32)
