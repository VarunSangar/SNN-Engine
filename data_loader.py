"""
axiom_neuro/io/data_loader.py
===============================
Neural Data I/O & Replay Engine
=================================

Supports:
  - CSV spike train ingestion  (columns: time_ms, neuron_id, [optional: value])
  - NEF / NWB-lite flat export
  - Replay: feed recorded spike trains through SNN to find optimal weights
  - Synthetic data generators for testing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Spike Train Data Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpikeData:
    """
    Container for a recorded spike train.

    Attributes
    ----------
    times      : (S,) float64  — spike times in ms
    neurons    : (S,) int32    — neuron indices
    n_neurons  : int           — population size (if known)
    dt         : float         — original time step (ms)
    metadata   : dict          — experiment metadata
    """
    times:     np.ndarray
    neurons:   np.ndarray
    n_neurons: int   = 0
    dt:        float = 0.1
    metadata:  dict  = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.n_neurons == 0:
            self.n_neurons = int(self.neurons.max()) + 1 if len(self.neurons) else 0

    @property
    def duration_ms(self) -> float:
        return float(self.times.max() - self.times.min()) if len(self.times) else 0.0

    @property
    def n_spikes(self) -> int:
        return len(self.times)

    @property
    def mean_firing_rate(self) -> float:
        """Population mean firing rate (Hz)."""
        dur = self.duration_ms * 1e-3
        return self.n_spikes / (self.n_neurons * dur + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CSV Loader
# ─────────────────────────────────────────────────────────────────────────────

class SpikeDataLoader:
    """
    Load spike trains from CSV files.

    Expected CSV format:
        time_ms,neuron_id[,value]
        0.1,42
        0.3,17
        ...

    Usage
    -----
    >>> loader = SpikeDataLoader()
    >>> data = loader.load_csv("spikes.csv", n_neurons=1000)
    """

    def load_csv(
        self,
        path:      str,
        n_neurons: Optional[int]  = None,
        dt:        float          = 0.1,
        time_col:  str            = "time_ms",
        neuron_col:str            = "neuron_id",
        max_rows:  Optional[int]  = None,
    ) -> SpikeData:
        """Load a CSV spike file."""
        df = pd.read_csv(path, nrows=max_rows)

        # Auto-detect column names
        cols = {c.lower().strip(): c for c in df.columns}
        tc = cols.get(time_col.lower(), df.columns[0])
        nc = cols.get(neuron_col.lower(), df.columns[1])

        times   = df[tc].values.astype(np.float64)
        neurons = df[nc].values.astype(np.int32)

        # Sort by time
        order   = np.argsort(times)
        times   = times[order]
        neurons = neurons[order]

        N = n_neurons if n_neurons else int(neurons.max()) + 1

        return SpikeData(
            times=times, neurons=neurons, n_neurons=N, dt=dt,
            metadata={"source": str(path), "n_rows": len(times)},
        )

    def load_npy(self, times_path: str, neurons_path: str, **kwargs) -> SpikeData:
        """Load from .npy arrays."""
        times   = np.load(times_path).astype(np.float64)
        neurons = np.load(neurons_path).astype(np.int32)
        return SpikeData(times=times, neurons=neurons, **kwargs)

    def save_csv(self, data: SpikeData, path: str) -> None:
        """Export spike train to CSV."""
        df = pd.DataFrame({"time_ms": data.times, "neuron_id": data.neurons})
        df.to_csv(path, index=False)
        print(f"Saved {len(data.times):,} spikes → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Synthetic Data Generators
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Generate synthetic spike trains for testing & benchmarking.
    """

    @staticmethod
    def poisson_spikes(
        n_neurons:   int,
        duration_ms: float,
        rate_hz:     float = 10.0,
        dt:          float = 0.1,
        seed:        int   = 0,
    ) -> SpikeData:
        """
        Homogeneous Poisson process spike trains.
        P(spike in dt) = rate * dt * 1e-3
        """
        rng = np.random.default_rng(seed)
        p   = rate_hz * dt * 1e-3
        n_steps = int(duration_ms / dt)

        times, neurons = [], []
        for step in range(n_steps):
            fired = np.where(rng.random(n_neurons) < p)[0]
            if len(fired):
                times.extend([step * dt] * len(fired))
                neurons.extend(fired.tolist())

        return SpikeData(
            times=np.array(times, dtype=np.float64),
            neurons=np.array(neurons, dtype=np.int32),
            n_neurons=n_neurons, dt=dt,
            metadata={"type": "poisson", "rate_hz": rate_hz},
        )

    @staticmethod
    def burst_spikes(
        n_neurons:   int,
        duration_ms: float,
        burst_rate:  float = 2.0,    # bursts/s per neuron
        burst_len:   float = 10.0,   # ms
        intra_rate:  float = 100.0,  # Hz within burst
        dt:          float = 0.1,
        seed:        int   = 0,
    ) -> SpikeData:
        """Bursty spike trains (common in cortical data)."""
        rng = np.random.default_rng(seed)
        n_steps = int(duration_ms / dt)
        p_burst_start = burst_rate * dt * 1e-3
        burst_steps   = int(burst_len / dt)
        p_intra       = intra_rate * dt * 1e-3

        times, neurons = [], []
        in_burst = np.zeros(n_neurons, dtype=int)  # countdown steps

        for step in range(n_steps):
            t = step * dt
            # Start new bursts
            new_bursts = (in_burst == 0) & (rng.random(n_neurons) < p_burst_start)
            in_burst[new_bursts] = burst_steps
            # Fire within bursts
            burst_mask = in_burst > 0
            fired = burst_mask & (rng.random(n_neurons) < p_intra)
            in_burst[burst_mask] -= 1
            idx = np.where(fired)[0]
            if len(idx):
                times.extend([t] * len(idx))
                neurons.extend(idx.tolist())

        return SpikeData(
            times=np.array(times, dtype=np.float64),
            neurons=np.array(neurons, dtype=np.int32),
            n_neurons=n_neurons, dt=dt,
            metadata={"type": "burst"},
        )

    @staticmethod
    def oscillatory_spikes(
        n_neurons:   int,
        duration_ms: float,
        freq_hz:     float = 40.0,   # gamma oscillation
        base_rate:   float = 5.0,    # Hz background
        dt:          float = 0.1,
        seed:        int   = 0,
    ) -> SpikeData:
        """Oscillatory (gamma-band) spike trains."""
        rng  = np.random.default_rng(seed)
        n_steps = int(duration_ms / dt)
        times, neurons = [], []

        for step in range(n_steps):
            t = step * dt
            # Modulated rate: base_rate * (1 + sin(2π*freq*t/1000))
            inst_rate = base_rate * (1.0 + np.sin(2*np.pi*freq_hz*t*1e-3))
            p = inst_rate * dt * 1e-3
            fired = np.where(rng.random(n_neurons) < p)[0]
            if len(fired):
                times.extend([t] * len(fired))
                neurons.extend(fired.tolist())

        return SpikeData(
            times=np.array(times, dtype=np.float64),
            neurons=np.array(neurons, dtype=np.int32),
            n_neurons=n_neurons, dt=dt,
            metadata={"type": "oscillatory", "freq_hz": freq_hz},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Replay Engine
# ─────────────────────────────────────────────────────────────────────────────

class ReplayEngine:
    """
    Replay recorded spike trains through an SNN to find optimal weights.

    Strategy:
      For each recorded timestep, inject the recorded spike pattern as
      a 'teaching signal'. The STDP rule then updates weights to predict
      the next spike pattern.

    Usage
    -----
    >>> replay = ReplayEngine(population, weight_matrix, stdp_engine)
    >>> result = replay.run(spike_data, n_epochs=5)
    """

    def __init__(self, population, weight_matrix, stdp_engine):
        self.pop   = population
        self.W     = weight_matrix
        self.stdp  = stdp_engine

    def run(
        self,
        data:             SpikeData,
        n_epochs:         int    = 1,
        base_current:     float  = 2.0,
        replay_gain:      float  = 5.0,   # extra current for recorded spikes
        verbose:          bool   = True,
    ) -> Dict:
        """
        Replay spike data through the SNN with STDP learning.

        At each timestep:
          1. Convert recorded spikes → extra injected current
          2. Step the LIF population
          3. Apply STDP based on (recorded, simulated) spike pairs
        """
        p       = self.pop.p
        dt      = p.dt
        t_min   = data.times.min() if len(data.times) else 0.0
        t_max   = data.times.max() if len(data.times) else 100.0
        n_steps = int((t_max - t_min) / dt) + 1
        N       = self.pop.N

        results = {
            "epoch_losses": [],
            "weight_mean_history": [],
            "firing_rate_history": [],
        }

        for epoch in range(n_epochs):
            self.pop.reset()
            self.stdp.reset_traces()

            total_loss = 0.0
            n_batches  = 0

            # Build spike time lookup: step_idx → neuron array
            step_lookup = self._build_step_lookup(data, t_min, dt, n_steps, N)

            for step_idx in range(n_steps):
                t = t_min + step_idx * dt

                # Recorded pattern → extra current
                recorded = step_lookup[step_idx]   # bool (N,)
                I_extra  = recorded.astype(np.float64) * replay_gain

                # Synaptic current
                I_syn  = self.W.compute_current(self.pop.spikes.astype(np.float32))
                I_total = base_current + I_extra + I_syn

                # LIF step
                sim_spikes = self.pop.step(t, I_total)

                # STDP: teach (recorded → sim) correlations
                self.stdp.step(
                    spikes_pre=recorded,
                    spikes_post=sim_spikes,
                    firing_rates=self.pop.firing_rate_estimate,
                )

                # Loss: Hamming distance between recorded and simulated
                loss = float(np.mean(recorded != sim_spikes))
                total_loss += loss
                n_batches  += 1

            mean_loss = total_loss / max(n_batches, 1)
            results["epoch_losses"].append(mean_loss)
            results["weight_mean_history"].append(self.W.mean_weight)
            results["firing_rate_history"].append(self.pop.mean_firing_rate())

            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs}  loss={mean_loss:.4f}  "
                      f"<W>={self.W.mean_weight:.4f}  "
                      f"<FR>={self.pop.mean_firing_rate():.2f} Hz")

        return results

    @staticmethod
    def _build_step_lookup(
        data:    SpikeData,
        t_min:   float,
        dt:      float,
        n_steps: int,
        N:       int,
    ) -> list[np.ndarray]:
        """Pre-build boolean spike matrices for fast replay."""
        lookup = [np.zeros(N, dtype=np.bool_) for _ in range(n_steps)]
        for t, nid in zip(data.times, data.neurons):
            step = int(round((t - t_min) / dt))
            if 0 <= step < n_steps and 0 <= nid < N:
                lookup[step][nid] = True
        return lookup
