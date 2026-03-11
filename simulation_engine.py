"""
axiom_neuro/core/simulation_engine.py
========================================
Axiom-Neuro Simulation Engine
================================

Top-level orchestration class that wires together:
  - LIFPopulation        (membrane dynamics)
  - SparseWeightMatrix   (synaptic connectivity)
  - STDPEngine           (learning)
  - ManifoldMapper       (geometric feature)
  - Visualization suite  (raster, dashboard)

Usage
-----
>>> from axiom_neuro.core.simulation_engine import SimulationEngine, SimConfig
>>> cfg = SimConfig(n_neurons=1000, duration_ms=500.0, dt=0.1)
>>> sim = SimulationEngine(cfg)
>>> result = sim.run()
>>> sim.save_results("run_001")
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from .lif_model        import LIFPopulation, LIFParams
from .synaptic_matrix  import SparseWeightMatrix, SynapseParams
from ..learning.stdp   import STDPEngine, STDPParams
from ..geometry.manifold_mapper import ManifoldMapper, NeuronEmbedding
from ..visualization.plotter   import NetworkDashboard, RasterPlot, save_manifold_timelapse


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """
    Unified simulation configuration.
    """
    # Population
    n_neurons:          int   = 1000
    duration_ms:        float = 500.0
    dt:                 float = 0.1       # ms
    seed:               int   = 42

    # LIF params
    tau_m:              float = 20.0
    V_rest:             float = -65.0
    V_thresh:           float = -50.0
    V_reset:            float = -70.0
    R_m:                float = 10.0
    t_refrac:           float = 2.0
    noise_sigma:        float = 0.5

    # Synaptic connectivity
    conn_density:       float = 0.05
    w_init_mean:        float = 0.5
    w_init_std:         float = 0.2
    exc_fraction:       float = 0.8

    # Input current
    base_current:       float = 2.0      # nA mean
    current_noise:      float = 1.0      # nA std

    # STDP
    stdp_enabled:       bool  = True
    A_plus:             float = 0.01
    A_minus:            float = 0.0105
    tau_plus:           float = 20.0
    tau_minus:          float = 20.0
    homeostasis:        bool  = True
    r_target:           float = 5.0      # Hz

    # Geometry
    manifold_enabled:   bool  = True
    embedding_strategy: str   = 'random_sphere'
    manifold_min_fire:  int   = 10

    # Output
    record_interval:    int   = 10       # record manifold every N steps
    save_raster:        bool  = True
    save_dashboard:     bool  = True
    save_manifold_gif:  bool  = False
    output_dir:         str   = "outputs"

    @property
    def n_steps(self) -> int:
        return int(self.duration_ms / self.dt)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Simulation Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    """All output data from a completed simulation run."""
    config:          SimConfig
    spike_times:     np.ndarray          # (S,)  ms
    spike_neurons:   np.ndarray          # (S,)  neuron ids
    vol_times:       np.ndarray          # (T,)  ms
    volumes:         np.ndarray          # (T,)  manifold volumes
    areas:           np.ndarray          # (T,)  manifold surface areas
    firing_rates:    np.ndarray          # (N,)  Hz per neuron
    weight_data:     Optional[np.ndarray]= None   # nonzero weights
    wall_time_s:     float               = 0.0
    n_synaptic_events: int               = 0
    events_per_second: float             = 0.0
    metadata:        Dict[str, Any]      = field(default_factory=dict)

    def summary(self) -> str:
        cfg = self.config
        lines = [
            "=" * 60,
            "  AXIOM-NEURO Simulation Summary",
            "=" * 60,
            f"  Neurons        : {cfg.n_neurons:,}",
            f"  Duration       : {cfg.duration_ms:.0f} ms",
            f"  dt             : {cfg.dt} ms  ({cfg.n_steps:,} steps)",
            f"  Total spikes   : {len(self.spike_times):,}",
            f"  Mean FR        : {self._mean_fr():.2f} Hz",
            f"  Syn. events    : {self.n_synaptic_events:,}",
            f"  Events/sec     : {self.events_per_second:,.0f}",
            f"  Wall time      : {self.wall_time_s:.2f} s",
            f"  Sim. speed     : {cfg.duration_ms / self.wall_time_s:.1f}× real-time",
            "=" * 60,
        ]
        return "\n".join(lines)

    def _mean_fr(self) -> float:
        if len(self.spike_times) == 0:
            return 0.0
        dur = self.config.duration_ms * 1e-3
        return len(self.spike_times) / (self.config.n_neurons * dur)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Simulation Engine
# ─────────────────────────────────────────────────────────────────────────────

class SimulationEngine:
    """
    Axiom-Neuro main simulation orchestrator.

    Wires LIF dynamics, sparse synaptic matrix, STDP learning, and the
    Minkowski-Convex-Hull manifold mapper into a single high-performance loop.

    Usage
    -----
    >>> cfg = SimConfig(n_neurons=5000, duration_ms=200.0, stdp_enabled=True)
    >>> sim = SimulationEngine(cfg)
    >>> result = sim.run(verbose=True)
    >>> result.summary()
    """

    def __init__(self, config: SimConfig):
        self.cfg = config
        cfg = config

        # ── LIF Population ─────────────────────────────────────────────────
        lif_params = LIFParams(
            tau_m       = cfg.tau_m,
            V_rest      = cfg.V_rest,
            V_thresh    = cfg.V_thresh,
            V_reset     = cfg.V_reset,
            R_m         = cfg.R_m,
            t_refrac    = cfg.t_refrac,
            dt          = cfg.dt,
            n_neurons   = cfg.n_neurons,
            noise_sigma = cfg.noise_sigma,
        )
        self.population = LIFPopulation(lif_params, seed=cfg.seed)

        # ── Synaptic Matrix ────────────────────────────────────────────────
        syn_params = SynapseParams(
            n_pre         = cfg.n_neurons,
            n_post        = cfg.n_neurons,
            density       = cfg.conn_density,
            w_init_mean   = cfg.w_init_mean,
            w_init_std    = cfg.w_init_std,
            exc_fraction  = cfg.exc_fraction,
        )
        self.W = SparseWeightMatrix(syn_params, seed=cfg.seed)

        # ── STDP ───────────────────────────────────────────────────────────
        if cfg.stdp_enabled:
            stdp_params = STDPParams(
                A_plus               = cfg.A_plus,
                A_minus              = cfg.A_minus,
                tau_plus             = cfg.tau_plus,
                tau_minus            = cfg.tau_minus,
                homeostasis_enabled  = cfg.homeostasis,
                r_target             = cfg.r_target,
            )
            self.stdp = STDPEngine(stdp_params, self.W, cfg.n_neurons, cfg.n_neurons, cfg.dt)
        else:
            self.stdp = None

        # ── Geometry ───────────────────────────────────────────────────────
        if cfg.manifold_enabled:
            self.embedding = NeuronEmbedding(cfg.n_neurons, cfg.embedding_strategy, cfg.seed)
            self.mapper    = ManifoldMapper(self.embedding, min_firing=cfg.manifold_min_fire)
        else:
            self.embedding = None
            self.mapper    = None

        # ── Input current (per-neuron baseline, heterogeneous) ─────────────
        rng = np.random.default_rng(cfg.seed + 1)
        self._I_base = rng.normal(cfg.base_current, cfg.current_noise, cfg.n_neurons)
        self._I_base = np.clip(self._I_base, 0.0, None)

        print(f"[AxiomNeuro] Engine initialized")
        print(f"  Population : {cfg.n_neurons:,} neurons")
        print(f"  Synapses   : {self.W.n_synapses:,}  ({self.W.density_actual():.4f} density)")
        print(f"  Memory W   : {self.W.memory_bytes/1024:.1f} KB (sparse CSR)")
        print(f"  STDP       : {'ON' if self.stdp else 'OFF'}")
        print(f"  Manifold   : {'ON' if self.mapper else 'OFF'}")

    # ── Main run loop ──────────────────────────────────────────────────────────

    def run(
        self,
        verbose:          bool     = True,
        progress_every:   int      = 1000,
        step_callback:    Optional[Callable] = None,
    ) -> SimResult:
        """
        Execute the full simulation.

        Parameters
        ----------
        verbose         : print progress
        progress_every  : log every N steps
        step_callback   : optional callable(step, t, spikes) called each step

        Returns
        -------
        SimResult
        """
        cfg   = self.cfg
        pop   = self.population
        W     = self.W
        stdp  = self.stdp
        mapper = self.mapper

        n_steps          = cfg.n_steps
        n_synaptic_events = 0
        t_wall_start     = time.perf_counter()

        if verbose:
            print(f"\n[AxiomNeuro] Running {n_steps:,} steps × {cfg.n_neurons:,} neurons...")

        for step in range(n_steps):
            t = step * cfg.dt

            # ── Synaptic current ──────────────────────────────────────────
            I_syn   = W.compute_current(pop.spikes.astype(np.float32))
            I_total = self._I_base + I_syn

            # ── LIF step ──────────────────────────────────────────────────
            spikes = pop.step(t, I_total)

            n_fired = int(spikes.sum())
            n_synaptic_events += n_fired * W.n_synapses // cfg.n_neurons

            # ── STDP ──────────────────────────────────────────────────────
            if stdp is not None:
                stdp.step(
                    spikes_pre  = spikes,
                    spikes_post = spikes,
                    firing_rates = pop.firing_rate_estimate,
                )

            # ── Manifold ──────────────────────────────────────────────────
            if mapper is not None and step % cfg.record_interval == 0:
                fired_idx = np.where(spikes)[0]
                mapper.update(t, fired_idx)

            # ── Callback ──────────────────────────────────────────────────
            if step_callback is not None:
                step_callback(step, t, spikes)

            # ── Progress ──────────────────────────────────────────────────
            if verbose and step % progress_every == 0:
                pct      = 100.0 * step / n_steps
                elapsed  = time.perf_counter() - t_wall_start
                eta      = (elapsed / (step+1)) * (n_steps - step) if step > 0 else 0
                fr_mean  = pop.mean_firing_rate()
                print(f"  [{pct:5.1f}%]  t={t:.1f}ms  "
                      f"spikes={n_fired:4d}  <FR>={fr_mean:.1f}Hz  "
                      f"ETA={eta:.0f}s")

        wall_time = time.perf_counter() - t_wall_start
        events_per_sec = n_synaptic_events / max(wall_time, 1e-12)

        # ── Collect results ────────────────────────────────────────────────
        spike_times, spike_neurons = pop.get_raster()

        vol_times = areas = volumes = np.array([])
        if mapper:
            vol_times, volumes = mapper.get_volume_trace()
            _,          areas  = mapper.get_area_trace()

        result = SimResult(
            config            = cfg,
            spike_times       = spike_times,
            spike_neurons     = spike_neurons,
            vol_times         = vol_times,
            volumes           = volumes,
            areas             = areas,
            firing_rates      = pop.firing_rate_estimate.copy(),
            weight_data       = W._W_csr.data.copy() if W._W_csr.nnz > 0 else None,
            wall_time_s       = wall_time,
            n_synaptic_events = n_synaptic_events,
            events_per_second = events_per_sec,
        )

        if verbose:
            print(result.summary())

        return result

    # ── Save results ──────────────────────────────────────────────────────────

    def save_results(
        self,
        run_name: str,
        result:   Optional[SimResult] = None,
    ) -> None:
        """Save raster, dashboard, and manifold animation."""
        cfg    = self.cfg
        outdir = Path(cfg.output_dir) / run_name
        outdir.mkdir(parents=True, exist_ok=True)

        if result is None:
            print("No result to save (run sim.run() first).")
            return

        # ── Raster ────────────────────────────────────────────────────────
        if cfg.save_raster and len(result.spike_times) > 0:
            rp  = RasterPlot(cfg.n_neurons, t_window=cfg.duration_ms)
            fig = rp.plot(result.spike_times, result.spike_neurons,
                          title=f"Axiom-Neuro  {run_name}")
            rp.save_png(fig, str(outdir / "raster.png"))
            print(f"  Saved raster → {outdir}/raster.png")

        # ── Dashboard ─────────────────────────────────────────────────────
        if cfg.save_dashboard:
            dash = NetworkDashboard()
            fig  = dash.render(
                times=result.spike_times,
                neurons=result.spike_neurons,
                vol_times=result.vol_times,
                volumes=result.volumes,
                areas=result.areas,
                weight_data=result.weight_data,
                firing_rates=result.firing_rates,
                title=f"Axiom-Neuro  |  {run_name}",
            )
            fig.savefig(str(outdir / "dashboard.png"), dpi=150,
                        bbox_inches='tight', facecolor="#0d0f14")
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"  Saved dashboard → {outdir}/dashboard.png")

        # ── Manifold GIF ──────────────────────────────────────────────────
        if cfg.save_manifold_gif and self.mapper:
            save_manifold_timelapse(
                self.mapper,
                str(outdir / "manifold.gif"),
                n_frames=40,
            )

        # ── Spike CSV ─────────────────────────────────────────────────────
        import pandas as pd
        df = pd.DataFrame({
            "time_ms":   result.spike_times,
            "neuron_id": result.spike_neurons,
        })
        df.to_csv(str(outdir / "spikes.csv"), index=False)
        print(f"  Saved spikes CSV → {outdir}/spikes.csv")

        # ── Manifold metrics ──────────────────────────────────────────────
        if len(result.vol_times) > 0:
            dfm = pd.DataFrame({
                "time_ms": result.vol_times,
                "volume":  result.volumes,
                "area":    result.areas,
            })
            dfm.to_csv(str(outdir / "manifold_metrics.csv"), index=False)
            print(f"  Saved manifold metrics → {outdir}/manifold_metrics.csv")
