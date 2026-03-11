"""
axiom_neuro/benchmarks/stress_test.py
========================================
Axiom-Neuro Stress Test & Performance Benchmarks
==================================================

Proves the engine can sustain ≥10^7 synaptic events per second.

Tests:
  1. LIF kernel throughput  (neurons/sec)
  2. Sparse matrix-vector throughput  (synops/sec)
  3. Full simulation throughput  (steps/sec, events/sec)
  4. STDP overhead
  5. Manifold computation overhead
  6. Memory footprint at 50,000 neurons

Run from repo root:
    python -m axiom_neuro.benchmarks.stress_test
"""

import time
import sys
import numpy as np
from pathlib import Path

# Add parent to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from axiom_neuro.core.lif_model       import LIFPopulation, LIFParams
from axiom_neuro.core.synaptic_matrix  import SparseWeightMatrix, SynapseParams
from axiom_neuro.learning.stdp         import STDPEngine, STDPParams
from axiom_neuro.geometry.manifold_mapper import NeuronEmbedding, ManifoldMapper
from axiom_neuro.core.simulation_engine  import SimulationEngine, SimConfig


# ── ANSI colours ──────────────────────────────────────────────────────────────
GRN = "\033[92m"
YLW = "\033[93m"
BLU = "\033[94m"
RED = "\033[91m"
RST = "\033[0m"
BLD = "\033[1m"

def hr(char="═", n=64): return char * n
def ok(msg): print(f"  {GRN}✓{RST}  {msg}")
def warn(msg): print(f"  {YLW}⚠{RST}  {msg}")
def hdr(msg): print(f"\n{BLD}{BLU}{hr()}{RST}\n{BLD}  {msg}{RST}\n{BLD}{BLU}{hr()}{RST}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LIF Kernel Throughput
# ─────────────────────────────────────────────────────────────────────────────

def bench_lif_kernel(n_neurons: int = 50_000, n_steps: int = 1000) -> dict:
    hdr(f"Bench 1: LIF Kernel — {n_neurons:,} neurons × {n_steps:,} steps")

    params = LIFParams(n_neurons=n_neurons, dt=0.1)
    pop    = LIFPopulation(params, seed=0)
    I_ext  = np.ones(n_neurons, dtype=np.float64) * 2.5

    # Warmup (JIT compile if Numba available)
    print("  Warming up JIT...")
    for _ in range(10):
        pop.step(0.0, I_ext)
    pop.reset()

    # Timed run
    t0 = time.perf_counter()
    for step in range(n_steps):
        pop.step(step * 0.1, I_ext)
    dt_wall = time.perf_counter() - t0

    neurons_per_sec = n_neurons * n_steps / dt_wall
    steps_per_sec   = n_steps / dt_wall
    sim_speed       = (n_steps * 0.1) / (dt_wall * 1000)  # simulated ms / wall ms

    ok(f"Wall time      : {dt_wall:.3f} s")
    ok(f"Steps/sec      : {steps_per_sec:,.0f}")
    ok(f"Neurons×steps/s: {neurons_per_sec:,.0f}")
    ok(f"Sim speed      : {sim_speed:.1f}× real-time (ms_sim/ms_wall)")
    ok(f"Mean FR        : {pop.mean_firing_rate():.2f} Hz")

    return {"neurons_per_sec": neurons_per_sec, "steps_per_sec": steps_per_sec}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Sparse Matrix-Vector (Synaptic Current)
# ─────────────────────────────────────────────────────────────────────────────

def bench_sparse_matvec(n_neurons: int = 50_000, density: float = 0.002, n_reps: int = 5000) -> dict:
    hdr(f"Bench 2: Sparse MatVec — {n_neurons:,}² × ρ={density}")

    params = SynapseParams(n_pre=n_neurons, n_post=n_neurons, density=density)
    W      = SparseWeightMatrix(params)

    print(f"  Synapses     : {W.n_synapses:,}")
    print(f"  Memory (CSR) : {W.memory_bytes/1024**2:.1f} MB")

    # Typical 1% firing rate
    spikes = np.zeros(n_neurons, dtype=np.float32)
    fired  = np.random.choice(n_neurons, size=int(0.01*n_neurons), replace=False)
    spikes[fired] = 1.0

    # Warmup
    for _ in range(20):
        _ = W.compute_current(spikes)

    t0 = time.perf_counter()
    for _ in range(n_reps):
        I_syn = W.compute_current(spikes)
    dt_wall = time.perf_counter() - t0

    synops_per_rep = W.n_synapses * float(spikes.mean())
    synops_per_sec = synops_per_rep * n_reps / dt_wall

    ok(f"Wall time      : {dt_wall:.3f} s  ({n_reps:,} reps)")
    ok(f"Synaptic ops/s : {synops_per_sec:,.0f}")
    ok(f"MegaSynOps/sec : {synops_per_sec/1e6:.2f}")

    target = 1e7
    if synops_per_sec >= target:
        ok(f"{GRN}TARGET ACHIEVED{RST}: {synops_per_sec:.2e} ≥ {target:.0e} syn/sec")
    else:
        warn(f"Below target: {synops_per_sec:.2e} < {target:.0e} (higher density needed)")

    return {"synops_per_sec": synops_per_sec}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Full Simulation Throughput
# ─────────────────────────────────────────────────────────────────────────────

def bench_full_simulation(n_neurons: int = 10_000, duration_ms: float = 100.0) -> dict:
    hdr(f"Bench 3: Full Simulation — {n_neurons:,} neurons × {duration_ms:.0f} ms")

    cfg = SimConfig(
        n_neurons       = n_neurons,
        duration_ms     = duration_ms,
        dt              = 0.1,
        conn_density    = 0.01,
        stdp_enabled    = True,
        manifold_enabled= True,
        record_interval = 10,
        save_raster     = False,
        save_dashboard  = False,
    )

    sim = SimulationEngine(cfg)

    t0     = time.perf_counter()
    result = sim.run(verbose=False)
    t_wall = time.perf_counter() - t0

    ok(f"Wall time         : {t_wall:.3f} s")
    ok(f"Sim speed         : {duration_ms/t_wall:.1f}× real-time")
    ok(f"Total spikes      : {len(result.spike_times):,}")
    ok(f"Syn events        : {result.n_synaptic_events:,}")
    ok(f"Events/sec        : {result.events_per_second:,.0f}")
    ok(f"Mean firing rate  : {result._mean_fr():.2f} Hz")

    if result.events_per_second >= 1e7:
        ok(f"{GRN}🚀 TARGET ACHIEVED: {result.events_per_second:.2e} syn-events/sec ≥ 10^7{RST}")
    else:
        warn(f"Events/sec: {result.events_per_second:.2e}  (scale n_neurons to hit 10^7)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Manifold Computation Overhead
# ─────────────────────────────────────────────────────────────────────────────

def bench_manifold(n_neurons: int = 10_000, n_steps: int = 500, firing_frac: float = 0.05) -> dict:
    hdr(f"Bench 4: Manifold Mapper — {n_neurons:,} neurons, f={firing_frac}")

    emb    = NeuronEmbedding(n_neurons, 'random_sphere', seed=0)
    mapper = ManifoldMapper(emb, min_firing=10)
    rng    = np.random.default_rng(0)

    n_fire = max(10, int(n_neurons * firing_frac))
    times  = []

    for step in range(n_steps):
        idx = rng.choice(n_neurons, size=n_fire, replace=False)
        t0  = time.perf_counter()
        snap = mapper.update(float(step), idx)
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms

    ok(f"Mean hull compute : {times.mean():.3f} ms")
    ok(f"Max  hull compute : {times.max():.3f} ms")
    ok(f"Overhead fraction : {times.mean()/(0.1 + times.mean())*100:.1f}% of 0.1ms step")

    valid = sum(1 for s in mapper.history if s.valid)
    ok(f"Valid snapshots   : {valid}/{n_steps}")

    if valid > 0:
        t_v, vols = mapper.get_volume_trace()
        ok(f"Mean manifold vol : {vols.mean():.4f}")

    return {"mean_ms": float(times.mean()), "max_ms": float(times.max())}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Memory Footprint
# ─────────────────────────────────────────────────────────────────────────────

def bench_memory(n_neurons: int = 50_000, density: float = 0.002) -> dict:
    hdr(f"Bench 5: Memory Footprint — N={n_neurons:,}")

    params = LIFParams(n_neurons=n_neurons)
    pop    = LIFPopulation(params)

    syn_params = SynapseParams(n_pre=n_neurons, n_post=n_neurons, density=density)
    W     = SparseWeightMatrix(syn_params)

    emb   = NeuronEmbedding(n_neurons, 'random_sphere')

    # Memory estimates
    lif_mem   = (pop.V.nbytes + pop.refrac_count.nbytes + pop.spikes.nbytes) / 1024
    w_mem     = W.memory_bytes / 1024 / 1024
    emb_mem   = emb.coords.nbytes / 1024

    dense_mem = n_neurons**2 * 4 / 1024 / 1024 / 1024  # GB if dense float32

    ok(f"LIF state arrays  : {lif_mem:.1f} KB")
    ok(f"Weight matrix CSR : {W.memory_bytes/1024:.1f} KB  ({W.n_synapses:,} synapses)")
    ok(f"Embedding coords  : {emb_mem:.1f} KB")
    ok(f"Dense W would be  : {dense_mem:.1f} GB  → {dense_mem/(w_mem/1024):.0f}× larger!")
    ok(f"Sparsity saving   : {(1 - density)*100:.1f}%")

    return {"w_sparse_kb": W.memory_bytes/1024, "w_dense_gb": dense_mem}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  STDP Update Speed
# ─────────────────────────────────────────────────────────────────────────────

def bench_stdp(n_neurons: int = 5_000, n_steps: int = 2000) -> dict:
    hdr(f"Bench 6: STDP Speed — {n_neurons:,} neurons × {n_steps:,} steps")

    syn_params  = SynapseParams(n_pre=n_neurons, n_post=n_neurons, density=0.02)
    W           = SparseWeightMatrix(syn_params)
    stdp_params = STDPParams(homeostasis_enabled=False)
    stdp        = STDPEngine(stdp_params, W, n_neurons, n_neurons, dt=0.1)

    rng        = np.random.default_rng(0)
    fire_prob  = 0.01
    times_list = []

    for _ in range(n_steps):
        sp = rng.random(n_neurons) < fire_prob
        t0 = time.perf_counter()
        stdp.step(sp, sp)
        times_list.append(time.perf_counter() - t0)

    times = np.array(times_list) * 1000  # ms
    ok(f"Mean STDP step    : {times.mean():.3f} ms")
    ok(f"STDP steps/sec    : {1000/times.mean():,.0f}")
    ok(f"Mean |ΔW|         : {np.mean(stdp.dW_history) if stdp.dW_history else 0:.6f}")

    return {"mean_ms": float(times.mean())}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"""
{BLD}{BLU}
  ╔══════════════════════════════════════════════════════════════╗
  ║         AXIOM-NEURO  —  Performance Stress Test             ║
  ║         Target: ≥10^7 Synaptic Events / Second              ║
  ╚══════════════════════════════════════════════════════════════╝
{RST}""")

    results = {}

    try:
        results["lif"] = bench_lif_kernel(n_neurons=50_000, n_steps=500)
    except Exception as e:
        warn(f"LIF bench failed: {e}")

    try:
        results["sparse"] = bench_sparse_matvec(n_neurons=50_000, density=0.002, n_reps=2000)
    except Exception as e:
        warn(f"Sparse bench failed: {e}")

    try:
        results["full_sim"] = bench_full_simulation(n_neurons=10_000, duration_ms=100.0)
    except Exception as e:
        warn(f"Full sim bench failed: {e}")

    try:
        results["manifold"] = bench_manifold(n_neurons=5_000, n_steps=200)
    except Exception as e:
        warn(f"Manifold bench failed: {e}")

    try:
        results["memory"] = bench_memory(n_neurons=50_000, density=0.002)
    except Exception as e:
        warn(f"Memory bench failed: {e}")

    try:
        results["stdp"] = bench_stdp(n_neurons=3_000, n_steps=1000)
    except Exception as e:
        warn(f"STDP bench failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    hdr("BENCHMARK SUMMARY")

    if "lif" in results:
        ok(f"LIF throughput    : {results['lif']['neurons_per_sec']:>15,.0f} neuron-steps/sec")
    if "sparse" in results:
        r = results["sparse"]
        flag = GRN if r["synops_per_sec"] >= 1e7 else YLW
        ok(f"Syn. event rate   : {flag}{r['synops_per_sec']:>15,.0f}{RST} synops/sec  "
           f"{'✓ ≥10^7' if r['synops_per_sec']>=1e7 else '< 10^7'}")
    if "full_sim" in results:
        r = results["full_sim"]
        ok(f"Full sim speed    : {r.events_per_second:>15,.0f} events/sec")
        ok(f"Sim × real-time   : {r.config.duration_ms/r.wall_time_s:>15.1f}×")
    if "manifold" in results:
        ok(f"Hull compute time : {results['manifold']['mean_ms']:>14.3f} ms/step")
    if "memory" in results:
        ok(f"Sparse W memory   : {results['memory']['w_sparse_kb']:>14.1f} KB")
        ok(f"Dense W (avoided) : {results['memory']['w_dense_gb']:>14.2f} GB")

    print(f"""
{BLD}
  ┌─────────────────────────────────────────────────────────┐
  │  AXIOM-NEURO stress test complete.                      │
  │  Scale n_neurons and density proportionally to push     │
  │  throughput beyond 10^8 events/sec on modern hardware.  │
  └─────────────────────────────────────────────────────────┘
{RST}""")


if __name__ == "__main__":
    main()
