"""
axiom_neuro/examples/run_full_pipeline.py
==========================================
Full Axiom-Neuro Pipeline Demonstration: Final Stable Build
=========================================
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless mode for server/cloud stability
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. DYNAMIC PATH RESOLUTION ---
# Ensures the script can find 'axiom_neuro' regardless of where it's launched
current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# --- 2. SECURE IMPORTS ---
try:
    from axiom_neuro.core.simulation_engine import SimulationEngine, SimConfig
    from axiom_neuro.io.data_loader import SyntheticDataGenerator, SpikeDataLoader, ReplayEngine
    from axiom_neuro.core.lif_model import LIFPopulation, LIFParams
    from axiom_neuro.core.synaptic_matrix import SparseWeightMatrix, SynapseParams
    from axiom_neuro.learning.stdp import STDPEngine, STDPParams
    from axiom_neuro.geometry.manifold_mapper import NeuronEmbedding, ManifoldMapper
except ImportError as e:
    print(f"CRITICAL: Module loading failed. Ensure folder structure is correct. Error: {e}")
    sys.exit(1)

def example_1_basic_simulation():
    """Run a basic LIF network and produce raster + dashboard."""
    print("\n" + "="*60)
    print("  Example 1: Basic LIF Simulation with STDP")
    print("="*60)

    cfg = SimConfig(
        n_neurons         = 500,
        duration_ms       = 200.0,
        dt                = 0.1,
        conn_density      = 0.08,
        base_current      = 2.2,
        current_noise     = 0.8,
        stdp_enabled      = True,
        homeostasis       = True,
        r_target          = 8.0,
        manifold_enabled  = True,
        record_interval   = 5,
        save_raster       = True,
        save_dashboard    = True,
        output_dir        = "outputs",
    )

    sim = SimulationEngine(cfg)
    result = sim.run(verbose=True, progress_every=500)

    if sim.mapper:
        t_v, vols = sim.mapper.get_volume_trace()
        if len(vols) > 0:
            print(f"Volume range       : [{vols.min():.4f}, {vols.max():.4f}]")

        # Minkowski Sum of last two manifolds
        try:
            mink_verts = sim.mapper.minkowski_sum_consecutive()
            if mink_verts is not None:
                print(f"Minkowski Sum verts: {len(mink_verts)}")
        except Exception:
            print("Minkowski Sum: Insufficient geometry data.")

    sim.save_results("example_1", result)
    return result

def example_2_synthetic_replay():
    """Generate oscillatory spike data and replay through SNN."""
    print("\n" + "="*60)
    print("  Example 2: Synthetic Data Replay + Weight Learning")
    print("="*60)

    gen = SyntheticDataGenerator()
    data = gen.oscillatory_spikes(n_neurons=200, duration_ms=100.0, freq_hz=40.0)
    
    loader = SpikeDataLoader()
    Path("outputs").mkdir(exist_ok=True)
    csv_path = "outputs/synthetic_spikes.csv"
    loader.save_csv(data, csv_path)
    data2 = loader.load_csv(csv_path, n_neurons=200)

    # Build SNN Replay Architecture
    lp = LIFParams(n_neurons=200, dt=0.1)
    sp = SynapseParams(n_pre=200, n_post=200, density=0.1)
    pop = LIFPopulation(lp)
    W = SparseWeightMatrix(sp)
    stdp = STDPEngine(STDPParams(homeostasis_enabled=True), W, 200, 200)

    engine = ReplayEngine(pop, W, stdp)
    res = engine.run(data2, n_epochs=3, base_current=1.5)

    print(f"Replay complete. Final Mean Weight: {np.mean(W.weights):.4f}")
    return res

def example_3_manifold_analysis():
    """Information Geometry Deep-Dive."""
    print("\n" + "="*60)
    print("  Example 3: Information Manifold Geometry")
    print("="*60)

    N = 300
    lp = LIFParams(n_neurons=N)
    sp = SynapseParams(n_pre=N, n_post=N)
    pop, W = LIFPopulation(lp), SparseWeightMatrix(sp)
    
    emb = NeuronEmbedding(N, 'toroidal')
    mapper = ManifoldMapper(emb, history_len=1000)

    # Simulate and update manifold
    for step in range(500):
        t = step * 0.1
        spikes = pop.step(t, np.random.normal(3.0, 0.5, N))
        if step % 5 == 0:
            mapper.update(t, np.where(spikes)[0])

    t_v, vols = mapper.get_volume_trace()
    if len(vols) > 1:
        plt.figure(figsize=(10, 4), facecolor="#0d0f14")
        plt.plot(t_v, vols, color="#f77b4d")
        plt.title("Neural Manifold Volume Expansion", color="white")
        plt.savefig("outputs/manifold_geometry.png", facecolor="#0d0f14")
        print("Saved outputs/manifold_geometry.png")

def example_4_membrane_potential_trace():
    """Visualise V(t) for a population subset."""
    print("\n" + "="*60)
    print("  Example 4: Membrane Potential Traces")
    print("="*60)

    N = 10
    lp = LIFParams(n_neurons=N, dt=0.05)
    pop = LIFPopulation(lp)

    n_steps = 1000
    V_trace = np.zeros((N, n_steps))
    for step in range(n_steps):
        pop.step(step * 0.05, np.ones(N) * 2.5)
        V_trace[:, step] = pop.V

    plt.figure(figsize=(12, 6), facecolor="#0d0f14")
    for i in range(5):
        plt.plot(V_trace[i], label=f"Neuron {i}", alpha=0.8)
    plt.axhline(lp.V_thresh, color='red', linestyle='--', alpha=0.3)
    plt.legend()
    plt.savefig("outputs/membrane_potential.png", facecolor="#0d0f14")
    print("Saved outputs/membrane_potential.png")

if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    try:
        example_4_membrane_potential_trace()
        example_1_basic_simulation()
        example_2_synthetic_replay()
        example_3_manifold_analysis()
        print("\nPIPELINE SUCCESS: All modules verified.")
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
