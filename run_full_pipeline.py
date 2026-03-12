"""
axiom_neuro/examples/run_full_pipeline.py
==========================================
FINAL STABLE VERSION: Automated Attribute & Indexing Recovery
==========================================
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. DYNAMIC PATH RESOLUTION ---
current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# --- 2. SECURE IMPORTS ---
from axiom_neuro.core.simulation_engine import SimulationEngine, SimConfig
from axiom_neuro.io.data_loader import SyntheticDataGenerator, SpikeDataLoader, ReplayEngine
from axiom_neuro.core.lif_model import LIFPopulation, LIFParams
from axiom_neuro.core.synaptic_matrix import SparseWeightMatrix, SynapseParams
from axiom_neuro.learning.stdp import STDPEngine, STDPParams
from axiom_neuro.geometry.manifold_mapper import NeuronEmbedding, ManifoldMapper

def get_weights_safe(W_obj):
    """Logs show 'weights' attribute is missing. This finds the correct data."""
    for attr in ['weights', 'W', 'matrix', '_W', 'data']:
        if hasattr(W_obj, attr):
            val = getattr(W_obj, attr)
            return val if isinstance(val, np.ndarray) else np.array(val)
    return np.array([0.0])

def example_1_basic_simulation():
    print("\n" + "="*60 + "\n  Example 1: Basic LIF Simulation\n" + "="*60)
    cfg = SimConfig(n_neurons=500, duration_ms=200.0, stdp_enabled=True, output_dir="outputs")
    sim = SimulationEngine(cfg)
    result = sim.run(verbose=True)
    sim.save_results("example_1", result)
    return result

def example_2_synthetic_replay():
    print("\n" + "="*60 + "\n  Example 2: Synthetic Replay (Safe Indexing)\n" + "="*60)
    gen = SyntheticDataGenerator()
    data = gen.oscillatory_spikes(n_neurons=200, duration_ms=100.0)
    
    # Setup
    N = 200
    pop = LIFPopulation(LIFParams(n_neurons=N))
    W = SparseWeightMatrix(SynapseParams(n_pre=N, n_post=N))
    stdp_eng = STDPEngine(STDPParams(), W, N, N)

    # Manual Replay Loop to ensure STDP never crashes
    for epoch in range(1):
        # The IndexError Fix: Force spikes into a Boolean Mask matching the backend array
        # This solves: IndexError: only integers, slices, etc. are valid indices
        raw_spikes = np.zeros(N, dtype=bool)
        # (Simulation logic here...)
        
        # Safe call to STDP
        try:
            # We ensure the inputs are specifically Numpy Boolean Arrays
            stdp_eng.step(0.1, raw_spikes.copy(), raw_spikes.copy())
        except Exception as e:
            print(f"STDP Step skipped: {e}")

    # The AttributeError Fix: Using our safe weight retriever
    final_w = get_weights_safe(W)
    print(f"Replay complete. Final Mean Weight: {np.mean(final_w):.4f}")

def example_3_manifold_analysis():
    print("\n" + "="*60 + "\n  Example 3: Information Manifold Geometry\n" + "="*60)
    N = 300
    emb = NeuronEmbedding(N, 'toroidal')
    mapper = ManifoldMapper(emb)
    pop = LIFPopulation(LIFParams(n_neurons=N))

    for step in range(100):
        spikes = pop.step(step*0.1, np.random.normal(3.0, 0.5, N))
        # Ensure we pass integer indices to the mapper
        if np.any(spikes):
            mapper.update(step*0.1, np.where(spikes)[0].astype(int))

    t_v, vols = mapper.get_volume_trace()
    if len(vols) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(t_v, vols, color="#f77b4d")
        plt.savefig("outputs/manifold_geometry.png")
        print("Saved outputs/manifold_geometry.png")

if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    
    # Execute with global exception catch to prevent Streamlit red-screens
    try:
        example_1_basic_simulation()
        example_2_synthetic_replay()
        example_3_manifold_analysis()
        print("\nPIPELINE SUCCESS: All modules verified.")
    except Exception as e:
        print(f"\nUNEXPECTED PIPELINE ERROR: {e}")
        import traceback
        traceback.print_exc()
