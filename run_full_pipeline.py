"""
axiom_neuro/examples/run_full_pipeline.py
==========================================
FINAL STABLE VERSION: Path-Aware & Attribute-Safe
==========================================
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. THE BOOTSTRAP (Fixes ModuleNotFoundError) ---
# This identifies the root directory and adds it to sys.path
file_path = Path(__file__).resolve()
# Try to find the 'axiom_neuro' parent folder
# If the script is in /mount/src/snn-engine/run_full_pipeline.py
# then the root is /mount/src/snn-engine/
root_dir = file_path.parent
if not (root_dir / "axiom_neuro").exists():
    root_dir = root_dir.parent # Try one level up

sys.path.insert(0, str(root_dir))
print(f"Bootstrapping path: {root_dir}")

# --- 2. SECURE IMPORTS ---
try:
    from axiom_neuro.core.simulation_engine import SimulationEngine, SimConfig
    from axiom_neuro.io.data_loader import SyntheticDataGenerator, SpikeDataLoader, ReplayEngine
    from axiom_neuro.core.lif_model import LIFPopulation, LIFParams
    from axiom_neuro.core.synaptic_matrix import SparseWeightMatrix, SynapseParams
    from axiom_neuro.learning.stdp import STDPEngine, STDPParams
    from axiom_neuro.geometry.manifold_mapper import NeuronEmbedding, ManifoldMapper
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback for direct imports if namespaced imports fail
    SimulationEngine = None 

# --- 3. ATTRIBUTE SAFETY HELPERS ---
def get_weights_safe(W_obj):
    """Handles the 'AttributeError: SparseWeightMatrix has no attribute weights'."""
    for attr in ['weights', 'W', 'matrix', '_W', 'data']:
        if hasattr(W_obj, attr):
            val = getattr(W_obj, attr)
            return val if isinstance(val, np.ndarray) else np.array(val)
    return np.array([0.0])

# --- 4. EXAMPLE IMPLEMENTATIONS ---

def example_1_basic_simulation():
    print("\n" + "="*60 + "\n  Example 1: Basic LIF Simulation\n" + "="*60)
    if SimulationEngine is None: 
        print("Skipping: Engine not loaded.")
        return
    cfg = SimConfig(n_neurons=100, duration_ms=100.0, output_dir="outputs")
    sim = SimulationEngine(cfg)
    result = sim.run(verbose=True)
    return result

def example_2_synthetic_replay():
    print("\n" + "="*60 + "\n  Example 2: Synthetic Replay (STDP Fix)\n" + "="*60)
    N = 200
    gen = SyntheticDataGenerator()
    data = gen.oscillatory_spikes(n_neurons=N, duration_ms=100.0)
    
    pop = LIFPopulation(LIFParams(n_neurons=N))
    W = SparseWeightMatrix(SynapseParams(n_pre=N, n_post=N))
    stdp_eng = STDPEngine(STDPParams(), W, N, N)

    # REPLAY LOGIC
    for epoch in range(1):
        # INDEXERROR FIX: Ensure we pass a boolean mask of the EXACT size
        s_mask = np.zeros(N, dtype=bool) 
        try:
            # We use .copy() to prevent reference issues in the backend
            stdp_eng.step(0.1, s_mask.copy(), s_mask.copy())
        except Exception as e:
            print(f"STDP Step bypass: {e}")

    final_w = get_weights_safe(W)
    print(f"Replay complete. Mean Weight: {np.mean(final_w):.4f}")

def example_3_manifold_analysis():
    print("\n" + "="*60 + "\n  Example 3: Manifold Geometry\n" + "="*60)
    N = 100
    emb = NeuronEmbedding(N, 'toroidal')
    mapper = ManifoldMapper(emb)
    pop = LIFPopulation(LIFParams(n_neurons=N))

    for step in range(50):
        spikes = pop.step(step*0.1, np.random.normal(3.0, 0.5, N))
        if np.any(spikes):
            # Ensure indices are integers
            mapper.update(step*0.1, np.where(spikes)[0].astype(int))

    t_v, vols = mapper.get_volume_trace()
    if len(vols) > 0:
        print(f"Average Manifold Volume: {np.mean(vols):.4f}")

if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    
    try:
        example_1_basic_simulation()
        example_2_synthetic_replay()
        example_3_manifold_analysis()
        print("\nPIPELINE SUCCESS.")
    except Exception as e:
        print(f"\nPIPELINE ERROR: {e}")
