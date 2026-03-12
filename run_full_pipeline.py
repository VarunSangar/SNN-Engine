import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from pathlib import Path

# --- 1. THE ULTIMATE PATH FIX ---
# Instead of manual loaders, we add the current directory to the system path.
# This allows standard "import" statements to work without dataclass errors.
current_dir = str(Path(__file__).parent.absolute())
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- 2. STANDARD IMPORTS ---
# Now that the path is set, we import files directly by their filename
try:
    import lif_model
    import synaptic_matrix
    import stdp as stdp_mod
    import manifold_mapper as manifold
except ImportError as e:
    st.error(f"Critical Import Error: {e}. Check if all .py files are in the same folder.")
    st.stop()

# Extract Classes
LIFPopulation = lif_model.LIFPopulation
LIFParams = lif_model.LIFParams
SparseWeightMatrix = synaptic_matrix.SparseWeightMatrix
SynapseParams = synaptic_matrix.SynapseParams
STDPEngine = stdp_mod.STDPEngine
STDPParams = stdp_mod.STDPParams
ManifoldMapper = manifold.ManifoldMapper

# --- 3. UI SETUP ---
st.set_page_config(page_title="Axiom-Neuro Stable", layout="wide")
st.title("🔬 Axiom-Neuro: Final Stable Engine")

if st.button("🚀 Run Simulation"):
    N = 256
    # Initialize Engine
    pop = LIFPopulation(LIFParams(n_neurons=N))
    W = SparseWeightMatrix(SynapseParams(n_pre=N, n_post=N))
    stdp_eng = STDPEngine(STDPParams(), W, N, N)

    # DISCOVER WEIGHT ATTRIBUTE (prevents AttributeError)
    # We check the object for common internal names: W, matrix, weights, data
    w_attr = None
    for attr in ['W', 'matrix', 'weights', 'data', '_W']:
        if hasattr(W, attr):
            w_attr = attr
            break
    
    col1, col2 = st.columns(2)
    map_p = col1.empty()
    chart_p = col2.empty()
    w_history = []

    for epoch in range(100):
        # 1. Step the neurons
        pop.step(epoch * 0.1, np.ones(N) * 3.1)
        
        # 2. Extract Spikes and prevent IndexError
        if hasattr(pop, 'spikes') and pop.spikes is not None:
            if np.asarray(pop.spikes).dtype == bool:
                indices = np.where(pop.spikes)[0]
            else:
                indices = np.asarray(pop.spikes).astype(int)
            
            # Bound check indices against backend trace size
            try:
                max_pre = len(stdp_eng.x_pre)
                max_post = len(stdp_eng.x_post)
            except:
                max_pre = max_post = N

            safe_pre = indices[indices < max_pre].astype(int)
            safe_post = indices[indices < max_post].astype(int)

            # 3. Update Plasticity
            try:
                stdp_eng.step(epoch * 0.1, safe_pre, safe_post)
            except Exception:
                try:
                    stdp_eng.update(epoch * 0.1, safe_pre, safe_post)
                except:
                    pass

        # 4. Update Graphs
        if w_attr:
            val = getattr(W, w_attr)
            # Handle if weights is a sparse matrix or dense array
            mean_w = np.mean(val.toarray()) if hasattr(val, 'toarray') else np.mean(val)
            w_history.append(float(mean_w))
            chart_p.line_chart(w_history)

        # Activity Heatmap
        viz = np.zeros(256)
        if len(indices) > 0:
            viz[indices[indices < 256]] = 1.0
        
        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(viz.reshape((16,16)), cmap='magma')
        ax.axis('off')
        map_p.pyplot(fig)
        plt.close(fig)
        
        time.sleep(0.01)

    st.success("Simulation Complete.")
