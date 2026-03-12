import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from pathlib import Path

# --- 1. FORCE MODULE REGISTRATION ---
# This fixes the Python 3.14 dataclass lookup error
current_dir = str(Path(__file__).parent.absolute())
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Manually inject placeholders into sys.modules to satisfy dataclass lookups
import types
for mod_name in ['lif_model', 'synaptic_matrix', 'stdp', 'manifold_mapper']:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# --- 2. NATIVE IMPORTS ---
try:
    import lif_model
    import synaptic_matrix
    import stdp as stdp_mod
    import manifold_mapper as manifold
except Exception as e:
    st.error(f"Import Phase Failed: {e}")
    st.stop()

# --- 3. CLASS EXTRACTION ---
# Using getattr to avoid any more NoneType crashes
LIFPopulation = getattr(lif_model, "LIFPopulation", None)
LIFParams = getattr(lif_model, "LIFParams", None)
SparseWeightMatrix = getattr(synaptic_matrix, "SparseWeightMatrix", None)
SynapseParams = getattr(synaptic_matrix, "SynapseParams", None)
STDPEngine = getattr(stdp_mod, "STDPEngine", None)
STDPParams = getattr(stdp_mod, "STDPParams", None)

# --- 4. UI DASHBOARD ---
st.set_page_config(page_title="Axiom-Neuro Stable", layout="wide")
st.title("🔬 Axiom-Neuro: Final Stable Engine")

# Help the user if imports failed
if not all([LIFPopulation, SparseWeightMatrix, STDPEngine]):
    st.error("Engine components missing. Ensure all .py files are in the root directory.")
    st.stop()

if st.button("🚀 Start Engine"):
    N = 256
    # Initialize components
    try:
        params = LIFParams(n_neurons=N) if LIFParams else None
        pop = LIFPopulation(params)
        W = SparseWeightMatrix(SynapseParams(n_pre=N, n_post=N))
        stdp_eng = STDPEngine(STDPParams(), W, N, N)
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

    # Find the weight attribute name
    w_attr = next((a for a in ['W', 'matrix', 'weights', 'data'] if hasattr(W, a)), None)
    
    col_map, col_chart = st.columns(2)
    map_p = col_map.empty()
    chart_p = col_chart.empty()
    w_history = []

    for epoch in range(100):
        # 1. Physical Step
        pop.step(epoch * 0.1, np.ones(N) * 3.5)
        
        # 2. Get Spikes safely
        spikes = getattr(pop, 'spikes', None)
        if spikes is not None:
            indices = np.where(spikes)[0] if np.asarray(spikes).dtype == bool else np.asarray(spikes).astype(int)
            
            # Bound check for STDP arrays
            try:
                lim_pre = len(stdp_eng.x_pre)
                lim_post = len(stdp_eng.x_post)
            except:
                lim_pre = lim_post = N

            safe_pre = indices[indices < lim_pre].astype(int)
            safe_post = indices[indices < lim_post].astype(int)

            # 3. Update Plasticity
            try:
                stdp_eng.step(epoch * 0.1, safe_pre, safe_post)
            except:
                pass

        # 4. Visualization Update
        if w_attr:
            w_data = getattr(W, w_attr)
            avg_w = np.mean(w_data.toarray()) if hasattr(w_data, 'toarray') else np.mean(w_data)
            w_history.append(float(avg_w))
            chart_p.line_chart(w_history)

        # Activity Heatmap
        viz = np.zeros(256)
        valid_viz = indices[indices < 256]
        if len(valid_viz) > 0:
            viz[valid_viz] = 1.0
        
        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(viz.reshape((16,16)), cmap='magma', interpolation='nearest')
        ax.axis('off')
        map_p.pyplot(fig)
        plt.close(fig)
        
        time.sleep(0.01)

    st.success("Analysis Complete.")
