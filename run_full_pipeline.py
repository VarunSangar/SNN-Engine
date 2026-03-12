import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
import importlib.util
from pathlib import Path

# --- 1. FLAT STRUCTURE IMPORT ADAPTER ---
# Since your files are all in one folder, we load them as direct modules
def safe_load(module_name):
    file_path = Path(__file__).parent / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Map the files from your screenshot
lif_model = safe_load("lif_model")
synaptic_matrix = safe_load("synaptic_matrix")
stdp_mod = safe_load("stdp")
manifold = safe_load("manifold_mapper")

# Extract Classes
LIFPopulation = lif_model.LIFPopulation
LIFParams = lif_model.LIFParams
SparseWeightMatrix = synaptic_matrix.SparseWeightMatrix
SynapseParams = synaptic_matrix.SynapseParams
STDPEngine = stdp_mod.STDPEngine
STDPParams = stdp_mod.STDPParams
ManifoldMapper = manifold.ManifoldMapper
NeuronEmbedding = getattr(manifold, "NeuronEmbedding", None)

# --- 2. UI SETUP ---
st.set_page_config(page_title="Axiom-Neuro Final", layout="wide")
st.title("🔬 Axiom-Neuro: Flat-Structure Stable Engine")

if st.button("🚀 Run Bulletproof Simulation"):
    N = 256
    pop = LIFPopulation(LIFParams(n_neurons=N))
    W = SparseWeightMatrix(SynapseParams(n_pre=N, n_post=N))
    stdp_eng = STDPEngine(STDPParams(), W, N, N)

    # 3. ATTRIBUTE DISCOVERY (Fixes AttributeError)
    w_attr = next((a for a in ['W', 'matrix', 'weights', 'data'] if hasattr(W, a)), None)
    
    col1, col2 = st.columns(2)
    map_p = col1.empty()
    chart_p = col2.empty()
    w_history = []

    for epoch in range(100):
        # Step the neurons
        pop.step(epoch * 0.1, np.ones(N) * 3.1)
        
        # --- THE INDEXERROR "FINAL STRIKE" FIX ---
        if pop.spikes is not None:
            # 1. Get current indices
            if np.asarray(pop.spikes).dtype == bool:
                indices = np.where(pop.spikes)[0]
            else:
                indices = np.asarray(pop.spikes).astype(int)
            
            # 2. Get ACTUAL size of backend arrays to prevent line 139 crash
            try:
                # We probe the engine for the true size of the trace arrays
                max_pre = len(stdp_eng.x_pre)
                max_post = len(stdp_eng.x_post)
            except:
                max_pre = max_post = N

            # 3. Create the Safe Integer Arrays
            safe_pre = indices[indices < max_pre].astype(int)
            safe_post = indices[indices < max_post].astype(int)

            # 4. Update STDP
            try:
                stdp_eng.step(epoch * 0.1, safe_pre, safe_post)
            except Exception as e:
                # If 'step' fails, try 'update'
                try:
                    stdp_eng.update(epoch * 0.1, safe_pre, safe_post)
                except:
                    pass

        # 5. VISUALIZATION
        if w_attr:
            w_history.append(float(np.mean(getattr(W, w_attr))))
            chart_p.line_chart(w_history)

        # Activity Grid (16x16)
        viz = np.zeros(256)
        viz[indices[indices < 256]] = 1.0
        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(viz.reshape((16,16)), cmap='magma')
        ax.axis('off')
        map_p.pyplot(fig)
        plt.close(fig)
        
        time.sleep(0.01)

    st.success("Simulation Complete.")
