import sys
import os
import importlib.util
from pathlib import Path
import types
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time

# --- 1. BOOTSTRAP SYSTEM ---
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

def initialize_system():
    pkg_name = "axiom_neuro"
    if pkg_name in sys.modules: return
    axiom_neuro = types.ModuleType(pkg_name)
    axiom_neuro.__path__ = [str(root_dir)]
    axiom_neuro.__package__ = pkg_name
    sys.modules[pkg_name] = axiom_neuro
    for sub in ["core", "io", "learning", "geometry", "visualization"]:
        m = types.ModuleType(f"{pkg_name}.{sub}")
        m.__path__ = [str(root_dir)]
        m.__package__ = pkg_name
        sys.modules[f"{pkg_name}.{sub}"] = m
        setattr(axiom_neuro, sub, m)

initialize_system()

def load_and_inject(module_path, filename):
    full_path = root_dir / f"{filename}.py"
    if not full_path.exists(): return None
    spec = importlib.util.spec_from_file_location(module_path, str(full_path))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = module_path.rsplit('.', 1)[0]
    sys.modules[module_path] = mod
    spec.loader.exec_module(mod)
    return mod

# --- 2. LOAD MODULES ---
lif_model = load_and_inject("axiom_neuro.core.lif_model", "lif_model")
synaptic_matrix = load_and_inject("axiom_neuro.core.synaptic_matrix", "synaptic_matrix")
stdp = load_and_inject("axiom_neuro.learning.stdp", "stdp")
manifold_mapper = load_and_inject("axiom_neuro.geometry.manifold_mapper", "manifold_mapper")

def get_attr(mod, *names):
    for n in names:
        a = getattr(mod, n, None)
        if a: return a
    return None

LIFPopulation = get_attr(lif_model, "LIFPopulation")
LIFParams = get_attr(lif_model, "LIFParams")
SparseWeightMatrix = get_attr(synaptic_matrix, "SparseWeightMatrix")
SynapseParams = get_attr(synaptic_matrix, "SynapseParams")
STDPEngine = get_attr(stdp, "STDPEngine")
STDPParams = get_attr(stdp, "STDPParams")
ManifoldMapper = get_attr(manifold_mapper, "ManifoldMapper")
NeuronEmbedding = get_attr(manifold_mapper, "NeuronEmbedding", "Neuron")

# --- 3. UTILITY: FIND WEIGHTS ---
def get_weights_safely(W_obj):
    """Probes the object to find the weight array since '.weights' is missing."""
    for target in ['W', 'matrix', 'weights', 'data', '_weights']:
        if hasattr(W_obj, target):
            return getattr(W_obj, target)
    return np.array([0]) # Fallback

# --- 4. UI DASHBOARD ---
st.set_page_config(page_title="Axiom-Neuro Research", layout="wide")
st.title(r"🔬 Axiom-Neuro: High-Fidelity SNN Engine")

tab1, tab2 = st.tabs(["📊 Manifold Geometry", "🧠 Plasticity & Activity Map"])

with tab1:
    st.header("Topological Analysis")
    if st.button("🚀 Run Analysis"):
        N_topo = 400
        pop = LIFPopulation(LIFParams(n_neurons=N_topo))
        emb = NeuronEmbedding(N_topo, 'toroidal')
        mapper = ManifoldMapper(emb)
        for t in range(50):
            spikes = pop.step(t*0.1, np.random.normal(3.5, 0.5, N_topo))
            if np.any(spikes): mapper.update(t*0.1, np.where(spikes)[0])
        
        _, vols = mapper.get_volume_trace()
        if len(vols) > 0:
            st.line_chart(vols)
        else:
            st.warning("Insufficient spiking activity to map manifold.")

with tab2:
    st.header("Neuro-Activity Map & STDP")
    col_map, col_trace = st.columns(2)
    map_placeholder = col_map.empty()
    chart_placeholder = col_trace.empty()

    if st.button("🚀 Start Live Simulation"):
        N_stdp = 256 
        pop = LIFPopulation(LIFParams(n_neurons=N_stdp))
        W = SparseWeightMatrix(SynapseParams(n_pre=N_stdp, n_post=N_stdp))
        stdp_eng = STDPEngine(STDPParams(), W, N_stdp, N_stdp)
        
        # Determine internal expected shapes
        try:
            n_pre_limit = stdp_eng.x_pre.shape[0]
            n_post_limit = stdp_eng.x_post.shape[0]
        except:
            n_pre_limit, n_post_limit = N_stdp, N_stdp

        w_history = []
        for epoch in range(100):
            pop.step(epoch*0.1, np.ones(N_stdp) * 3.2)
            
            # --- INDEXERROR PREVENTION ---
            # Create boolean masks but ensure they match the BACKEND size exactly
            s_mask_pre = np.zeros(n_pre_limit, dtype=bool)
            s_mask_post = np.zeros(n_post_limit, dtype=bool)
            
            if pop.spikes is not None:
                # Get integer indices of spikes
                idx = np.where(pop.spikes)[0] if np.asarray(pop.spikes).dtype == bool else np.asarray(pop.spikes).astype(int)
                # Clip indices to prevent IndexError
                valid_idx_pre = idx[idx < n_pre_limit]
                valid_idx_post = idx[idx < n_post_limit]
                s_mask_pre[valid_idx_pre] = True
                s_mask_post[valid_idx_post] = True

            # Execute STDP
            method = getattr(stdp_eng, 'step', getattr(stdp_eng, 'update', None))
            if method:
                try:
                    method(epoch*0.1, s_mask_pre, s_mask_post)
                except IndexError:
                    # Final fallback: Try passing integer indices if boolean mask fails
                    method(epoch*0.1, np.where(s_mask_pre)[0], np.where(s_mask_post)[0])
            
            # Record Weights
            current_W = get_weights_safely(W)
            w_history.append(float(np.mean(current_W)))
            
            # Render Activity Map
            viz_grid = s_mask_pre[:256].astype(float).reshape((16, 16))
            fig, ax = plt.subplots(figsize=(3,3))
            ax.imshow(viz_grid, cmap='magma', interpolation='nearest')
            ax.axis('off')
            map_placeholder.pyplot(fig)
            plt.close(fig)
            
            # Render Weight Graph
            chart_placeholder.line_chart(w_history)
            time.sleep(0.01)
