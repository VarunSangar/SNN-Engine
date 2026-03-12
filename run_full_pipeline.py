import sys
import os
import importlib.util
from pathlib import Path
import types
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time

# --- 1. CORE ARCHITECTURAL INJECTOR ---
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

# --- 2. MODULE LOADING ---
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
SynapseParams = get_attr(synapse_params if 'synapse_params' in locals() else synaptic_matrix, "SynapseParams")
STDPEngine = get_attr(stdp, "STDPEngine")
STDPParams = get_attr(stdp, "STDPParams")
ManifoldMapper = get_attr(manifold_mapper, "ManifoldMapper")
NeuronEmbedding = get_attr(manifold_mapper, "NeuronEmbedding", "Neuron")

# --- 3. RESEARCH DASHBOARD UI ---
st.set_page_config(page_title="Axiom-Neuro Research", layout="wide", page_icon="🔬")
st.title(r"🔬 Axiom-Neuro: High-Fidelity SNN Engine")

tab1, tab2 = st.tabs(["📊 Manifold Geometry", "🧠 Plasticity & Activity Map"])

with tab1:
    st.header("Topological Manifold Analysis")
    if st.button("🚀 Run Topological Analysis"):
        with st.spinner("Mapping Neural State-Space..."):
            N_topo = 400
            emb = NeuronEmbedding(N_topo, 'toroidal')
            mapper = ManifoldMapper(emb)
            pop = LIFPopulation(LIFParams(n_neurons=N_topo))
            for t_step in range(150):
                spikes = pop.step(t_step*0.1, np.random.normal(3.2, 0.4, N_topo))
                if np.any(spikes):
                    mapper.update(t_step*0.1, np.where(spikes)[0])
            t_vals, vol_data = mapper.get_volume_trace()
            _, area_data = mapper.get_area_trace()
            if len(vol_data) > 2:
                iso_ratio = (36 * np.pi * vol_data**2) / (area_data**3 + 1e-9)
                m1, m2, m3 = st.columns(3)
                m1.metric("Volume", f"{np.mean(vol_data):.4f}")
                m2.metric("Surface", f"{np.mean(area_data):.4f}")
                m3.metric(r"Efficiency ($\eta$)", f"{np.mean(iso_ratio):.4f}")
                st.line_chart(vol_data)

with tab2:
    st.header("Neuro-Activity Map & STDP")
    col_map, col_trace = st.columns([1, 1])
    with col_map: map_placeholder = st.empty()
    with col_trace: chart_placeholder = st.empty()

    if st.button("🚀 Start Live Simulation"):
        # USE 256 TO ENSURE SQUARE 16x16 GRID
        N_stdp = 256 
        grid_dim = 16
        lp = LIFParams(n_neurons=N_stdp)
        # Ensure SynapseParams matches N_stdp
        sp = SynapseParams(n_pre=N_stdp, n_post=N_stdp)
        
        pop = LIFPopulation(lp)
        W = SparseWeightMatrix(sp)
        stdp_eng = STDPEngine(STDPParams(), W, N_stdp, N_stdp)
        
        # --- THE FIX: DETECT BACKEND SHAPE ---
        # We look at the actual trace array in your STDPEngine to get the true N
        target_n_pre = stdp_eng.x_pre.shape[0] if hasattr(stdp_eng, 'x_pre') else N_stdp
        target_n_post = stdp_eng.x_post.shape[0] if hasattr(stdp_eng, 'x_post') else N_stdp

        w_history = []
        for epoch in range(100):
            pop.step(epoch*0.1, np.ones(N_stdp) * 2.8)
            
            # --- ROBUST MASK GENERATION ---
            # Pre-spike mask
            s_mask_pre = np.zeros(target_n_pre, dtype=bool)
            # Post-spike mask (assuming same pop for this trial)
            s_mask_post = np.zeros(target_n_post, dtype=bool)
            
            if pop.spikes is not None:
                # Get indices regardless of format
                indices = np.where(pop.spikes)[0] if np.asarray(pop.spikes).dtype == bool else np.asarray(pop.spikes).astype(int)
                # Only map indices that fit the target backend size
                valid_indices_pre = indices[indices < target_n_pre]
                valid_indices_post = indices[indices < target_n_post]
                s_mask_pre[valid_indices_pre] = True
                s_mask_post[valid_indices_post] = True

            # Universal Dispatcher
            method = getattr(stdp_eng, 'step', getattr(stdp_eng, 'update', None))
            if method:
                # Passing the specific shapes the backend expects
                method(epoch*0.1, s_mask_pre, s_mask_post)
            
            w_history.append(float(np.mean(W.weights)))
            
            # Update Heatmap (using s_mask_pre for the 16x16 view)
            viz_grid = s_mask_pre[:256].reshape((16, 16)).astype(float)
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(viz_grid, cmap='magma')
            ax.axis('off')
            map_placeholder.pyplot(fig)
            plt.close(fig)
            
            chart_placeholder.line_chart(w_history)
            time.sleep(0.01)
