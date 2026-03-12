import sys
import os
import importlib.util
from pathlib import Path
import types
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time

# --- 1. ARCHITECTURAL INJECTOR ---
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
SynapseParams = get_attr(synaptic_matrix, "SynapseParams")
STDPEngine = get_attr(stdp, "STDPEngine")
STDPParams = get_attr(stdp, "STDPParams")
ManifoldMapper = get_attr(manifold_mapper, "ManifoldMapper")
NeuronEmbedding = get_attr(manifold_mapper, "NeuronEmbedding", "Neuron")

# --- 3. UI DASHBOARD ---
st.set_page_config(page_title="Axiom-Neuro Research", layout="wide")
st.title(r"🔬 Axiom-Neuro: High-Fidelity SNN Engine")

tab1, tab2 = st.tabs(["📊 Manifold Geometry", "🧠 Plasticity & Activity Map"])

with tab1:
    st.header("Topological Manifold Analysis")
    if st.button("🚀 Run Topological Analysis"):
        N_topo = 400
        emb = NeuronEmbedding(N_topo, 'toroidal')
        mapper = ManifoldMapper(emb)
        pop = LIFPopulation(LIFParams(n_neurons=N_topo))
        for t_step in range(150):
            spikes = pop.step(t_step*0.1, np.random.normal(3.2, 0.4, N_topo))
            if np.any(spikes):
                mapper.update(t_step*0.1, np.where(spikes)[0])
        _, vol_data = mapper.get_volume_trace()
        st.line_chart(vol_data)

with tab2:
    st.header("Neuro-Activity Map & STDP")
    col_map, col_trace = st.columns([1, 1])
    with col_map: map_placeholder = st.empty()
    with col_trace: chart_placeholder = st.empty()

    if st.button("🚀 Start Live Simulation"):
        N_stdp = 256 
        lp = LIFParams(n_neurons=N_stdp)
        sp = SynapseParams(n_pre=N_stdp, n_post=N_stdp)
        pop = LIFPopulation(lp)
        W = SparseWeightMatrix(sp)
        stdp_eng = STDPEngine(STDPParams(), W, N_stdp, N_stdp)
        
        # --- SHAPE DISCOVERY ---
        n_pre_limit = stdp_eng.x_pre.shape[0]
        n_post_limit = stdp_eng.x_post.shape[0]

        w_history = []
        for epoch in range(100):
            # 1. Step the population
            pop.step(epoch*0.1, np.ones(N_stdp) * 2.8)
            
            # 2. Convert spikes to INTEGER INDICES (The Fix)
            # This ensures pop.spikes is handled correctly whether it's bool or int
            raw_spikes = np.asarray(pop.spikes)
            if raw_spikes.dtype == bool:
                spike_idx = np.where(raw_spikes)[0].astype(int)
            else:
                spike_idx = raw_spikes.astype(int)

            # 3. BOUNDARY CLIPPING (Prevents IndexError)
            idx_pre = spike_idx[spike_idx < n_pre_limit]
            idx_post = spike_idx[spike_idx < n_post_limit]

            # 4. DISPATCH
            method = getattr(stdp_eng, 'step', getattr(stdp_eng, 'update', None))
            if method:
                # We pass the integer arrays directly
                method(epoch*0.1, idx_pre, idx_post)
            
            w_history.append(float(np.mean(W.weights)))
            
            # --- ACTIVITY HEATMAP ---
            # Create visualization mask for the grid
            viz_mask = np.zeros(256)
            viz_mask[spike_idx[spike_idx < 256]] = 1.0
            viz_grid = viz_mask.reshape((16, 16))
            
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(viz_grid, cmap='magma', interpolation='nearest')
            ax.axis('off')
            map_placeholder.pyplot(fig)
            plt.close(fig)
            
            chart_placeholder.line_chart(w_history)
            time.sleep(0.01)
