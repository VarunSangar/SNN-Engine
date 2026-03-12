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
    st.header("Topological Analysis")
    if st.button("🚀 Run Analysis"):
        N_topo = 400
        pop = LIFPopulation(LIFParams(n_neurons=N_topo))
        emb = NeuronEmbedding(N_topo, 'toroidal')
        mapper = ManifoldMapper(emb)
        for t in range(100):
            spikes = pop.step(t*0.1, np.random.normal(3.0, 0.5, N_topo))
            if np.any(spikes): mapper.update(t*0.1, np.where(spikes)[0])
        st.line_chart(mapper.get_volume_trace()[1])

with tab2:
    st.header("Neuro-Activity Map & STDP")
    col_map, col_trace = st.columns(2)
    map_placeholder = col_map.empty()
    chart_placeholder = col_trace.empty()

    if st.button("🚀 Start Live Simulation"):
        # We use 256 neurons for a clean 16x16 grid
        N_stdp = 256 
        pop = LIFPopulation(LIFParams(n_neurons=N_stdp))
        W = SparseWeightMatrix(SynapseParams(n_pre=N_stdp, n_post=N_stdp))
        stdp_eng = STDPEngine(STDPParams(), W, N_stdp, N_stdp)
        
        # --- THE ULTIMATE SHAPE SYNCHRONIZER ---
        # We grab the EXACT expected size from the backend arrays themselves
        # This bypasses any mismatch between UI config and STDP internal state
        try:
            n_pre_required = len(stdp_eng.x_pre)
            n_post_required = len(stdp_eng.x_post)
        except AttributeError:
            n_pre_required = N_stdp
            n_post_required = N_stdp

        w_history = []
        for epoch in range(100):
            pop.step(epoch*0.1, np.ones(N_stdp) * 3.0)
            
            # --- CREATE THE PERFECT MASK ---
            # 1. Start with zeros of the EXACT size the backend expects
            s_mask_pre = np.zeros(n_pre_required, dtype=bool)
            s_mask_post = np.zeros(n_post_required, dtype=bool)
            
            if pop.spikes is not None:
                # 2. Get active indices from current population
                active_indices = np.where(pop.spikes)[0] if np.asarray(pop.spikes).dtype == bool else np.asarray(pop.spikes).astype(int)
                
                # 3. Only activate indices that actually exist in the backend
                valid_pre = active_indices[active_indices < n_pre_required]
                valid_post = active_indices[active_indices < n_post_required]
                
                s_mask_pre[valid_pre] = True
                s_mask_post[valid_post] = True

            # 4. EXECUTE
            method = getattr(stdp_eng, 'step', getattr(stdp_eng, 'update', None))
            if method:
                # We send the perfectly sized boolean mask
                method(epoch*0.1, s_mask_pre, s_mask_post)
            
            w_history.append(float(np.mean(W.weights)))
            
            # Update Activity Map (16x16)
            viz_grid = np.zeros(256)
            # Ensure viz matches the local N_stdp size
            current_spikes = np.where(pop.spikes)[0] if np.asarray(pop.spikes).dtype == bool else pop.spikes
            viz_grid[current_spikes[current_spikes < 256]] = 1.0
            
            fig, ax = plt.subplots(figsize=(3,3))
            ax.imshow(viz_grid.reshape((16,16)), cmap='magma')
            ax.axis('off')
            map_placeholder.pyplot(fig)
            plt.close(fig)
            
            chart_placeholder.line_chart(w_history)
            time.sleep(0.01)
