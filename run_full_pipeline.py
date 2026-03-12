import sys
import os
import importlib.util
from pathlib import Path
import types
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- 1. CORE ARCHITECTURAL INJECTOR ---
# This ensures Axiom-Neuro behaves like a real installed package
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

# --- 2. MODULE LOADING & CLASS MAPPING ---
lif_model = load_and_inject("axiom_neuro.core.lif_model", "lif_model")
synaptic_matrix = load_and_inject("axiom_neuro.core.synaptic_matrix", "synaptic_matrix")
stdp = load_and_inject("axiom_neuro.learning.stdp", "stdp")
manifold_mapper = load_and_inject("axiom_neuro.geometry.manifold_mapper", "manifold_mapper")
simulation_engine = load_and_inject("axiom_neuro.core.simulation_engine", "simulation_engine")

def get_attr(mod, *names):
    for n in names:
        a = getattr(mod, n, None)
        if a: return a
    return None

SimulationEngine = get_attr(simulation_engine, "SimulationEngine")
LIFPopulation = get_attr(lif_model, "LIFPopulation")
LIFParams = get_attr(lif_model, "LIFParams")
SparseWeightMatrix = get_attr(synaptic_matrix, "SparseWeightMatrix")
SynapseParams = get_attr(synaptic_matrix, "SynapseParams")
STDPEngine = get_attr(stdp, "STDPEngine")
STDPParams = get_attr(stdp, "STDPParams")
ManifoldMapper = get_attr(manifold_mapper, "ManifoldMapper")
NeuronEmbedding = get_attr(manifold_mapper, "NeuronEmbedding", "Neuron")

# --- 3. RESEARCH DASHBOARD UI ---
st.set_page_config(page_title="Axiom-Neuro Research", layout="wide", page_icon="🔬")
st.title(r"🔬 Axiom-Neuro: High-Fidelity SNN Engine")

tab1, tab2, tab3 = st.tabs(["Manifold Geometry", "Plasticity Trials", "Engine Specs"])

with tab1:
    st.header("Minkowski Functional Analysis")
    st.markdown("Quantifying neural state-space topology through Minkowski TDA.")
    
    if st.button("🚀 Run Topological Analysis"):
        with st.spinner("Mapping Neural State-Space..."):
            N = 400
            emb = NeuronEmbedding(N, 'toroidal')
            mapper = ManifoldMapper(emb)
            pop = LIFPopulation(LIFParams(n_neurons=N))
            
            for t_step in range(150):
                # Using a slightly higher base current to ensure spiking for the manifold
                spikes = pop.step(t_step*0.1, np.random.normal(3.0, 0.5, N))
                if np.any(spikes):
                    mapper.update(t_step*0.1, np.where(spikes)[0])
                
            t_vals, vol_data = mapper.get_volume_trace()
            _, area_data = mapper.get_area_trace()
            
            if len(vol_data) > 0 and not np.all(vol_data == 0):
                # Isoperimetric Ratio: A measure of spherical efficiency in state-space
                iso_ratio = (36 * np.pi * vol_data**2) / (area_data**3 + 1e-9)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Manifold Volume ($V$)", f"{np.mean(vol_data):.4f}")
                m2.metric("Surface Complexity ($A$)", f"{np.mean(area_data):.4f}")
                # Fixed: Raw string for Eta symbol
                m3.metric(r"Info Efficiency ($\eta$)", f"{np.mean(iso_ratio):.4f}")
                
                st.line_chart(vol_data)
            else:
                st.warning("Insufficient spiking activity. Increase base current in config.")

with tab2:
    st.header("Synaptic Dynamics (STDP)")
    if st.button("🚀 Start Learning Trial"):
        with st.spinner("Executing Hebbian Plasticity..."):
            N = 100
            lp, sp = LIFParams(n_neurons=N), SynapseParams(n_pre=N, n_post=N)
            pop, W = LIFPopulation(lp), SparseWeightMatrix(sp)
            stdp_eng = STDPEngine(STDPParams(), W, N, N)
            
            w_history = []
            for epoch in range(100):
                pop.step(epoch*0.1, np.ones(N)*2.5)
                
                # --- THE CRITICAL FIX: BOOLEAN MASKING ---
                # This prevents the IndexError in stdp.py line 139
                s_mask = np.zeros(N, dtype=bool)
                if pop.spikes is not None:
                    # Works whether pop.spikes is a list of indices or a mask
                    s_mask[pop.spikes] = True 
                
                # --- UNIVERSAL DISPATCHER ---
                # Probes the object for the correct method name
                if hasattr(stdp_eng, 'step'):
                    stdp_eng.step(epoch*0.1, s_mask, s_mask)
                elif hasattr(stdp_eng, 'update'):
                    stdp_eng.update(epoch*0.1, s_mask, s_mask)
                elif hasattr(stdp_eng, 'apply_stdp'):
                    stdp_eng.apply_stdp(epoch*0.1, s_mask, s_mask)
                
                w_history.append(np.mean(W.weights))
            
            st.line_chart(w_history)
            st.success("Synaptic stabilization complete.")

with tab3:
    st.header("Engine Specifications")
    st.write("Current Axiom-Neuro Build: **v2.1 (MA-Stable)**")
    st.code(f"""
    Neuron Model: Leaky Integrate-and-Fire (LIF)
    Plasticity: Spike-Timing-Dependent (STDP)
    Geometry: Minkowski Functional Mapper
    State-Space: Toroidal Embedding
    Python Version: {sys.version}
    """)
