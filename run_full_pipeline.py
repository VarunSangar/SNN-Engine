import sys
import os
import importlib.util
from pathlib import Path
import types
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- 1. CORE ARCHITECTURE ---
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

# --- 2. INJECT & ASSIGN ---
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

# --- 3. MA-LEVEL RESEARCH UI ---
st.set_page_config(page_title="Axiom-Neuro Research", layout="wide")
st.title(r"🔬 Axiom-Neuro: Neuromorphic Information Dynamics")

tab1, tab2 = st.tabs(["Topological Manifolds", "Plasticity Analysis"])

with tab1:
    st.header("Minkowski Functional Analysis")
    if st.button("🚀 Analyze Geometry"):
        with st.spinner("Computing..."):
            N = 400
            emb = NeuronEmbedding(N, 'toroidal')
            mapper = ManifoldMapper(emb)
            pop = LIFPopulation(LIFParams(n_neurons=N))
            
            for t_step in range(100):
                spikes = pop.step(t_step*0.1, np.random.normal(2.5, 0.5, N))
                mapper.update(t_step*0.1, np.where(spikes)[0])
                
            t_vals, vol_data = mapper.get_volume_trace()
            _, area_data = mapper.get_area_trace()
            
            # Use Raw Strings (r"") for LaTeX to avoid SyntaxWarnings
            iso_ratio = (36 * np.pi * vol_data**2) / (area_data**3 + 1e-9)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Manifold Volume ($V$)", f"{vol_data.mean():.4f}")
            m2.metric("Surface Complexity ($A$)", f"{area_data.mean():.4f}")
            # FIXED: Raw string for Eta
            m3.metric(r"Info Efficiency ($\eta$)", f"{iso_ratio.mean():.4f}")
            st.line_chart(vol_data)

with tab2:
    st.header("Synaptic Dynamics (STDP)")
    if st.button("Run Plasticity Trial"):
        with st.spinner("Learning..."):
            N = 100
            lp, sp = LIFParams(n_neurons=N), SynapseParams(n_pre=N, n_post=N)
            pop, W = LIFPopulation(lp), SparseWeightMatrix(sp)
            stdp_eng = STDPEngine(STDPParams(), W, N, N)
            
            w_history = []
            for epoch in range(50):
                pop.step(epoch*0.1, np.ones(N)*2.0)
                
                # FIXED: Flexible method caller to avoid AttributeError
                if hasattr(stdp_eng, 'step'):
                    stdp_eng.step(epoch*0.1, pop.spikes, pop.spikes)
                elif hasattr(stdp_eng, 'update'):
                    stdp_eng.update(epoch*0.1, pop.spikes, pop.spikes)
                
                w_history.append(W.weights.mean())
            
            st.line_chart(w_history)
            st.success("STDP Trial Complete")
