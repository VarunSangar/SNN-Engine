import sys
import os
import importlib.util
from pathlib import Path
import types
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# 1. SETUP ROOT PATH
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

# 2. CREATE VIRTUAL PACKAGE STRUCTURE
pkg_name = "axiom_neuro"
if pkg_name not in sys.modules:
    axiom_neuro = types.ModuleType(pkg_name)
    axiom_neuro.__path__ = [str(root_dir)]
    axiom_neuro.__package__ = pkg_name
    sys.modules[pkg_name] = axiom_neuro

    for sub in ["core", "io", "learning", "geometry", "visualization"]:
        full_sub = f"{pkg_name}.{sub}"
        m = types.ModuleType(full_sub)
        m.__path__ = [str(root_dir)] 
        m.__package__ = pkg_name
        sys.modules[full_sub] = m
        setattr(axiom_neuro, sub, m)

# 3. THE MAGIC LOADER
def load_and_inject(module_path, filename):
    full_path = root_dir / f"{filename}.py"
    if not full_path.exists():
        return None
    if module_path in sys.modules:
        return sys.modules[module_path]

    spec = importlib.util.spec_from_file_location(module_path, str(full_path))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = module_path.rsplit('.', 1)[0]
    sys.modules[module_path] = mod
    spec.loader.exec_module(mod)
    return mod

# 4. INJECT IN ORDER
lif_model        = load_and_inject("axiom_neuro.core.lif_model", "lif_model")
synaptic_matrix  = load_and_inject("axiom_neuro.core.synaptic_matrix", "synaptic_matrix")
stdp             = load_and_inject("axiom_neuro.learning.stdp", "stdp")
manifold_mapper  = load_and_inject("axiom_neuro.geometry.manifold_mapper", "manifold_mapper")
plotter          = load_and_inject("axiom_neuro.visualization.plotter", "plotter")
data_loader      = load_and_inject("axiom_neuro.io.data_loader", "data_loader")
simulation_engine = load_and_inject("axiom_neuro.core.simulation_engine", "simulation_engine")

# 5. ASSIGN CLASSES
SimulationEngine       = getattr(simulation_engine, "SimulationEngine", None)
SimConfig              = getattr(simulation_engine, "SimConfig", None)
LIFPopulation          = getattr(lif_model, "LIFPopulation", None)
LIFParams              = getattr(lif_model, "LIFParams", None)
SparseWeightMatrix     = getattr(synaptic_matrix, "SparseWeightMatrix", None)
SynapseParams          = getattr(synaptic_matrix, "SynapseParams", None)
STDPEngine             = getattr(stdp, "STDPEngine", None)
STDPParams             = getattr(stdp, "STDPParams", None)
ManifoldMapper         = getattr(manifold_mapper, "ManifoldMapper", None)
NeuronEmbedding        = getattr(manifold_mapper, "NeuronEmbedding", getattr(manifold_mapper, "Neuron", None))
SyntheticDataGenerator = getattr(data_loader, "SyntheticDataGenerator", None)
SpikeDataLoader        = getattr(data_loader, "SpikeDataLoader", None)
ReplayEngine           = getattr(data_loader, "ReplayEngine", None)
RasterPlot             = getattr(plotter, "RasterPlot", None)
NetworkDashboard       = getattr(plotter, "NetworkDashboard", None)

# --- APP CONFIG ---
st.set_page_config(page_title="Axiom-Neuro Engine", page_icon="🧠", layout="wide")
Path("outputs").mkdir(exist_ok=True)

# --- UI TABS ---
st.title("🧠 Axiom-Neuro: Revolutionary SNN Engine")
tabs = st.tabs(["LIF Simulation", "Synthetic Replay", "Manifold Analysis", "Membrane Potential"])

with tabs[0]:
    st.header("Example 1: Basic LIF + STDP")
    if st.button("Run Simulation", key="ex1"):
        with st.spinner("Processing..."):
            cfg = SimConfig(n_neurons=500, duration_ms=200.0, dt=0.1, conn_density=0.08, base_current=2.2,
                            current_noise=0.8, stdp_enabled=True, homeostasis=True, r_target=8.0,
                            manifold_enabled=True, output_dir="outputs")
            sim = SimulationEngine(cfg)
            sim.run()
            st.success("Simulation Complete")
            if Path("outputs/raster_final.png").exists():
                st.image("outputs/raster_final.png")

with tabs[1]:
    st.header("Example 2: Data Replay & Weight Learning")
    if st.button("Run Replay", key="ex2"):
        with st.spinner("Generating Synthetic Spikes..."):
            gen = SyntheticDataGenerator()
            data = gen.oscillatory_spikes(n_neurons=200, duration_ms=100.0, freq_hz=40.0)
            
            lp, sp = LIFParams(n_neurons=200), SynapseParams(n_pre=200, n_post=200)
            pop, W = LIFPopulation(lp), SparseWeightMatrix(sp)
            stdp_eng = STDPEngine(STDPParams(), W, 200, 200)
            
            engine = ReplayEngine(pop, W, stdp_eng)
            res = engine.run(data, n_epochs=3)
            st.line_chart(res["weight_mean_history"])
            st.write(f"Final Weight Mean: {res['weight_mean_history'][-1]:.4f}")

with tabs[2]:
    st.header("Example 3: Information Manifold Geometry")
    if st.button("Compute Manifold", key="ex3"):
        with st.spinner("Mapping Neural State-Space..."):
            N = 300
            emb = NeuronEmbedding(N, 'toroidal')
            mapper = ManifoldMapper(emb)
            pop = LIFPopulation(LIFParams(n_neurons=N))
            W = SparseWeightMatrix(SynapseParams(n_pre=N, n_post=N))
            
            for step in range(100):
                spikes = pop.step(step*0.1, np.ones(N)*2.5)
                mapper.update(step*0.1, np.where(spikes)[0])
            
            t_v, vols = mapper.get_volume_trace()
            fig, ax = plt.subplots(facecolor="#0d0f14")
            ax.plot(t_v, vols, color="#cc88ff")
            ax.set_title("Manifold Volume Over Time", color="white")
            st.pyplot(fig)

with tabs[3]:
    st.header("Example 4: Membrane Dynamics")
    if st.button("Generate V(t)", key="ex4"):
        N, n_steps = 20, 1000
        pop = LIFPopulation(LIFParams(n_neurons=N))
        v_trace = np.zeros((N, n_steps))
        for s in range(n_steps):
            pop.step(s*0.05, np.linspace(1.0, 3.0, N))
            v_trace[:, s] = pop.V
        
        st.line_chart(v_trace[:5].T)
