import sys
import os
import importlib.util
from pathlib import Path
import types
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup root path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

# 2. CREATE VIRTUAL PACKAGE STRUCTURE
pkg_name = "axiom_neuro"
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

print("🧠 Axiom-Neuro: Virtual Environment Virtualized.")

# 3. THE MAGIC LOADER
def load_and_inject(module_path, filename):
    full_path = root_dir / f"{filename}.py"
    if not full_path.exists():
        print(f"⚠️ Warning: {filename}.py not found!")
        return None

    spec = importlib.util.spec_from_file_location(module_path, str(full_path))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = module_path.rsplit('.', 1)[0]
    sys.modules[module_path] = mod
    spec.loader.exec_module(mod)
    return mod

# 4. INJECT IN ORDER
print("📦 Injecting Dependencies...")
lif_model        = load_and_inject("axiom_neuro.core.lif_model", "lif_model")
synaptic_matrix  = load_and_inject("axiom_neuro.core.synaptic_matrix", "synaptic_matrix")
stdp             = load_and_inject("axiom_neuro.learning.stdp", "stdp")
manifold_mapper  = load_and_inject("axiom_neuro.geometry.manifold_mapper", "manifold_mapper")
plotter          = load_and_inject("axiom_neuro.visualization.plotter", "plotter")
data_loader      = load_and_inject("axiom_neuro.io.data_loader", "data_loader")

print("⚙️ Loading Simulation Engine...")
simulation_engine = load_and_inject("axiom_neuro.core.simulation_engine", "simulation_engine")

# 5. ASSIGN CLASSES (The fix for the AttributeError)
SimulationEngine       = getattr(simulation_engine, "SimulationEngine", None)
SimConfig              = getattr(simulation_engine, "SimConfig", None)
LIFPopulation          = getattr(lif_model, "LIFPopulation", None)
LIFParams              = getattr(lif_model, "LIFParams", None)
SparseWeightMatrix     = getattr(synaptic_matrix, "SparseWeightMatrix", None)
SynapseParams          = getattr(synaptic_matrix, "SynapseParams", None)
STDPEngine             = getattr(stdp, "STDPEngine", None)
STDPParams             = getattr(stdp, "STDPParams", None)
ManifoldMapper         = getattr(manifold_mapper, "ManifoldMapper", None)
# Fix for line 76 error: try both 'Neuron' and 'NeuronEmbedding'
NeuronEmbedding        = getattr(manifold_mapper, "NeuronEmbedding", getattr(manifold_mapper, "Neuron", None))
SyntheticDataGenerator = getattr(data_loader, "SyntheticDataGenerator", None)
SpikeDataLoader        = getattr(data_loader, "SpikeDataLoader", None)
ReplayEngine           = getattr(data_loader, "ReplayEngine", None)
RasterPlot             = getattr(plotter, "RasterPlot", None)
NetworkDashboard       = getattr(plotter, "NetworkDashboard", None)

# NOTE: Original import lines are removed/skipped because load_and_inject 
# already handled the placement in sys.modules.

def example_1_basic_simulation():
    """Run a basic LIF network and produce raster + dashboard."""
    print("\n" + "="*60)
    print("  Example 1: Basic LIF Simulation with STDP")
    print("="*60)

    cfg = SimConfig(
        n_neurons         = 500,
        duration_ms       = 200.0,
        dt                 = 0.1,
        conn_density      = 0.08,
        base_current      = 2.2,
        current_noise      = 0.8,
        stdp_enabled      = True,
        homeostasis       = True,
        r_target          = 8.0,
        manifold_enabled  = True,
        record_interval   = 5,
        save_raster       = True,
        save_dashboard    = True,
        output_dir        = "outputs",
    )

    sim    = SimulationEngine(cfg)
    result = sim.run(verbose=True, progress_every=500)

    if sim.mapper:
        t_v, vols = sim.mapper.get_volume_trace()
        if len(vols):
            print(f"Volume range       : [{vols.min():.4f}, {vols.max():.4f}]")

        mink_verts = sim.mapper.minkowski_sum_consecutive()
        if mink_verts is not None:
            print(f"Minkowski Sum verts: {len(mink_verts)}")

    sim.save_results("example_1", result)
    return result

def example_2_synthetic_replay():
    """Generate oscillatory spike data and replay through SNN."""
    print("\n" + "="*60)
    print("  Example 2: Synthetic Data Replay + Weight Learning")
    print("="*60)

    gen = SyntheticDataGenerator()
    data = gen.oscillatory_spikes(
        n_neurons   = 200,
        duration_ms = 100.0,
        freq_hz     = 40.0,
        base_rate   = 8.0,
        dt          = 0.1,
    )
    
    loader = SpikeDataLoader()
    Path("outputs").mkdir(exist_ok=True)
    loader.save_csv(data, "outputs/synthetic_spikes.csv")
    data2 = loader.load_csv("outputs/synthetic_spikes.csv", n_neurons=200)

    N  = data.n_neurons
    lp = LIFParams(n_neurons=N, dt=0.1)
    sp = SynapseParams(n_pre=N, n_post=N, density=0.1, w_init_mean=0.3)
    pop = LIFPopulation(lp, seed=0)
    W   = SparseWeightMatrix(sp, seed=0)
    stdp_p = STDPParams(A_plus=0.02, A_minus=0.021, homeostasis_enabled=True, r_target=8.0)
    stdp_eng = STDPEngine(stdp_p, W, N, N, dt=0.1)

    engine = ReplayEngine(pop, W, stdp_eng)
    res    = engine.run(data2, n_epochs=3, base_current=1.5, replay_gain=4.0)

    return res

def example_3_manifold_analysis():
    """Deep-dive into manifold geometry during network dynamics."""
    print("\n" + "="*60)
    print("  Example 3: Information Manifold Geometry Analysis")
    print("="*60)

    N   = 300
    lp  = LIFParams(n_neurons=N, dt=0.1, tau_m=15.0)
    sp  = SynapseParams(n_pre=N, n_post=N, density=0.1, w_init_mean=0.8)
    pop = LIFPopulation(lp, seed=7)
    W   = SparseWeightMatrix(sp, seed=7)

    emb    = NeuronEmbedding(N, 'toroidal', seed=7)
    mapper = ManifoldMapper(emb, min_firing=8, history_len=1000)

    I_base = np.random.default_rng(7).normal(2.5, 0.7, N).clip(0)

    for step in range(1000):
        t = step * 0.1
        I_syn   = W.compute_current(pop.spikes.astype(np.float32))
        spikes  = pop.step(t, I_base + I_syn)
        if step % 5 == 0:
            mapper.update(t, np.where(spikes)[0])

    t_v, vols = mapper.get_volume_trace()
    if len(t_v) > 1:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), facecolor="#0d0f14")
        # Plotting logic...
        plt.tight_layout()
        Path("outputs").mkdir(exist_ok=True)
        fig.savefig("outputs/manifold_geometry.png", dpi=150, facecolor="#0d0f14")
        plt.close(fig)
    return mapper

def example_4_membrane_potential_trace():
    """Visualise single-neuron membrane potential dynamics."""
    print("\n" + "="*60)
    print("  Example 4: Membrane Potential Traces (V(t))")
    print("="*60)

    N  = 20
    lp = LIFParams(n_neurons=N, dt=0.05, noise_sigma=0.3)
    pop = LIFPopulation(lp, seed=0)

    n_steps = 2000
    V_trace = np.zeros((N, n_steps))
    I_ext   = np.array([1.0 + 0.3*i for i in range(N)])

    for step in range(n_steps):
        t = step * lp.dt
        pop.step(t, I_ext)
        V_trace[:, step] = pop.V.copy()

    fig, axes = plt.subplots(5, 1, figsize=(14, 8), sharex=True, facecolor="#0d0f14")
    # Plotting logic...
    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/membrane_potential.png", dpi=150, facecolor="#0d0f14")
    plt.close(fig)

if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    example_4_membrane_potential_trace()
    example_1_basic_simulation()
    example_2_synthetic_replay()
    example_3_manifold_analysis()
    print("\n" + "="*60)
    print("  All examples complete. Check outputs/ directory.")
    print("="*60)
