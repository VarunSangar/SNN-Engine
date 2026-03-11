import sys
import os
import importlib.util
from pathlib import Path

root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

def force_load(name, filename):
    full_name = f"axiom_neuro.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    
    file_path = root_dir / f"{filename}.py"
    if not file_path.exists():
        print(f"⚠️ Error: {filename}.py not found in root!")
        return None

    spec = importlib.util.spec_from_file_location(full_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "axiom_neuro"
    sys.modules[full_name] = mod
    sys.modules[name] = mod 
    spec.loader.exec_module(mod)
    return mod

print("🧠 Axiom-Neuro: Engaging Universal Discovery...")

# 1. Load the Modules
lif_model = force_load("lif_model", "lif_model")
synaptic_matrix = force_load("synaptic_matrix", "synaptic_matrix")
stdp = force_load("stdp", "stdp")
manifold_mapper = force_load("manifold_mapper", "manifold_mapper")
plotter = force_load("plotter", "plotter")
simulation_engine = force_load("simulation_engine", "simulation_engine")
data_loader = force_load("data_loader", "data_loader")

# 2. Safety Mapping Helper
def get_attr(module, name):
    if hasattr(module, name):
        return getattr(module, name)
    # If not found, look for similar names (case-insensitive)
    for attr in dir(module):
        if attr.lower() == name.lower():
            return getattr(module, attr)
    return None

# 3. Final Class Mapping
SimulationEngine = get_attr(simulation_engine, "SimulationEngine")
SimConfig        = get_attr(simulation_engine, "SimConfig")
SyntheticDataGenerator = get_attr(data_loader, "SyntheticDataGenerator")
SpikeDataLoader        = get_attr(data_loader, "SpikeDataLoader")
ReplayEngine           = get_attr(data_loader, "ReplayEngine")
LIFPopulation    = get_attr(lif_model, "LIFPopulation")
LIFParams        = get_attr(lif_model, "LIFParams")
SparseWeightMatrix = get_attr(synaptic_matrix, "SparseWeightMatrix")
SynapseParams      = get_attr(synaptic_matrix, "SynapseParams")
STDPEngine       = get_attr(stdp, "STDPEngine")
STDPParams       = get_attr(stdp, "STDPParams")
ManifoldMapper   = get_attr(manifold_mapper, "ManifoldMapper")
NeuronEmbedding  = get_attr(manifold_mapper, "NeuronEmbedding")
RasterPlot       = get_attr(plotter, "RasterPlot")
NetworkDashboard = get_attr(plotter, "NetworkDashboard")

if SimulationEngine is None:
    print(f"❌ CRITICAL ERROR: Could not find SimulationEngine in simulation_engine.py")
    print(f"Available attributes: {[a for a in dir(simulation_engine) if not a.startswith('_')]}")

print("🚀 Pipeline Ready.")

# NOW your original imports will work:
from simulation_engine import SimulationEngine, SimConfig
from data_loader       import SyntheticDataGenerator, SpikeDataLoader, ReplayEngine
from lif_model          import LIFPopulation, LIFParams
from synaptic_matrix    import SparseWeightMatrix, SynapseParams
from stdp               import STDPEngine, STDPParams
from manifold_mapper    import NeuronEmbedding, ManifoldMapper
from plotter            import RasterPlot, NetworkDashboard


def example_1_basic_simulation():
    """Run a basic LIF network and produce raster + dashboard."""
    print("\n" + "="*60)
    print("  Example 1: Basic LIF Simulation with STDP")
    print("="*60)

    cfg = SimConfig(
        n_neurons         = 500,
        duration_ms       = 200.0,
        dt                = 0.1,
        conn_density      = 0.08,
        base_current      = 2.2,
        current_noise     = 0.8,
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

    print(f"\nManifold snapshots : {len([s for s in sim.mapper.history if s.valid])}")
    if sim.mapper:
        t_v, vols = sim.mapper.get_volume_trace()
        if len(vols):
            print(f"Volume range       : [{vols.min():.4f}, {vols.max():.4f}]")

        # Minkowski Sum of last two manifolds
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

    # Generate oscillatory (40Hz gamma) spike train
    data = gen.oscillatory_spikes(
        n_neurons   = 200,
        duration_ms = 100.0,
        freq_hz     = 40.0,
        base_rate   = 8.0,
        dt          = 0.1,
    )
    print(f"Synthetic data: {data.n_spikes:,} spikes, {data.mean_firing_rate:.1f} Hz mean")

    # Save to CSV and reload (demonstrates CSV pipeline)
    loader = SpikeDataLoader()
    Path("outputs").mkdir(exist_ok=True)
    loader.save_csv(data, "outputs/synthetic_spikes.csv")
    data2 = loader.load_csv("outputs/synthetic_spikes.csv", n_neurons=200)
    print(f"Reload from CSV : {data2.n_spikes:,} spikes  ✓")

    # Build SNN for replay
    N  = data.n_neurons
    lp = LIFParams(n_neurons=N, dt=0.1)
    sp = SynapseParams(n_pre=N, n_post=N, density=0.1, w_init_mean=0.3)
    pop = LIFPopulation(lp, seed=0)
    W   = SparseWeightMatrix(sp, seed=0)
    stdp_p = STDPParams(A_plus=0.02, A_minus=0.021, homeostasis_enabled=True, r_target=8.0)
    stdp = STDPEngine(stdp_p, W, N, N, dt=0.1)

    engine = ReplayEngine(pop, W, stdp)
    res    = engine.run(data2, n_epochs=3, base_current=1.5, replay_gain=4.0)

    print(f"\nReplay complete:")
    for i, (loss, fw) in enumerate(zip(res["epoch_losses"], res["weight_mean_history"])):
        print(f"  Epoch {i+1}: loss={loss:.4f}  <W>={fw:.4f}")

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

    # Short simulation
    n_steps = 1000
    for step in range(n_steps):
        t = step * 0.1
        I_syn   = W.compute_current(pop.spikes.astype(np.float32))
        spikes  = pop.step(t, I_base + I_syn)
        if step % 5 == 0:
            mapper.update(t, np.where(spikes)[0])

    t_v, vols = mapper.get_volume_trace()
    t_a, areas = mapper.get_area_trace()
    t_i, iso   = mapper.get_isoperimetric_trace()

    print(f"Valid snapshots   : {len(t_v)}")
    print(f"Volume  mean±std  : {vols.mean():.4f} ± {vols.std():.4f}")
    print(f"Area    mean±std  : {areas.mean():.4f} ± {areas.std():.4f}")
    print(f"Iso     mean±std  : {iso.mean():.4f} ± {iso.std():.4f}  (sphere=1)")

    # Minkowski sum sequence
    print("\nMinkowski Sum pairs (consecutive manifolds):")
    for k in range(-3, 0):
        verts = mapper.minkowski_sum_consecutive(k-1, k)
        if verts is not None:
            t1 = mapper.history[k-1].t
            t2 = mapper.history[k].t
            print(f"  M({t1:.1f}) ⊕ M({t2:.1f}) : {len(verts)} vertices")

    # Save geometry plots
    if len(t_v) > 1:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), facecolor="#0d0f14")
        fig.suptitle("Information Manifold Geometry", color="#7eb8f7", fontsize=12, fontweight='bold')

        for ax, (t_data, y_data, color, label) in zip(axes, [
            (t_v, vols,  "#f77b4d", "Volume V(t)"),
            (t_a, areas, "#cc88ff", "Surface Area A(t)"),
            (t_i, iso,   "#f7e94d", "Isoperimetric Ratio I(t)"),
        ]):
            ax.set_facecolor("#0d0f14")
            ax.plot(t_data, y_data, color=color, lw=1.5)
            ax.fill_between(t_data, 0, y_data, color=color, alpha=0.2)
            ax.set_ylabel(label, color=color, fontsize=9)
            ax.tick_params(colors="#c8d0e0")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2d48")
            ax.grid(True, alpha=0.3, color="#1e2d48")

        axes[-1].set_xlabel("Time (ms)", color="#c8d0e0")
        plt.tight_layout()
        Path("outputs").mkdir(exist_ok=True)
        fig.savefig("outputs/manifold_geometry.png", dpi=150, bbox_inches='tight', facecolor="#0d0f14")
        plt.close(fig)
        print("\nSaved outputs/manifold_geometry.png")

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
    I_ext   = np.array([1.0 + 0.3*i for i in range(N)])   # gradient of input

    for step in range(n_steps):
        t = step * lp.dt
        pop.step(t, I_ext)
        V_trace[:, step] = pop.V.copy()

    fig, axes = plt.subplots(5, 1, figsize=(14, 8), sharex=True, facecolor="#0d0f14")
    fig.suptitle("LIF Membrane Potential  V(t)\n"
                 r"τ_m dV/dt = -(V - V_rest) + R_m I(t)",
                 color="#7eb8f7", fontsize=11, fontweight='bold')
    t_ax = np.arange(n_steps) * lp.dt

    colors = ["#4d9ef7","#f77b4d","#7df7a0","#f7e94d","#cc88ff"]
    for k, (ax, col) in enumerate(zip(axes, colors)):
        ax.set_facecolor("#0d0f14")
        ax.plot(t_ax, V_trace[k], color=col, lw=0.8)
        ax.axhline(lp.V_thresh, color='white', lw=0.5, linestyle='--', alpha=0.4)
        ax.axhline(lp.V_rest,   color='gray',  lw=0.5, linestyle=':',  alpha=0.4)
        ax.set_ylabel(f"N{k}", color=col, fontsize=8)
        ax.set_ylim(lp.V_reset - 5, lp.V_thresh + 5)
        ax.tick_params(colors="#c8d0e0", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2d48")
        ax.grid(True, alpha=0.25, color="#1e2d48")

    axes[-1].set_xlabel("Time (ms)", color="#c8d0e0")
    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/membrane_potential.png", dpi=150, bbox_inches='tight', facecolor="#0d0f14")
    plt.close(fig)
    print("Saved outputs/membrane_potential.png")


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)

    example_4_membrane_potential_trace()
    example_1_basic_simulation()
    example_2_synthetic_replay()
    example_3_manifold_analysis()

    print("\n" + "="*60)
    print("  All examples complete. Check outputs/ directory.")
    print("="*60)
