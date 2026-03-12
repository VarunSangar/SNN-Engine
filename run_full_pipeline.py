import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time
import types
import streamlit as st

# ── 1. BOOTSTRAP PATH ─────────────────────────────────────────────────────────
# We define root_dir immediately and globally to prevent NameErrors
file_path = Path(__file__).resolve()
root_dir = file_path.parent
if root_dir not in sys.path:
    sys.path.insert(0, str(root_dir))

# ── 2. MODULE REGISTRATION (Python 3.14 Fix) ──────────────────────────────────
# This prevents the 'NoneType' object has no attribute '__dict__' error
# by pre-registering the flat files in sys.modules
module_names = [
    'lif_model', 'synaptic_matrix', 'stdp', 
    'manifold_mapper', 'data_loader', 'simulation_engine'
]
for name in module_names:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

# ── 3. SECURE NATIVE IMPORTS ──────────────────────────────────────────────────
try:
    # Import as direct modules because the structure is FLAT
    import simulation_engine
    import lif_model
    import synaptic_matrix
    import stdp as stdp_mod
    import manifold_mapper as manifold
    import data_loader

    # Map the classes to the names used in your existing code
    SimulationEngine = simulation_engine.SimulationEngine
    SimConfig        = simulation_engine.SimConfig
    LIFPopulation    = lif_model.LIFPopulation
    LIFParams        = lif_model.LIFParams
    SparseWeightMatrix = synaptic_matrix.SparseWeightMatrix
    SynapseParams    = synaptic_matrix.SynapseParams
    STDPEngine       = stdp_mod.STDPEngine
    STDPParams       = stdp_mod.STDPParams
    NeuronEmbedding  = getattr(manifold, "NeuronEmbedding", None)
    ManifoldMapper   = manifold.ManifoldMapper
    SyntheticDataGenerator = data_loader.SyntheticDataGenerator
    SpikeDataLoader        = data_loader.SpikeDataLoader
    ReplayEngine           = data_loader.ReplayEngine
    
    _IMPORTS_OK = True
except Exception as e:
    _IMPORTS_OK = False
    _IMPORT_ERR = str(e)

# ── 4. UI CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Axiom-Neuro",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE & CSS
# ══════════════════════════════════════════════════════════════════════════════
 
DARK     = "#070a10"
MID      = "#0d1117"
PANEL    = "#111827"
BORDER   = "#1f2d45"
ACCENT   = "#00d4ff"
ACCENT2  = "#7b61ff"
GREEN    = "#00ff9d"
ORANGE   = "#ff6b35"
YELLOW   = "#ffd93d"
PINK     = "#ff4d8d"
TEXT     = "#cdd9f0"
TEXTDIM  = "#4a607a"
 
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
 
/* ── Reset ── */
html, body, [class*="css"] {{
    font-family: 'Syne', sans-serif;
    background: {DARK};
    color: {TEXT};
}}
.stApp {{ background: {DARK}; }}
.main .block-container {{ padding: 1.5rem 2rem; max-width: 100%; }}
 
/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {MID} !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
 
/* ── Metric cards ── */
.axm-card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'Space Mono', monospace;
    position: relative;
    overflow: hidden;
}}
.axm-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent-color, {ACCENT});
}}
.axm-label {{
    font-size: 9px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: {TEXTDIM};
    margin-bottom: 6px;
}}
.axm-value {{
    font-size: 24px;
    font-weight: 700;
    color: var(--accent-color, {ACCENT});
    line-height: 1;
}}
.axm-sub {{
    font-size: 10px;
    color: {TEXTDIM};
    margin-top: 4px;
}}
 
/* ── Section header ── */
.axm-section {{
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: {ACCENT};
    border-bottom: 1px solid {BORDER};
    padding-bottom: 8px;
    margin: 24px 0 14px 0;
}}
 
/* ── Page title ── */
.axm-title {{
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT2} 50%, {GREEN} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
}}
.axm-subtitle {{
    color: {TEXTDIM};
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.5px;
    margin-top: 6px;
}}
 
/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {MID};
    border-radius: 8px;
    padding: 4px;
    border: 1px solid {BORDER};
    gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {TEXTDIM} !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
}}
.stTabs [aria-selected="true"] {{
    background: {BORDER} !important;
    color: {ACCENT} !important;
}}
 
/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {ACCENT}22, {ACCENT2}22) !important;
    border: 1px solid {ACCENT}66 !important;
    color: {ACCENT} !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{
    border-color: {ACCENT} !important;
    background: linear-gradient(135deg, {ACCENT}44, {ACCENT2}44) !important;
    box-shadow: 0 0 20px {ACCENT}33 !important;
}}
 
/* ── Sliders & selects ── */
.stSlider label, .stSelectbox label, .stNumberInput label,
.stRadio label, .stCheckbox label {{
    color: {TEXTDIM} !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}}
.stSlider [data-baseweb="slider"] {{ color: {ACCENT} !important; }}
div[data-baseweb="select"] > div {{
    background: {PANEL} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
}}
 
/* ── Alert / info ── */
.stAlert {{ background: {PANEL} !important; border-color: {BORDER} !important; }}
.stSuccess {{ border-left-color: {GREEN} !important; }}
.stInfo    {{ border-left-color: {ACCENT} !important; }}
.stWarning {{ border-left-color: {ORANGE} !important; }}
 
/* ── Dataframe ── */
.stDataFrame {{ background: {PANEL} !important; }}
 
/* ── Horizontal rule ── */
hr {{ border-color: {BORDER}; margin: 16px 0; }}
 
/* ── Code ── */
code {{ background: {PANEL} !important; color: {GREEN} !important;
        font-family: 'Space Mono', monospace !important; }}
 
/* ── Progress ── */
.stProgress > div > div {{ background: {ACCENT} !important; }}
 
/* ── Expander ── */
.streamlit-expanderHeader {{
    background: {PANEL} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
}}
</style>
""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME HELPER
# ══════════════════════════════════════════════════════════════════════════════
 
def _plotly_layout(**kw):
    base = dict(
        paper_bgcolor=DARK, plot_bgcolor=MID,
        font=dict(family="Space Mono, monospace", color=TEXT, size=10),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT),
    )
    base.update(kw)
    return base
 
def _3d_scene():
    return dict(
        xaxis=dict(backgroundcolor=MID, gridcolor=BORDER, showbackground=True, color=TEXT),
        yaxis=dict(backgroundcolor=MID, gridcolor=BORDER, showbackground=True, color=TEXT),
        zaxis=dict(backgroundcolor=MID, gridcolor=BORDER, showbackground=True, color=TEXT),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)),
        aspectmode="cube",
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  IMPORT GUARD
# ══════════════════════════════════════════════════════════════════════════════
 
if not _IMPORTS_OK:
    st.markdown('<p class="axm-title">AXIOM-NEURO</p>', unsafe_allow_html=True)
    st.error(f"**Import failed:** `{_IMPORT_ERR}`")
    st.info("Ensure `axiom_neuro/` package is in the same directory as this file.")
    st.stop()
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION RUNNER  (cached by config hash)
# ══════════════════════════════════════════════════════════════════════════════
 
@st.cache_data(show_spinner=False)
def run_simulation(
    n_neurons, duration_ms, dt, conn_density, base_current,
    current_noise, tau_m, noise_sigma, stdp_on, homeostasis,
    r_target, a_plus, a_minus, embedding, seed,
):
    cfg = SimConfig(
        n_neurons          = n_neurons,
        duration_ms        = duration_ms,
        dt                 = dt,
        conn_density       = conn_density,
        base_current       = base_current,
        current_noise      = current_noise,
        tau_m              = tau_m,
        noise_sigma        = noise_sigma,
        stdp_enabled       = stdp_on,
        homeostasis        = homeostasis,
        r_target           = r_target,
        A_plus             = a_plus,
        A_minus            = a_minus,
        manifold_enabled   = True,
        embedding_strategy = embedding,
        manifold_min_fire  = 4,
        record_interval    = max(1, int(0.5 / dt)),   # every 0.5 ms sim-time
        save_raster        = False,
        save_dashboard     = False,
        seed               = seed,
    )
 
    # Run engine
    sim    = SimulationEngine(cfg)
    result = sim.run(verbose=False)
 
    # Pull manifold data
    t_v, vols  = sim.mapper.get_volume_trace()
    t_a, areas = sim.mapper.get_area_trace()
    t_i, iso   = sim.mapper.get_isoperimetric_trace()
 
    # Last valid manifold snapshot
    snap = sim.mapper.latest_valid()
    mink = sim.mapper.minkowski_sum_consecutive()
 
    return dict(
        result   = result,
        t_v=t_v, vols=vols, areas=areas, iso=iso,
        snap=snap, mink=mink,
        sim_n_synapses = sim.W.n_synapses,
        sim_mem_kb     = sim.W.memory_bytes / 1024,
    )
 
 
@st.cache_data(show_spinner=False)
def run_replay(n_neurons, freq_hz, base_rate, duration_ms, n_epochs, a_plus, a_minus, seed):
    gen  = SyntheticDataGenerator()
    data = gen.oscillatory_spikes(
        n_neurons=n_neurons, duration_ms=duration_ms,
        freq_hz=freq_hz, base_rate=base_rate, dt=0.1, seed=seed,
    )
    lp   = LIFParams(n_neurons=n_neurons, dt=0.1, noise_sigma=0.5)
    sp   = SynapseParams(n_pre=n_neurons, n_post=n_neurons, density=0.08, w_init_mean=0.3)
    pop  = LIFPopulation(lp, seed=seed)
    W    = SparseWeightMatrix(sp, seed=seed)
    stdp = STDPEngine(STDPParams(A_plus=a_plus, A_minus=a_minus, homeostasis_enabled=True, r_target=base_rate), W, n_neurons, n_neurons, dt=0.1)
    eng  = ReplayEngine(pop, W, stdp)
    res  = eng.run(data, n_epochs=n_epochs, base_current=1.5, replay_gain=4.0, verbose=False)
    res["weight_data"] = W._W_csr.data.copy()
    res["n_spikes"]    = data.n_spikes
    res["mean_fr"]     = data.mean_firing_rate
    return res
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
 
def fig_raster(spike_times, spike_neurons, n_neurons, duration_ms):
    max_show = min(n_neurons, 600)
    mask     = spike_neurons < max_show
    t, n     = spike_times[mask], spike_neurons[mask]
 
    # Bin for rate
    bin_ms   = max(2.0, duration_ms / 200)
    bins     = np.arange(0, duration_ms + bin_ms, bin_ms)
    cnt, edges = np.histogram(spike_times, bins=bins)
    rate     = cnt / (bin_ms * 1e-3 * n_neurons)
    centers  = 0.5 * (edges[:-1] + edges[1:])
 
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)
 
    # Raster scatter
    fig.add_trace(go.Scattergl(
        x=t, y=n, mode="markers",
        marker=dict(size=1.5, color=t, colorscale=[
            [0, ACCENT2], [0.5, ACCENT], [1, GREEN]
        ], opacity=0.7),
        name="Spikes", hoverinfo="skip",
    ), row=1, col=1)
 
    # Rate fill
    fig.add_trace(go.Scatter(
        x=centers, y=rate, mode="lines",
        fill="tozeroy", fillcolor=f"{GREEN}22",
        line=dict(color=GREEN, width=1.5),
        name="Rate (Hz)",
    ), row=2, col=1)
 
    fig.update_layout(
        **_plotly_layout(
            height=460,
            title=dict(text="SPIKE RASTER  ·  POPULATION FIRING RATE",
                       font=dict(size=11, color=ACCENT), x=0),
            showlegend=False,
        ),
        xaxis2=dict(title="Time (ms)", gridcolor=BORDER, color=TEXT),
        yaxis=dict(title="Neuron", gridcolor=BORDER, color=TEXT),
        yaxis2=dict(title="Hz", gridcolor=BORDER, color=TEXT),
    )
    return fig
 
 
def fig_manifold_3d(snap, mink=None):
    traces = []
    if snap and snap.valid and snap.hull_vertices is not None:
        verts = snap.hull_vertices
        tri   = snap.hull_simplices
 
        traces.append(go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2],
            i=tri[:,0], j=tri[:,1], k=tri[:,2],
            color=ACCENT, opacity=0.30, name="M(t)",
            flatshading=True,
            lighting=dict(ambient=0.7, diffuse=0.9, specular=0.4),
            lightposition=dict(x=1, y=1, z=1),
        ))
        # Wireframe
        ex, ey, ez = [], [], []
        for face in tri:
            for a, b in [(0,1),(1,2),(0,2)]:
                pa, pb = verts[face[a]], verts[face[b]]
                ex += [pa[0],pb[0],None]; ey += [pa[1],pb[1],None]; ez += [pa[2],pb[2],None]
        traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color=ACCENT, width=1.2),
            showlegend=False, hoverinfo="skip",
        ))
        if snap.centroid is not None:
            c = snap.centroid
            traces.append(go.Scatter3d(
                x=[c[0]], y=[c[1]], z=[c[2]],
                mode="markers",
                marker=dict(size=10, color=YELLOW, symbol="diamond"),
                name="Centroid",
            ))
 
    if mink is not None:
        traces.append(go.Scatter3d(
            x=mink[:,0], y=mink[:,1], z=mink[:,2],
            mode="markers",
            marker=dict(size=2.5, color=GREEN, opacity=0.6),
            name="M(t)⊕M(t+1)",
        ))
 
    annotations = []
    if snap and snap.valid:
        annotations.append(dict(
            text=(f"<b>t = {snap.t:.1f} ms</b><br>"
                  f"firing: {snap.n_firing}<br>"
                  f"Vol: {snap.volume:.4f}<br>"
                  f"Area: {snap.area:.4f}<br>"
                  f"Iso: {snap.isoperimetric:.3f}"),
            align="left", showarrow=False,
            xref="paper", yref="paper", x=0.01, y=0.98,
            font=dict(size=9, color=TEXT, family="Space Mono, monospace"),
            bgcolor=f"{PANEL}ee", bordercolor=BORDER, borderwidth=1,
        ))
 
    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor=DARK,
        scene=_3d_scene(),
        title=dict(text="INFORMATION MANIFOLD  M(t) = conv(Φ(F(t)))",
                   font=dict(size=11, color=ACCENT2), x=0),
        margin=dict(l=0, r=0, t=40, b=0),
        height=480,
        annotations=annotations,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=TEXT)),
    )
    return fig
 
 
def fig_geometry_traces(t_v, vols, areas, iso):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06,
                        subplot_titles=["Volume V(t)", "Surface Area A(t)", "Isoperimetric Ratio I(t)"])
 
    for row, (td, yd, col, name) in enumerate([
        (t_v, vols,  ORANGE, "V(t)"),
        (t_v, areas, ACCENT2, "A(t)"),
        (t_v, iso,   YELLOW, "I(t)"),
    ], 1):
        fig.add_trace(go.Scatter(
            x=td, y=yd, mode="lines", name=name,
            fill="tozeroy", fillcolor=f"{col}18",
            line=dict(color=col, width=2),
        ), row=row, col=1)
 
    # Isoperimetric reference line
    if len(t_v) > 0:
        fig.add_hline(y=1.0, line=dict(color="white", width=0.8, dash="dot"),
                      row=3, col=1, annotation_text="sphere", annotation_font_color=TEXTDIM)
 
    fig.update_layout(
        **_plotly_layout(height=500, showlegend=False,
                         title=dict(text="MANIFOLD GEOMETRY OVER TIME",
                                    font=dict(size=11, color=ACCENT2), x=0)),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=BORDER, color=TEXT, row=i, col=1)
        fig.update_yaxes(gridcolor=BORDER, color=TEXT, row=i, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=3, col=1)
    fig.update_layout(paper_bgcolor=DARK, plot_bgcolor=MID)
 
    # Fix subplot title colors
    for ann in fig.layout.annotations:
        ann.font.color = TEXTDIM
        ann.font.size  = 9
 
    return fig
 
 
def fig_weights(weight_data, epoch_losses, weight_history):
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Weight Distribution", "Loss per Epoch", "Mean Weight Evolution"])
 
    if weight_data is not None and len(weight_data) > 0:
        counts, edges = np.histogram(weight_data, bins=50)
        fig.add_trace(go.Bar(
            x=0.5*(edges[:-1]+edges[1:]), y=counts,
            marker_color=ACCENT, name="Weights", opacity=0.8,
        ), row=1, col=1)
 
    if epoch_losses:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(epoch_losses)+1)), y=epoch_losses,
            mode="lines+markers", line=dict(color=PINK, width=2),
            marker=dict(size=7, color=PINK), name="Loss",
        ), row=1, col=2)
 
    if weight_history:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(weight_history)+1)), y=weight_history,
            mode="lines+markers", line=dict(color=GREEN, width=2),
            marker=dict(size=7, color=GREEN), name="<W>",
        ), row=1, col=3)
 
    fig.update_layout(
        **_plotly_layout(height=300, showlegend=False,
                         title=dict(text="STDP WEIGHT LEARNING",
                                    font=dict(size=11, color=PINK), x=0)),
    )
    fig.update_layout(paper_bgcolor=DARK, plot_bgcolor=MID)
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=BORDER, color=TEXT, row=1, col=i)
        fig.update_yaxes(gridcolor=BORDER, color=TEXT, row=1, col=i)
    for ann in fig.layout.annotations:
        ann.font.color = TEXTDIM; ann.font.size = 9
    return fig
 
 
def fig_membrane_potential(n_neurons=8, n_steps=1500, dt=0.05, I_grad_start=1.2, I_grad_end=5.5):
    lp  = LIFParams(n_neurons=n_neurons, dt=dt, noise_sigma=0.4, R_m=12.0)
    pop = LIFPopulation(lp, seed=42)
    I_ext = np.linspace(I_grad_start, I_grad_end, n_neurons)
 
    V_trace = np.zeros((n_neurons, n_steps))
    for step in range(n_steps):
        pop.step(step * dt, I_ext)
        V_trace[:, step] = pop.V.copy()
 
    t_ax = np.arange(n_steps) * dt
    cols = [ACCENT, ACCENT2, GREEN, ORANGE, YELLOW, PINK, "#aaffcc", "#ffaacc"]
 
    fig = go.Figure()
    offset_step = 20.0
    for k in range(n_neurons):
        offset = k * offset_step
        fig.add_trace(go.Scatter(
            x=t_ax, y=V_trace[k] + offset,
            mode="lines", name=f"N{k}  I={I_ext[k]:.1f}nA",
            line=dict(color=cols[k % len(cols)], width=1.0),
            hovertemplate=f"Neuron {k}  I={I_ext[k]:.1f}nA<br>V=%{{y:.1f}}mV<br>t=%{{x:.1f}}ms",
        ))
    # Threshold lines
    for k in range(n_neurons):
        fig.add_hline(
            y=lp.V_thresh + k * offset_step,
            line=dict(color="white", width=0.4, dash="dot"),
            opacity=0.2,
        )
 
    fig.update_layout(
        **_plotly_layout(
            height=520,
            title=dict(
                text=("MEMBRANE POTENTIAL  V(t)  ·  "
                      "τ_m dV/dt = −(V−V_rest) + R_m I(t)  "
                      "[offset for clarity]"),
                font=dict(size=10, color=ACCENT), x=0,
            ),
            xaxis_title="Time (ms)",
            yaxis_title="V + offset (mV)",
            showlegend=True,
        )
    )
    return fig
 
 
def metric_card(label, value, sub="", color=ACCENT):
    return f"""
    <div class="axm-card" style="--accent-color:{color}">
        <div class="axm-label">{label}</div>
        <div class="axm-value">{value}</div>
        {"" if not sub else f'<div class="axm-sub">{sub}</div>'}
    </div>"""
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
 
with st.sidebar:
    st.markdown(
        '<div style="font-family:Space Mono,monospace;font-size:11px;'
        f'letter-spacing:3px;color:{ACCENT};text-transform:uppercase;'
        f'padding:8px 0 16px">⬡  AXIOM-NEURO</div>',
        unsafe_allow_html=True,
    )
 
    st.markdown(f'<div class="axm-section">Network</div>', unsafe_allow_html=True)
    n_neurons    = st.slider("Neurons",         100, 2000, 500, 50)
    duration_ms  = st.slider("Duration (ms)",    50,  500, 200, 25)
    conn_density = st.slider("Conn. Density",  0.01, 0.20, 0.08, 0.01)
 
    st.markdown(f'<div class="axm-section">LIF Biophysics</div>', unsafe_allow_html=True)
    tau_m        = st.slider("τ_m (ms)",   5.0, 50.0, 20.0, 1.0)
    base_current = st.slider("I_base (nA)",0.5,  8.0,  3.0, 0.1)
    noise_sigma  = st.slider("σ_noise",    0.0,  5.0,  1.5, 0.1)
 
    st.markdown(f'<div class="axm-section">STDP Learning</div>', unsafe_allow_html=True)
    stdp_on      = st.checkbox("Enable STDP",     value=True)
    homeostasis  = st.checkbox("Homeostatic Plasticity", value=True)
    r_target     = st.slider("Target Rate (Hz)", 1.0, 30.0, 10.0, 0.5)
    a_plus       = st.slider("A+  (LTP)",  0.001, 0.05, 0.010, 0.001, format="%.3f")
    a_minus      = st.slider("A−  (LTD)",  0.001, 0.05, 0.011, 0.001, format="%.3f")
 
    st.markdown(f'<div class="axm-section">Geometry</div>', unsafe_allow_html=True)
    embedding    = st.selectbox("Embedding", ["random_sphere","toroidal","grid"], index=0)
 
    st.markdown(f'<div class="axm-section">Run</div>', unsafe_allow_html=True)
    seed         = st.number_input("Random Seed", value=42, step=1)
    run_btn      = st.button("⬡  RUN SIMULATION", use_container_width=True)
 
    st.markdown("---")
    st.markdown(
        f'<div style="font-size:9px;color:{TEXTDIM};font-family:Space Mono,monospace;'
        f'line-height:1.8">Leaky Integrate-and-Fire<br>STDP · Homeostasis · BCM<br>'
        f'Minkowski Manifold Mapper<br>Sparse CSR Synaptic Matrix</div>',
        unsafe_allow_html=True,
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
 
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(
        '<p class="axm-title">AXIOM-NEURO</p>'
        '<p class="axm-subtitle">SPIKING NEURAL NETWORK · INFORMATION GEOMETRY ENGINE</p>',
        unsafe_allow_html=True,
    )
with col_h2:
    st.markdown(
        f'<div style="text-align:right;font-family:Space Mono,monospace;'
        f'font-size:9px;color:{TEXTDIM};padding-top:8px;line-height:2">'
        f'τ_m dV/dt = −(V−V_rest) + R_m I(t)<br>'
        f'h_K(u) = sup{{u·x | x∈K}}<br>'
        f'M(t) = conv(Φ(F(t)))<br>'
        f'P⊕Q = {{p+q | p∈P, q∈Q}}'
        f'</div>',
        unsafe_allow_html=True,
    )
st.markdown("---")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
 
tab_sim, tab_manifold, tab_replay, tab_lif, tab_bench = st.tabs([
    "⬡  Simulation",
    "⬢  Information Manifold",
    "⟳  Replay & Learning",
    "∿  Membrane Dynamics",
    "▲  Benchmarks",
])
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Simulation
# ─────────────────────────────────────────────────────────────────────────────
with tab_sim:
    if run_btn or "sim_data" in st.session_state:
 
        if run_btn:
            with st.spinner("Running SNN simulation..."):
                t0 = time.perf_counter()
                data = run_simulation(
                    n_neurons, duration_ms, 0.1, conn_density, base_current,
                    1.0, tau_m, noise_sigma, stdp_on, homeostasis,
                    r_target, a_plus, a_minus, embedding, int(seed),
                )
                data["wall_time"] = time.perf_counter() - t0
                st.session_state["sim_data"] = data
 
        data   = st.session_state["sim_data"]
        result = data["result"]
 
        # ── Metric row ───────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        fr_mean = result._mean_fr()
 
        c1.markdown(metric_card("Neurons", f"{n_neurons:,}", "LIF population", ACCENT),         unsafe_allow_html=True)
        c2.markdown(metric_card("Spikes",  f"{len(result.spike_times):,}", f"{fr_mean:.1f} Hz mean", GREEN), unsafe_allow_html=True)
        c3.markdown(metric_card("Synapses", f"{data['sim_n_synapses']:,}", f"{data['sim_mem_kb']:.0f} KB CSR", ACCENT2), unsafe_allow_html=True)
        c4.markdown(metric_card("Events/s", f"{result.events_per_second:,.0f}", "syn events", ORANGE), unsafe_allow_html=True)
        c5.markdown(metric_card("Speed", f"{duration_ms/max(result.wall_time_s,1e-3):.0f}×", "vs real-time", YELLOW), unsafe_allow_html=True)
 
        st.markdown("")
 
        # ── Raster ───────────────────────────────────────────────────────────
        if len(result.spike_times) > 0:
            st.plotly_chart(
                fig_raster(result.spike_times, result.spike_neurons, n_neurons, duration_ms),
                use_container_width=True,
            )
        else:
            st.warning("No spikes recorded — try increasing I_base or σ_noise.")
 
        # ── Firing rate distribution ──────────────────────────────────────────
        fr_data = result.firing_rates
        fr_pos  = fr_data[fr_data > 0.1]
        if len(fr_pos) > 0:
            fig_fr = go.Figure()
            fig_fr.add_trace(go.Histogram(
                x=fr_pos, nbinsx=40,
                marker_color=ACCENT2, opacity=0.8, name="Firing Rate",
            ))
            if homeostasis:
                fig_fr.add_vline(x=r_target, line=dict(color=YELLOW, dash="dash", width=1.5),
                                  annotation_text=f"target {r_target}Hz",
                                  annotation_font_color=YELLOW, annotation_font_size=9)
            fig_fr.update_layout(
                **_plotly_layout(height=220,
                                  title=dict(text="FIRING RATE DISTRIBUTION",
                                             font=dict(size=10, color=ACCENT2), x=0),
                                  xaxis_title="Rate (Hz)", yaxis_title="Neurons",
                                  showlegend=False),
            )
            st.plotly_chart(fig_fr, use_container_width=True)
 
    else:
        # Default state — show LIF equation nicely
        st.markdown(
            f'<div style="text-align:center;padding:60px;color:{TEXTDIM};'
            f'font-family:Space Mono,monospace;font-size:12px;">'
            f'Configure parameters in the sidebar<br>and press<br><br>'
            f'<span style="color:{ACCENT};font-size:14px;">⬡  RUN SIMULATION</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Information Manifold
# ─────────────────────────────────────────────────────────────────────────────
with tab_manifold:
    if "sim_data" not in st.session_state:
        st.info("Run a simulation first to compute the Information Manifold.")
    else:
        data = st.session_state["sim_data"]
        t_v, vols, areas, iso = data["t_v"], data["vols"], data["areas"], data["iso"]
        snap, mink = data["snap"], data["mink"]
 
        st.markdown(
            f'<div style="font-family:Space Mono,monospace;font-size:10px;'
            f'color:{TEXTDIM};line-height:2;padding:8px 0 16px">'
            f'M(t) = conv({{ Φ(i) | i ∈ F(t) }})  ·  '
            f'F(t) = firing population at time t  ·  '
            f'h_{{P⊕Q}}(u) = h_P(u) + h_Q(u)</div>',
            unsafe_allow_html=True,
        )
 
        col3d, colg = st.columns([1, 1])
 
        with col3d:
            st.plotly_chart(fig_manifold_3d(snap, mink), use_container_width=True)
 
        with colg:
            if len(t_v) > 1:
                st.plotly_chart(fig_geometry_traces(t_v, vols, areas, iso), use_container_width=True)
            else:
                st.warning("Insufficient manifold snapshots — increase Duration or reduce Min Firing threshold.")
 
        # Manifold stats
        if len(vols) > 0:
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(metric_card("Snapshots",  f"{len(vols)}",       "valid hulls",      ACCENT),  unsafe_allow_html=True)
            m2.markdown(metric_card("Mean Vol",   f"{vols.mean():.4f}", "manifold volume",  ORANGE),  unsafe_allow_html=True)
            m3.markdown(metric_card("Mean Area",  f"{areas.mean():.4f}","surface area",     ACCENT2), unsafe_allow_html=True)
            m4.markdown(metric_card("Mean Iso",   f"{iso.mean():.4f}",  "sphere=1.000",     YELLOW),  unsafe_allow_html=True)
 
        if mink is not None:
            st.success(f"Minkowski Sum M(t)⊕M(t+1)  →  **{len(mink)} vertices**  "
                       f"(shown as green points in 3D view)")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Replay & Learning
# ─────────────────────────────────────────────────────────────────────────────
with tab_replay:
    st.markdown(
        f'<div class="axm-section">Synthetic Spike Replay  ·  STDP Weight Convergence</div>',
        unsafe_allow_html=True,
    )
 
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        rp_neurons  = st.slider("Neurons (Replay)", 100, 500, 200, 50)
        rp_duration = st.slider("Duration (ms)",     50, 300, 100, 25)
    with rc2:
        rp_freq   = st.slider("Oscillation (Hz)",  10.0, 100.0, 40.0, 5.0)
        rp_rate   = st.slider("Base Rate (Hz)",      1.0,  20.0,  8.0, 0.5)
    with rc3:
        rp_epochs = st.slider("Epochs",   1, 10, 3, 1)
        rp_seed   = st.number_input("Seed (Replay)", value=7, step=1)
 
    replay_btn = st.button("⟳  RUN REPLAY", use_container_width=False)
 
    if replay_btn or "replay_data" in st.session_state:
        if replay_btn:
            with st.spinner("Replaying spike train through SNN..."):
                rdata = run_replay(
                    rp_neurons, rp_freq, rp_rate, rp_duration,
                    rp_epochs, a_plus, a_minus, int(rp_seed),
                )
                st.session_state["replay_data"] = rdata
 
        rdata = st.session_state["replay_data"]
 
        r1, r2, r3 = st.columns(3)
        r1.markdown(metric_card("Spikes",     f"{rdata['n_spikes']:,}",             "in dataset",   ACCENT),  unsafe_allow_html=True)
        r2.markdown(metric_card("Mean FR",    f"{rdata['mean_fr']:.1f} Hz",         "oscillatory",  GREEN),   unsafe_allow_html=True)
        r3.markdown(metric_card("Final Loss", f"{rdata['epoch_losses'][-1]:.5f}",   "Hamming dist", PINK),     unsafe_allow_html=True)
 
        st.markdown("")
        st.plotly_chart(
            fig_weights(rdata["weight_data"], rdata["epoch_losses"], rdata["weight_mean_history"]),
            use_container_width=True,
        )
 
        # Epoch table
        with st.expander("📋  Epoch-by-Epoch Metrics"):
            import pandas as pd
            df = pd.DataFrame({
                "Epoch"     : list(range(1, len(rdata["epoch_losses"])+1)),
                "Loss"      : rdata["epoch_losses"],
                "Mean W"    : rdata["weight_mean_history"],
                "Mean FR"   : rdata["firing_rate_history"],
            })
            st.dataframe(df.style.format({"Loss":"{:.5f}","Mean W":"{:.4f}","Mean FR":"{:.2f}"}),
                         use_container_width=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Membrane Dynamics
# ─────────────────────────────────────────────────────────────────────────────
with tab_lif:
    st.markdown(
        f'<div class="axm-section">LIF Membrane Potential  ·  V(t) per neuron</div>',
        unsafe_allow_html=True,
    )
 
    mc1, mc2 = st.columns(2)
    with mc1:
        lif_n      = st.slider("Neurons to trace", 3, 12, 8, 1)
        lif_steps  = st.slider("Timesteps", 500, 4000, 1500, 250)
    with mc2:
        lif_i_lo   = st.slider("I_min (nA)", 0.1, 3.0, 1.0, 0.1)
        lif_i_hi   = st.slider("I_max (nA)", 1.0, 10.0, 5.5, 0.25)
 
    st.plotly_chart(
        fig_membrane_potential(lif_n, lif_steps, 0.05, lif_i_lo, lif_i_hi),
        use_container_width=True,
    )
 
    with st.expander("📐  LIF Equation Details"):
        st.latex(r"\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R_m \left[I(t) + \sigma \xi(t)\right]")
        st.latex(r"\text{Spike if } V \geq V_{thresh} \Rightarrow V \leftarrow V_{reset}, \quad \text{refrac} = t_{ref}")
        st.latex(r"\text{Euler step: } V[t+\Delta t] = V[t] + \frac{\Delta t}{\tau_m}\left(-(V[t]-V_{rest}) + R_m I[t]\right)")
 
        cols = st.columns(4)
        for col, (lbl, val) in zip(cols, [
            ("V_rest",   "−65 mV"),
            ("V_thresh", "−50 mV"),
            ("V_reset",  "−70 mV"),
            ("τ_m",      "20 ms"),
        ]):
            col.markdown(metric_card(lbl, val, color=ACCENT2), unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Benchmarks
# ─────────────────────────────────────────────────────────────────────────────
with tab_bench:
    st.markdown(
        f'<div class="axm-section">Performance Benchmarks  ·  Throughput Proof</div>',
        unsafe_allow_html=True,
    )
 
    bench_btn = st.button("▲  RUN QUICK BENCHMARK", use_container_width=False)
 
    if bench_btn or "bench_data" in st.session_state:
        if bench_btn:
            results_b = {}
            prog = st.progress(0, "Running LIF kernel...")
            t0 = time.perf_counter()
            N_b, S_b = 10_000, 500
            params_b = LIFParams(n_neurons=N_b, dt=0.1)
            pop_b    = LIFPopulation(params_b, seed=0)
            I_b      = np.ones(N_b) * 3.0
            for _ in range(S_b):
                pop_b.step(0.0, I_b)
            lif_dt = time.perf_counter() - t0
            results_b["lif_ns"]     = N_b * S_b / lif_dt
            results_b["lif_time"]   = lif_dt
            prog.progress(33, "Running sparse MatVec...")
 
            from axiom_neuro.core.synaptic_matrix import SparseWeightMatrix, SynapseParams
            sp_b = SynapseParams(n_pre=N_b, n_post=N_b, density=0.005)
            W_b  = SparseWeightMatrix(sp_b)
            spk  = np.zeros(N_b, dtype=np.float32)
            spk[np.random.choice(N_b, 100)] = 1.0
            R = 500
            t0 = time.perf_counter()
            for _ in range(R):
                W_b.compute_current(spk)
            mv_dt = time.perf_counter() - t0
            results_b["synops"]   = W_b.n_synapses * spk.mean() * R / mv_dt
            results_b["nnz"]      = W_b.n_synapses
            results_b["mem_kb"]   = W_b.memory_bytes / 1024
            results_b["dense_gb"] = (N_b**2 * 4) / 1024**3
            prog.progress(66, "Running full simulation...")
 
            cfg_b = SimConfig(n_neurons=2000, duration_ms=50.0, dt=0.1,
                              conn_density=0.02, stdp_enabled=False,
                              manifold_enabled=False, save_raster=False, save_dashboard=False)
            sim_b = SimulationEngine(cfg_b)
            t0    = time.perf_counter()
            res_b = sim_b.run(verbose=False)
            sim_dt = time.perf_counter() - t0
            results_b["sim_evts"]  = res_b.events_per_second
            results_b["sim_speed"] = 50.0 / sim_dt
            prog.progress(100, "Done.")
            prog.empty()
 
            st.session_state["bench_data"] = results_b
 
        bd = st.session_state["bench_data"]
 
        b1, b2, b3, b4 = st.columns(4)
        b1.markdown(metric_card("LIF Throughput", f"{bd['lif_ns']/1e6:.1f}M",    "neuron-steps/sec", ACCENT), unsafe_allow_html=True)
        b2.markdown(metric_card("SynOps/sec",     f"{bd['synops']/1e6:.2f}M",    "sparse MatVec",    GREEN),  unsafe_allow_html=True)
        b3.markdown(metric_card("Sim Speed",      f"{bd['sim_speed']:.1f}×",     "vs real-time",     ORANGE), unsafe_allow_html=True)
        b4.markdown(metric_card("Memory Saving",  f"{bd['dense_gb']:.1f}GB→{bd['mem_kb']:.0f}KB", "dense vs CSR", YELLOW), unsafe_allow_html=True)
 
        st.markdown("")
 
        # Throughput bar chart
        fig_bm = go.Figure()
        cats   = ["LIF kernel\n(neuron-steps/s)", "Sparse MatVec\n(synops/s)", "Full Sim\n(events/s)"]
        vals   = [bd["lif_ns"], bd["synops"], bd["sim_evts"]]
        cols_b = [ACCENT, GREEN, ORANGE]
 
        fig_bm.add_trace(go.Bar(
            x=cats, y=vals,
            marker_color=cols_b, opacity=0.85,
            text=[f"{v/1e6:.2f}M" for v in vals],
            textposition="outside",
            textfont=dict(color=TEXT, size=11, family="Space Mono, monospace"),
        ))
        fig_bm.add_hline(y=1e7, line=dict(color=PINK, dash="dash", width=1.5),
                          annotation_text="10⁷ target", annotation_font_color=PINK,
                          annotation_font_size=10)
        fig_bm.update_layout(
            **_plotly_layout(
                height=380,
                title=dict(text="THROUGHPUT BENCHMARK  (log scale)",
                           font=dict(size=11, color=ACCENT), x=0),
                yaxis_type="log",
                yaxis_title="Operations / second",
                showlegend=False,
            )
        )
        st.plotly_chart(fig_bm, use_container_width=True)
 
        with st.expander("📋  Detailed Benchmark Report"):
            st.markdown(f"""
```
LIF kernel
  Wall time      : {bd['lif_time']:.3f} s  (10k neurons × 500 steps)
  Throughput     : {bd['lif_ns']/1e6:.2f}M neuron-steps/sec
 
Sparse MatVec (CSR)
  Synapses (nnz) : {bd['nnz']:,}
  Memory (CSR)   : {bd['mem_kb']:.1f} KB
  Dense W would  : {bd['dense_gb']:.2f} GB   ({bd['dense_gb']*1024**2/bd['mem_kb']:.0f}× larger)
  Throughput     : {bd['synops']/1e6:.2f}M synops/sec
 
Full simulation (2000 neurons)
  Speed          : {bd['sim_speed']:.1f}× real-time
  Events/sec     : {bd['sim_evts']:,.0f}
```""")
 
    else:
        st.markdown(
            f'<div style="color:{TEXTDIM};font-family:Space Mono,monospace;'
            f'font-size:11px;padding:30px 0;">Press RUN QUICK BENCHMARK to measure throughput.</div>',
            unsafe_allow_html=True,
        )
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
 
st.markdown("---")
st.markdown(
    f'<div style="text-align:center;font-family:Space Mono,monospace;'
    f'font-size:9px;color:{TEXTDIM};letter-spacing:2px;padding:8px 0">'
    f'AXIOM-NEURO  ·  LIF + STDP + HOMEOSTASIS  ·  '
    f'MINKOWSKI–CONVEX-HULL INFORMATION MANIFOLD  ·  SPARSE CSR SYNAPTIC MATRIX'
    f'</div>',
    unsafe_allow_html=True,
)
 
