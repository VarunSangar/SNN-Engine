"""
axiom_neuro/visualization/plotter.py
======================================
Real-Time Visualization Suite
================================

Provides:
  1. RasterPlot     — spike raster with firing rate overlay
  2. ManifoldViewer — 3-D convex hull manifold (Plotly)
  3. NetworkDashboard — combined live matplotlib dashboard
  4. ManifoldTimelapse — animated manifold evolution
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Optional, List, Tuple, TYPE_CHECKING
import io

if TYPE_CHECKING:
    from axiom_neuro.geometry.manifold_mapper import ManifoldSnapshot, ManifoldMapper

# ── Colour palette ─────────────────────────────────────────────────────────
DARK_BG    = "#0d0f14"
GRID_COL   = "#1e2d48"
TEXT_COL   = "#c8d0e0"
BLUE       = "#4d9ef7"
ORANGE     = "#f77b4d"
GREEN      = "#7df7a0"
YELLOW     = "#f7e94d"
PURPLE     = "#cc88ff"
CYAN       = "#4dfff7"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    DARK_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         9,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Raster Plot
# ─────────────────────────────────────────────────────────────────────────────

class RasterPlot:
    """
    Spike raster plot + instantaneous population firing rate.

    Usage
    -----
    >>> raster = RasterPlot(n_neurons=1000, t_window=500.0)
    >>> fig = raster.plot(times_ms, neuron_ids, title="SNN Run")
    >>> fig.savefig("raster.png", dpi=150)
    """

    def __init__(self, n_neurons: int, t_window: float = 500.0, bin_ms: float = 5.0):
        self.N        = n_neurons
        self.t_window = t_window
        self.bin_ms   = bin_ms

    def plot(
        self,
        times:   np.ndarray,      # (S,) spike times in ms
        neurons: np.ndarray,      # (S,) neuron indices
        title:   str = "Spike Raster",
        figsize: Tuple = (14, 7),
        max_neurons_shown: int = 500,
        color_by_time: bool = True,
    ) -> plt.Figure:
        """
        Render the raster plot.

        Parameters
        ----------
        times   : spike time array (ms)
        neurons : neuron index array (matching length)
        """
        fig = plt.figure(figsize=figsize, facecolor=DARK_BG)
        gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)
        ax_raster = fig.add_subplot(gs[0])
        ax_rate   = fig.add_subplot(gs[1], sharex=ax_raster)
        ax_volt   = fig.add_subplot(gs[2], sharex=ax_raster)

        # ── Raster ──────────────────────────────────────────────────────────
        if len(times) > 0:
            # Subsample neurons for display
            shown = np.where(neurons < max_neurons_shown)[0]
            t_show = times[shown]
            n_show = neurons[shown]

            if color_by_time and len(t_show) > 0:
                t_min, t_max = t_show.min(), t_show.max()
                if t_max > t_min:
                    norm   = Normalize(vmin=t_min, vmax=t_max)
                    colors = plt.cm.plasma(norm(t_show))
                else:
                    colors = BLUE
            else:
                colors = BLUE

            ax_raster.scatter(
                t_show, n_show,
                s=1.2, c=colors, alpha=0.6, linewidths=0,
                rasterized=True
            )

        ax_raster.set_ylabel("Neuron ID", color=TEXT_COL)
        ax_raster.set_ylim(-0.5, min(max_neurons_shown, self.N) + 0.5)
        ax_raster.set_title(title, color=BLUE, fontsize=11, fontweight='bold', pad=8)
        ax_raster.grid(True, alpha=0.3)
        plt.setp(ax_raster.get_xticklabels(), visible=False)

        # ── Population firing rate ────────────────────────────────────────────
        if len(times) > 0 and times.max() > times.min():
            t_min, t_max = times.min(), times.max()
            bins = np.arange(t_min, t_max + self.bin_ms, self.bin_ms)
            counts, edges = np.histogram(times, bins=bins)
            rate = counts / (self.bin_ms * 1e-3 * self.N)   # Hz
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax_rate.fill_between(centers, 0, rate, color=GREEN, alpha=0.6, linewidth=0)
            ax_rate.plot(centers, rate, color=GREEN, linewidth=0.8)
            ax_rate.set_ylabel("Rate (Hz)", color=TEXT_COL)
            ax_rate.grid(True, alpha=0.3)
            plt.setp(ax_rate.get_xticklabels(), visible=False)

        # ── Neuron count per bin ───────────────────────────────────────────────
        if len(times) > 0 and times.max() > times.min():
            ax_volt.fill_between(centers, 0, counts, color=PURPLE, alpha=0.5, linewidth=0)
            ax_volt.set_xlabel("Time (ms)", color=TEXT_COL)
            ax_volt.set_ylabel("# Spikes", color=TEXT_COL)
            ax_volt.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_png(self, fig: plt.Figure, path: str, dpi: int = 150) -> None:
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=DARK_BG)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  3-D Manifold Viewer  (Plotly)
# ─────────────────────────────────────────────────────────────────────────────

def manifold_figure_plotly(
    snapshot: "ManifoldSnapshot",
    title:    str = "Information Manifold",
    show_minksum_verts: Optional[np.ndarray] = None,
) -> "go.Figure":
    """
    Build a Plotly 3-D figure of the convex-hull manifold.

    Parameters
    ----------
    snapshot            : ManifoldSnapshot with valid hull
    show_minksum_verts  : optional (V,3) vertices for the Minkowski sum overlay

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    traces = []
    layout_kwargs = dict(
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#0d0f14",
        font=dict(family="JetBrains Mono, monospace", color=TEXT_COL, size=10),
        title=dict(text=title, font=dict(color=BLUE, size=14)),
        scene=dict(
            xaxis=dict(backgroundcolor="#0d0f14", gridcolor=GRID_COL, showbackground=True),
            yaxis=dict(backgroundcolor="#0d0f14", gridcolor=GRID_COL, showbackground=True),
            zaxis=dict(backgroundcolor="#0d0f14", gridcolor=GRID_COL, showbackground=True),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=550,
    )

    if snapshot.valid and snapshot.hull_vertices is not None:
        verts = snapshot.hull_vertices
        tri   = snapshot.hull_simplices

        # Hull mesh
        traces.append(go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2],
            i=tri[:,0],   j=tri[:,1],   k=tri[:,2],
            color=BLUE, opacity=0.35, name="M(t)",
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.8),
            hoverinfo="name",
        ))

        # Wireframe
        ex, ey, ez = [], [], []
        for face in tri:
            for a, b in [(0,1),(1,2),(0,2)]:
                pa, pb = verts[face[a]], verts[face[b]]
                ex += [pa[0],pb[0],None]; ey += [pa[1],pb[1],None]; ez += [pa[2],pb[2],None]
        traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color=BLUE, width=1), name="Hull edges",
            showlegend=False, hoverinfo="skip",
        ))

        # Centroid
        if snapshot.centroid is not None:
            c = snapshot.centroid
            traces.append(go.Scatter3d(
                x=[c[0]], y=[c[1]], z=[c[2]],
                mode="markers",
                marker=dict(size=8, color=YELLOW, symbol="diamond"),
                name="Centroid",
            ))

    # Minkowski Sum overlay
    if show_minksum_verts is not None:
        sv = show_minksum_verts
        traces.append(go.Scatter3d(
            x=sv[:,0], y=sv[:,1], z=sv[:,2],
            mode="markers",
            marker=dict(size=3, color=GREEN, opacity=0.7),
            name="M(t)⊕M(t+1)",
        ))

    # Annotation box
    if snapshot.valid:
        layout_kwargs["annotations"] = [dict(
            text=(f"<b>t = {snapshot.t:.1f} ms</b><br>"
                  f"n_firing = {snapshot.n_firing}<br>"
                  f"Vol = {snapshot.volume:.4f}<br>"
                  f"Area = {snapshot.area:.4f}<br>"
                  f"Iso = {snapshot.isoperimetric:.3f}"),
            align="left", showarrow=False,
            xref="paper", yref="paper", x=0.01, y=0.99,
            font=dict(size=10, color=TEXT_COL, family="monospace"),
            bgcolor="rgba(10,13,20,0.8)",
            bordercolor=GRID_COL, borderwidth=1,
        )]

    fig = go.Figure(data=traces)
    fig.update_layout(**layout_kwargs)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Network Dashboard  (Matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

class NetworkDashboard:
    """
    Comprehensive static dashboard: raster + manifold metrics + weight dist.

    Usage
    -----
    >>> dash = NetworkDashboard()
    >>> fig = dash.render(sim_result)
    """

    def render(
        self,
        times:          np.ndarray,
        neurons:        np.ndarray,
        vol_times:      np.ndarray,
        volumes:        np.ndarray,
        areas:          np.ndarray,
        weight_data:    Optional[np.ndarray] = None,
        firing_rates:   Optional[np.ndarray] = None,
        title:          str = "Axiom-Neuro Dashboard",
        figsize:        Tuple = (18, 11),
    ) -> plt.Figure:

        fig = plt.figure(figsize=figsize, facecolor=DARK_BG)
        fig.suptitle(title, color=BLUE, fontsize=13, fontweight='bold', y=0.98)
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

        # ── 1. Raster ────────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :2])
        if len(times) > 0:
            shown = neurons < 500
            ax1.scatter(times[shown], neurons[shown], s=0.8, c=BLUE, alpha=0.5,
                        linewidths=0, rasterized=True)
        ax1.set_title("Spike Raster", color=BLUE, fontsize=9)
        ax1.set_xlabel("Time (ms)"); ax1.set_ylabel("Neuron ID")
        ax1.grid(True, alpha=0.3)

        # ── 2. Population rate ───────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, :2], sharex=ax1)
        if len(times) > 1:
            n_neurons = int(neurons.max() + 1) if len(neurons) else 1
            bin_ms = max(1.0, (times.max()-times.min()) / 200)
            bins = np.arange(times.min(), times.max()+bin_ms, bin_ms)
            cnt, edges = np.histogram(times, bins=bins)
            rate = cnt / (bin_ms*1e-3 * n_neurons)
            ctr  = 0.5*(edges[:-1]+edges[1:])
            ax2.fill_between(ctr, 0, rate, color=GREEN, alpha=0.6, linewidth=0)
            ax2.plot(ctr, rate, color=GREEN, lw=0.8)
        ax2.set_title("Pop. Firing Rate (Hz)", color=GREEN, fontsize=9)
        ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("Hz")
        ax2.grid(True, alpha=0.3)

        # ── 3. Manifold Volume ────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        if len(vol_times) > 0:
            ax3.plot(vol_times, volumes, color=ORANGE, lw=1.5)
            ax3.fill_between(vol_times, 0, volumes, color=ORANGE, alpha=0.25)
        ax3.set_title("Manifold Volume V(t)", color=ORANGE, fontsize=9)
        ax3.set_xlabel("Time (ms)"); ax3.set_ylabel("V")
        ax3.grid(True, alpha=0.3)

        # ── 4. Manifold Surface Area ──────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 2])
        if len(vol_times) > 0 and len(areas) > 0:
            ax4.plot(vol_times, areas, color=PURPLE, lw=1.5)
            ax4.fill_between(vol_times, 0, areas, color=PURPLE, alpha=0.25)
        ax4.set_title("Manifold Area A(t)", color=PURPLE, fontsize=9)
        ax4.set_xlabel("Time (ms)"); ax4.set_ylabel("A")
        ax4.grid(True, alpha=0.3)

        # ── 5. Weight distribution ────────────────────────────────────────────
        ax5 = fig.add_subplot(gs[2, 0])
        if weight_data is not None and len(weight_data) > 0:
            ax5.hist(weight_data, bins=50, color=CYAN, alpha=0.75, edgecolor='none')
        ax5.set_title("Weight Distribution", color=CYAN, fontsize=9)
        ax5.set_xlabel("Weight (nA)"); ax5.set_ylabel("Count")
        ax5.grid(True, alpha=0.3)

        # ── 6. Firing rate distribution ───────────────────────────────────────
        ax6 = fig.add_subplot(gs[2, 1])
        if firing_rates is not None and len(firing_rates) > 0:
            ax6.hist(firing_rates[firing_rates > 0], bins=40, color=GREEN,
                     alpha=0.75, edgecolor='none')
        ax6.set_title("Firing Rate Dist.", color=GREEN, fontsize=9)
        ax6.set_xlabel("Rate (Hz)"); ax6.set_ylabel("Count")
        ax6.grid(True, alpha=0.3)

        # ── 7. Isoperimetric ratio ────────────────────────────────────────────
        ax7 = fig.add_subplot(gs[2, 2])
        if len(vol_times) > 1 and len(volumes) > 1 and len(areas) > 1:
            iso = np.where(
                areas > 1e-12,
                36 * np.pi * volumes**2 / (areas**3 + 1e-12),
                0.0
            )
            ax7.plot(vol_times, iso, color=YELLOW, lw=1.2)
            ax7.axhline(1.0, color='white', lw=0.6, linestyle='--', alpha=0.5)
        ax7.set_title("Isoperimetric Ratio", color=YELLOW, fontsize=9)
        ax7.set_xlabel("Time (ms)"); ax7.set_ylabel("I(t)")
        ax7.grid(True, alpha=0.3)

        return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Manifold Timelapse
# ─────────────────────────────────────────────────────────────────────────────

def save_manifold_timelapse(
    mapper:   "ManifoldMapper",
    path:     str,
    n_frames: int = 30,
    figsize:  Tuple = (8, 6),
    dpi:      int  = 100,
) -> None:
    """
    Save an animated GIF of the evolving information manifold.
    Requires matplotlib and Pillow.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    valid_snaps = [s for s in mapper.history if s.valid]
    if len(valid_snaps) < 2:
        print("Not enough valid snapshots for animation.")
        return

    # Sample uniformly
    idx    = np.linspace(0, len(valid_snaps)-1, n_frames, dtype=int)
    frames = [valid_snaps[i] for i in idx]

    fig   = plt.figure(figsize=figsize, facecolor=DARK_BG)
    ax    = fig.add_subplot(111, projection='3d')

    def draw_frame(snap):
        ax.cla()
        ax.set_facecolor(DARK_BG)
        ax.set_title(f"Manifold  t={snap.t:.1f}ms  V={snap.volume:.3f}",
                     color=BLUE, fontsize=10)
        if snap.hull_vertices is not None:
            verts = snap.hull_vertices
            tri   = snap.hull_simplices
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            mesh = Poly3DCollection(
                verts[tri],
                alpha=0.3, facecolor=BLUE, edgecolor=BLUE, linewidth=0.4
            )
            ax.add_collection3d(mesh)
            ax.scatter(verts[:,0], verts[:,1], verts[:,2],
                       s=20, c=YELLOW, alpha=0.8, depthshade=True)
            lim = 1.5
            ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_xlabel("x", color=TEXT_COL)
        ax.set_ylabel("y", color=TEXT_COL)
        ax.set_zlabel("z", color=TEXT_COL)

    def animate(i):
        draw_frame(frames[i])

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=80)
    try:
        writer = PillowWriter(fps=12)
        anim.save(path, writer=writer, dpi=dpi)
        print(f"Saved manifold timelapse → {path}")
    except Exception as e:
        print(f"Animation save failed: {e}")
    plt.close(fig)
