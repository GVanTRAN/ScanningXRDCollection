import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.widgets import TextBox, Button, PolygonSelector
from matplotlib.path import Path


# ============================================================
# LOAD DATA
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "results1.txt")

exclude_columns = ["Run_Number", "Row", "Col", "Index", "gof", "perc1"]

df = pd.read_csv(file_path, sep=r"\s+")

df["Row"] = df["Row"].astype(int)
df["Col"] = df["Col"].astype(int)

intensity_columns = [
    col for col in df.columns
    if col not in exclude_columns
]

# Keep widgets alive
figures = []
widgets = []
selectors = []


# ============================================================
# POLYGON STATS
# ============================================================

def polygon_stats(verts, heatmap_values, hot_threshold=np.inf):
    """
    Compute polygon stats on a 2D heatmap, optionally ignoring
    values above hot_threshold.
    """
    nrows, ncols = heatmap_values.shape

    yy, xx = np.mgrid[0:nrows, 0:ncols]
    points = np.vstack((xx.ravel(), yy.ravel())).T

    poly_path = Path(verts)
    mask = poly_path.contains_points(points).reshape(nrows, ncols)

    raw_values = heatmap_values[mask]
    raw_values = raw_values[np.isfinite(raw_values)]

    if len(raw_values) == 0:
        return None

    hot_mask = raw_values > hot_threshold
    values = raw_values[~hot_mask]

    if len(values) == 0:
        return {
            "N_total": len(raw_values),
            "N_used": 0,
            "N_hot_ignored": int(np.sum(hot_mask)),
            "threshold": hot_threshold,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "N_total": len(raw_values),
        "N_used": len(values),
        "N_hot_ignored": int(np.sum(hot_mask)),
        "threshold": hot_threshold,
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values, ddof=1) if len(values) > 1 else 0.0,
        "min": np.min(values),
        "max": np.max(values),
    }


# ============================================================
# HEATMAPS
# ============================================================

for col in intensity_columns:

    heatmap = df.pivot(index="Row", columns="Col", values=col)
    heatmap_values = heatmap.values.astype(float)

    full_vmin = np.nanmin(heatmap_values)
    full_vmax = np.nanmax(heatmap_values)

    fig, ax = plt.subplots(figsize=(10, 7))

    # More bottom space for widgets; stats text moved upward
    plt.subplots_adjust(bottom=0.34)

    im = ax.imshow(
        heatmap_values,
        origin="upper",
        aspect="equal",
        vmin=full_vmin,
        vmax=full_vmax,
        cmap="viridis"
    )

    fig.colorbar(im, ax=ax, label=col)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    ax.set_title(
        f"Heatmap of {col}\nFull scale: {full_vmin:.4g} \u2192 {full_vmax:.4g}"
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------
    state = {
        "verts": None
    }

    # --------------------------------------------------------
    # Stats text: moved UP
    # --------------------------------------------------------
    stats_text = fig.text(
        0.12, 0.52,   # was 0.24
        "Draw polygon to get stats",
        fontsize=11,
        va="top",
        ha="left",
        family="monospace"
    )

    # --------------------------------------------------------
    # Widgets
    # --------------------------------------------------------
    ax_vmin = fig.add_axes([0.12, 0.08, 0.16, 0.05])
    ax_vmax = fig.add_axes([0.34, 0.08, 0.16, 0.05])
    ax_thresh = fig.add_axes([0.56, 0.08, 0.16, 0.05])
    ax_apply = fig.add_axes([0.77, 0.08, 0.14, 0.05])
    ax_reset = fig.add_axes([0.77, 0.02, 0.14, 0.05])

    text_vmin = TextBox(ax_vmin, "vmin", initial=f"{full_vmin:.4g}")
    text_vmax = TextBox(ax_vmax, "vmax", initial=f"{full_vmax:.4g}")
    text_thresh = TextBox(ax_thresh, "hot max", initial="inf")

    button_apply = Button(ax_apply, "Apply")
    button_reset = Button(ax_reset, "Reset")

    # --------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------
    def get_hot_threshold(text_thresh=text_thresh):
        s = text_thresh.text.strip().lower()

        if s in ["", "inf", "+inf", "none", "no", "off"]:
            return np.inf

        try:
            return float(s)
        except ValueError:
            print("Invalid hot-pixel threshold. Using inf.")
            return np.inf

    def update_stats(
        fig=fig,
        col=col,
        state=state,
        heatmap_values=heatmap_values,
        stats_text=stats_text,
        text_thresh=text_thresh
    ):
        if state["verts"] is None:
            return

        hot_threshold = get_hot_threshold(text_thresh)
        stats = polygon_stats(state["verts"], heatmap_values, hot_threshold=hot_threshold)

        if stats is None:
            msg = "No valid pixels inside polygon"
            stats_text.set_text(msg)
            print(f"\n{col}\n{msg}")
            fig.canvas.draw_idle()
            return

        if stats["N_used"] == 0:
            msg = (
                f"Polygon stats for {col}\n"
                f"N total      = {stats['N_total']}\n"
                f"N used       = 0\n"
                f"hot ignored  = {stats['N_hot_ignored']}\n"
                f"threshold    = {stats['threshold']:.6g}\n"
                f"All selected values were above threshold."
            )
            stats_text.set_text(msg)
            print("\n" + "=" * 50)
            print(msg)
            print("=" * 50)
            fig.canvas.draw_idle()
            return

        threshold_str = "inf" if np.isinf(stats["threshold"]) else f"{stats['threshold']:.6g}"

        msg = (
            f"Polygon stats for {col}\n"
            f"N total      = {stats['N_total']}\n"
            f"N used       = {stats['N_used']}\n"
            f"hot ignored  = {stats['N_hot_ignored']}\n"
            f"threshold    = {threshold_str}\n"
            f"mean         = {stats['mean']:.6g}\n"
            f"median       = {stats['median']:.6g}\n"
            f"std          = {stats['std']:.6g}\n"
            f"min          = {stats['min']:.6g}\n"
            f"max          = {stats['max']:.6g}"
        )

        stats_text.set_text(msg)

        print("\n" + "=" * 50)
        print(msg)
        print("=" * 50)

        fig.canvas.draw_idle()

    # --------------------------------------------------------
    # Polygon callback
    # --------------------------------------------------------
    def onselect(
        verts,
        state=state,
        fig=fig,
        col=col,
        heatmap_values=heatmap_values,
        stats_text=stats_text,
        text_thresh=text_thresh
    ):
        state["verts"] = verts
        update_stats(
            fig=fig,
            col=col,
            state=state,
            heatmap_values=heatmap_values,
            stats_text=stats_text,
            text_thresh=text_thresh
        )

    # --------------------------------------------------------
    # Button callbacks
    # --------------------------------------------------------
    def apply_all(
        event,
        im=im,
        fig=fig,
        text_vmin=text_vmin,
        text_vmax=text_vmax,
        text_thresh=text_thresh,
        state=state,
        col=col,
        heatmap_values=heatmap_values,
        stats_text=stats_text
    ):
        try:
            vmin = float(text_vmin.text)
            vmax = float(text_vmax.text)

            if vmin >= vmax:
                print("vmin must be smaller than vmax")
                return

            im.set_clim(vmin, vmax)

        except ValueError:
            print("Please enter valid numbers for vmin and vmax")
            return

        update_stats(
            fig=fig,
            col=col,
            state=state,
            heatmap_values=heatmap_values,
            stats_text=stats_text,
            text_thresh=text_thresh
        )

        fig.canvas.draw_idle()

    def reset_all(
        event,
        im=im,
        fig=fig,
        text_vmin=text_vmin,
        text_vmax=text_vmax,
        text_thresh=text_thresh,
        full_vmin=full_vmin,
        full_vmax=full_vmax,
        state=state,
        col=col,
        heatmap_values=heatmap_values,
        stats_text=stats_text
    ):
        text_vmin.set_val(f"{full_vmin:.4g}")
        text_vmax.set_val(f"{full_vmax:.4g}")
        text_thresh.set_val("inf")

        im.set_clim(full_vmin, full_vmax)

        update_stats(
            fig=fig,
            col=col,
            state=state,
            heatmap_values=heatmap_values,
            stats_text=stats_text,
            text_thresh=text_thresh
        )

        fig.canvas.draw_idle()

    button_apply.on_clicked(apply_all)
    button_reset.on_clicked(reset_all)

    selector = PolygonSelector(
        ax,
        onselect,
        useblit=True,
        props=dict(color="red", linewidth=1.5, alpha=0.9),
        handle_props=dict(
            marker="o",
            markersize=4,
            mec="red",
            mfc="red",
            alpha=0.9
        )
    )

    figures.append(fig)
    widgets.append((text_vmin, text_vmax, text_thresh, button_apply, button_reset))
    selectors.append(selector)

plt.show()