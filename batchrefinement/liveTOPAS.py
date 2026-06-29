import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from matplotlib.animation import FuncAnimation


# ============================================================
# USER SETTINGS
# ============================================================

script_dir = Path(__file__).resolve().parent

TOPAS_EXE = r"C:\Topas-7\tc.exe"

INP_FILE = script_dir / "batch.inp"
RESULTS_FILE = script_dir / "results.txt"

REFRESH_INTERVAL_MS = 500

exclude_columns = ["Run_Number", "Row", "Col", "Index"]

CLEAR_OLD_RESULTS = True


# ============================================================
# ABSOLUTE MAP SIZE
# Change this to match your real scan grid
# ============================================================

ROW_MIN = 0
ROW_MAX = 109      # inclusive

COL_MIN = 0
COL_MAX = 39      # inclusive


# ============================================================
# PREPARE
# ============================================================

inp_path = INP_FILE
results_path = RESULTS_FILE

if not inp_path.exists():
    raise FileNotFoundError(f"INP file not found:\n{inp_path}")

if not Path(TOPAS_EXE).exists():
    raise FileNotFoundError(f"TOPAS executable not found:\n{TOPAS_EXE}")

if CLEAR_OLD_RESULTS and results_path.exists():
    results_path.unlink()
    print("Deleted old results.txt")


# ============================================================
# LAUNCH TOPAS
# ============================================================

print("Launching TOPAS...")
print(f"INP file: {inp_path}")

topas_process = subprocess.Popen(
    [TOPAS_EXE, str(inp_path)],
    cwd=str(inp_path.parent),
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)


# ============================================================
# FUNCTIONS
# ============================================================

def file_signature(path):
    """
    Detect whether results.txt has changed.
    """
    if not path.exists():
        return None

    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def load_results():
    """
    Safely load results.txt while TOPAS may still be writing it.
    """
    if not results_path.exists():
        return None

    try:
        df = pd.read_csv(results_path, sep=r"\s+")

        if df.empty:
            return None

        if "Row" not in df.columns or "Col" not in df.columns:
            return None

        df["Row"] = df["Row"].astype(int)
        df["Col"] = df["Col"].astype(int)

        return df

    except Exception:
        return None


def get_plot_columns(df):
    """
    Columns that can be plotted as heatmaps.
    """
    return [
        col for col in df.columns
        if col not in exclude_columns
    ]


def make_heatmap(df, column):
    """
    Build heatmap using absolute Row/Col coordinates.

    Row and Col are NOT normalized.
    A value with Row=20, Col=10 is placed at exactly Row=20, Col=10.
    Missing/unrefined pixels remain NaN.
    """

    n_rows = ROW_MAX - ROW_MIN + 1
    n_cols = COL_MAX - COL_MIN + 1

    arr = np.full((n_rows, n_cols), np.nan)

    rows_abs = df["Row"].astype(int).values
    cols_abs = df["Col"].astype(int).values
    vals = pd.to_numeric(df[column], errors="coerce").values

    valid = (
        np.isfinite(vals)
        & (rows_abs >= ROW_MIN)
        & (rows_abs <= ROW_MAX)
        & (cols_abs >= COL_MIN)
        & (cols_abs <= COL_MAX)
    )

    rows_local = rows_abs[valid] - ROW_MIN
    cols_local = cols_abs[valid] - COL_MIN

    arr[rows_local, cols_local] = vals[valid]

    return arr


# ============================================================
# WAIT FOR FIRST RESULT
# ============================================================

print("Waiting for first valid results.txt...")

df = None

while df is None:
    df = load_results()

    if topas_process.poll() is not None and df is None:
        raise RuntimeError("TOPAS finished, but results.txt could not be loaded.")

    time.sleep(0.5)


plot_columns = get_plot_columns(df)

if len(plot_columns) == 0:
    raise ValueError("No plottable columns found.")

current_column = plot_columns[0]

print("\nAvailable columns:")
for c in plot_columns:
    print("  ", c)

print(f"\nInitially plotting: {current_column}")


# ============================================================
# INITIAL HEATMAP
# ============================================================

heatmap = make_heatmap(df, current_column)

if np.all(np.isnan(heatmap)):
    raise ValueError(f"No valid values found for initial column: {current_column}")

full_vmin = np.nanmin(heatmap)
full_vmax = np.nanmax(heatmap)

cmap = plt.cm.viridis.copy()
cmap.set_bad(color="lightgray")

fig, ax_map = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.28)

im = ax_map.imshow(
    heatmap,
    origin="upper",
    aspect="equal",
    vmin=full_vmin,
    vmax=full_vmax,
    cmap=cmap,
    extent=[
        COL_MIN - 0.5,
        COL_MAX + 0.5,
        ROW_MAX + 0.5,
        ROW_MIN - 0.5
    ]
)

cbar = fig.colorbar(im, ax=ax_map, label=current_column)

ax_map.set_xlabel("Column")
ax_map.set_ylabel("Row")

title = fig.suptitle("")


# ============================================================
# WIDGETS
# ============================================================

ax_column = fig.add_axes([0.12, 0.16, 0.30, 0.05])
ax_switch = fig.add_axes([0.44, 0.16, 0.10, 0.05])

ax_vmin = fig.add_axes([0.12, 0.08, 0.15, 0.05])
ax_vmax = fig.add_axes([0.36, 0.08, 0.15, 0.05])

ax_apply = fig.add_axes([0.58, 0.08, 0.10, 0.05])
ax_reset = fig.add_axes([0.70, 0.08, 0.10, 0.05])

ax_prev = fig.add_axes([0.58, 0.16, 0.10, 0.05])
ax_next = fig.add_axes([0.70, 0.16, 0.10, 0.05])

text_column = TextBox(ax_column, "column", initial=current_column)
button_switch = Button(ax_switch, "Switch")

text_vmin = TextBox(ax_vmin, "vmin", initial=f"{full_vmin:.4g}")
text_vmax = TextBox(ax_vmax, "vmax", initial=f"{full_vmax:.4g}")

button_apply = Button(ax_apply, "Apply")
button_reset = Button(ax_reset, "Reset")

button_prev = Button(ax_prev, "Prev")
button_next = Button(ax_next, "Next")


manual_clim = {
    "active": False,
    "vmin": full_vmin,
    "vmax": full_vmax
}

state = {
    "df": df,
    "last_signature": file_signature(results_path),
    "current_column": current_column,
    "plot_columns": plot_columns,
    "last_n_rows": len(df)
}


# ============================================================
# UPDATE FUNCTIONS
# ============================================================

def update_title(df, column):
    if "Run_Number" in df.columns:
        run_text = f"last run = {df['Run_Number'].max()}"
    else:
        run_text = f"rows loaded = {len(df)}"

    if topas_process.poll() is None:
        status = "TOPAS running"
    else:
        status = "TOPAS finished"

    title.set_text(
        f"Live TOPAS refined parameters | {status} | {run_text}\n"
        f"Current column: {column}"
    )


def redraw_heatmap(df, force_clim_reset=False):
    column = state["current_column"]

    if column not in df.columns:
        print(f"Column not found in current dataframe: {column}")
        return

    values = make_heatmap(df, column)

    if np.all(np.isnan(values)):
        print(f"No valid values yet for column: {column}")
        return

    full_vmin = np.nanmin(values)
    full_vmax = np.nanmax(values)

    im.set_data(values)

    ax_map.set_xlim(COL_MIN - 0.5, COL_MAX + 0.5)
    ax_map.set_ylim(ROW_MAX + 0.5, ROW_MIN - 0.5)

    if force_clim_reset:
        manual_clim["active"] = False

    if manual_clim["active"]:
        im.set_clim(manual_clim["vmin"], manual_clim["vmax"])
    else:
        im.set_clim(full_vmin, full_vmax)
        text_vmin.set_val(f"{full_vmin:.4g}")
        text_vmax.set_val(f"{full_vmax:.4g}")

    ax_map.set_title(
        f"Heatmap of {column}\n"
        f"Full scale: {full_vmin:.4g} → {full_vmax:.4g}"
    )

    cbar.set_label(column)
    update_title(df, column)

    fig.canvas.draw_idle()


def reload_results_and_redraw(force=False, force_clim_reset=False):
    """
    Reload results.txt and redraw.

    force=False:
        only reload if results.txt changed.

    force=True:
        reload even if file signature did not change.
        This is important after switching column.
    """

    current_signature = file_signature(results_path)

    if current_signature is None:
        return

    if (not force) and current_signature == state["last_signature"]:
        return

    new_df = load_results()

    if new_df is None:
        return

    state["last_signature"] = current_signature
    state["df"] = new_df
    state["plot_columns"] = get_plot_columns(new_df)

    n_rows = len(new_df)

    if n_rows != state["last_n_rows"]:
        print(
            f"Updated: {n_rows} rows loaded | "
            f"current column = {state['current_column']}"
        )
        state["last_n_rows"] = n_rows

    redraw_heatmap(new_df, force_clim_reset=force_clim_reset)


def animation_update(frame):
    """
    Called continuously by matplotlib.
    This keeps updating the current selected column.
    """
    reload_results_and_redraw(force=False, force_clim_reset=False)


def switch_to_column(new_column):
    """
    Switch plotted column and continue live updating.

    Important:
    This forces a fresh reload from results.txt, not just redraw from old state["df"].
    """

    new_column = new_column.strip()

    # Reload once before checking columns,
    # because a new column may appear after TOPAS writes more output.
    latest_df = load_results()

    if latest_df is not None:
        state["df"] = latest_df
        state["plot_columns"] = get_plot_columns(latest_df)
        state["last_signature"] = file_signature(results_path)

    if new_column not in state["plot_columns"]:
        print(f"Column not found: {new_column}")
        print("Available columns:")
        for col in state["plot_columns"]:
            print("  ", col)
        return

    state["current_column"] = new_column
    text_column.set_val(new_column)

    manual_clim["active"] = False

    print(f"Switched to column: {new_column}")

    # Force redraw immediately with latest data
    reload_results_and_redraw(force=True, force_clim_reset=True)


def switch_button_clicked(event):
    switch_to_column(text_column.text)


def previous_column(event):
    cols = state["plot_columns"]
    col = state["current_column"]

    if col not in cols:
        return

    i = cols.index(col)
    switch_to_column(cols[(i - 1) % len(cols)])


def next_column(event):
    cols = state["plot_columns"]
    col = state["current_column"]

    if col not in cols:
        return

    i = cols.index(col)
    switch_to_column(cols[(i + 1) % len(cols)])


def apply_scale(event):
    try:
        vmin = float(text_vmin.text)
        vmax = float(text_vmax.text)

        if vmin >= vmax:
            print("vmin must be smaller than vmax")
            return

        manual_clim["active"] = True
        manual_clim["vmin"] = vmin
        manual_clim["vmax"] = vmax

        im.set_clim(vmin, vmax)
        fig.canvas.draw_idle()

    except ValueError:
        print("Please enter valid numbers for vmin and vmax")


def reset_scale(event):
    manual_clim["active"] = False

    # Redraw latest data, not old cached data
    reload_results_and_redraw(force=True, force_clim_reset=True)


button_switch.on_clicked(switch_button_clicked)
button_prev.on_clicked(previous_column)
button_next.on_clicked(next_column)
button_apply.on_clicked(apply_scale)
button_reset.on_clicked(reset_scale)

# ============================================================
# START LIVE ANIMATION
# ============================================================

ani = FuncAnimation(
    fig,
    animation_update,
    interval=REFRESH_INTERVAL_MS,
    cache_frame_data=False
)

# Force first draw
reload_results_and_redraw(force=True, force_clim_reset=True)

plt.show()


# ============================================================
# AFTER CLOSING FIGURE
# ============================================================

if topas_process.poll() is None:
    print("Figure closed, but TOPAS is still running.")

print("Done.")
