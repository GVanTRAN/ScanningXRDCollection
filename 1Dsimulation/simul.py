from pathlib import Path
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ============================================================
# Paths
# ============================================================

base_dir = Path(__file__).resolve().parent

template_file = base_dir / "cellulose_template.inp"
working_inp = base_dir / "cellulose_current_run.inp"

topas_exe = r"C:\Topas-7\tc.exe"

# Local temporary filenames for TOPAS
temp_yobs_xy = "temp_calc.xy"
temp_output_xy = "temp_simulated.xy"

temp_yobs_path = base_dir / temp_yobs_xy
temp_output_path = base_dir / temp_output_xy

# ============================================================
# Slider parameter settings
# ============================================================

PARAM_CONFIG = {
    "Phase_1_WP": {
        "label": "P1 wt%",
        "min": 0,
        "max": 100,
        "init": 50,
        "step": 1,
    },

    "D_1": {
        "label": "P1 D",
        "min": 0.5,
        "max": 4,
        "init": 2,
        "step": 0.1,
    },

    "PO_CA1": {
        "label": "P1 texture",
        "min": 0.2,
        "max": 2.0,
        "init": 1.0,
        "step": 0.05,
    },

    "D_2": {
        "label": "P2 D",
        "min": 0.5,
        "max": 30,
        "init": 2,
        "step": 0.5,
    },

    "Z_2": {
        "label": "P2 Z",
        "min": 0.5,
        "max": 30,
        "init": 2,
        "step": 0.5,
    },

    "PO_CA2_001": {
        "label": "P2 Texture 001",
        "min": 0.2,
        "max": 2.0,
        "init": 1.0,
        "step": 0.05,
    },

    "PO_CA2_100": {
        "label": "P2 Texture 100",
        "min": 0.2,
        "max": 2.0,
        "init": 1.0,
        "step": 0.05,
    },

    "PO_WEIGHT2": {
        "label": "P2 001/100 weight",
        "min": 0.0,
        "max": 1.0,
        "init": 0.5,
        "step": 0.05,
    },
}

DEFAULT_CONFIG = {
    "label": None,
    "min": 0,
    "max": 100,
    "init": 1,
    "step": 1,
}

NON_SLIDER_PLACEHOLDERS = {
    "OUTPUT_XY",
    "YOBS_XY",
}

# ============================================================
# Helper functions
# ============================================================

def normalize_y(y):
    ymax = np.max(y)
    if ymax != 0:
        return y / ymax
    return y


def load_xy(filepath):
    data = np.loadtxt(filepath)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    x = data[:, 0]
    y = data[:, 1]
    y = normalize_y(y)

    return x, y


def find_placeholders(text):
    placeholders = sorted(set(re.findall(r"\{([A-Za-z0-9_]+)\}", text)))
    return [p for p in placeholders if p not in NON_SLIDER_PLACEHOLDERS]


def make_inp_text(template_text, values):
    inp_text = template_text

    # Keep local filenames only
    # In the template:
    # yobs_eqn !{YOBS_XY} = X; min 4 max 25 del 0.01
    # Out_X_Ycalc( {OUTPUT_XY} )
    inp_text = inp_text.replace("{YOBS_XY}", temp_yobs_xy)
    inp_text = inp_text.replace("{OUTPUT_XY}", temp_output_xy)

    for name, value in values.items():
        inp_text = inp_text.replace(f"{{{name}}}", f"{value:.8g}")

    return inp_text


def run_topas(template_text, values):
    inp_text = make_inp_text(template_text, values)

    working_inp.write_text(inp_text, encoding="utf-8")

    if temp_output_path.exists():
        temp_output_path.unlink()

    existing_xy = {p.resolve() for p in base_dir.glob("*.xy")}

    subprocess.run(
        [topas_exe, str(working_inp)],
        check=True,
        cwd=base_dir
    )

    if temp_output_path.exists():
        return load_xy(temp_output_path)

    new_xy = sorted(
        p for p in {p.resolve() for p in base_dir.glob("*.xy")} - existing_xy
    )

    if new_xy:
        print(f"Using discovered TOPAS output: {new_xy[0].name}")
        return load_xy(new_xy[0])

    raise FileNotFoundError(
        f"TOPAS ran, but no .xy output was found in: {base_dir}"
    )


def values_from_sliders(sliders):
    return {name: slider.val for name, slider in sliders.items()}


def title_from_values(values):
    preferred_order = [
        "Phase_1_WP",
        "D_1",
        "PO_CA1",
        "D_2",
        "Z_2",
        "PO_CA2_001",
        "PO_CA2_100",
        "PO_WEIGHT2",
    ]

    parts = []

    for name in preferred_order:
        if name in values:
            parts.append(f"{name}={values[name]:g}")

    for name, value in values.items():
        if name not in preferred_order:
            parts.append(f"{name}={value:g}")

    return " | ".join(parts)


# ============================================================
# Read template and detect placeholders
# ============================================================

if not template_file.exists():
    raise FileNotFoundError(f"Template not found: {template_file}")

template_text = template_file.read_text(encoding="utf-8")

placeholders = find_placeholders(template_text)

if not placeholders:
    raise ValueError("No parameter placeholders found in template.")

print("Detected slider placeholders:")
for p in placeholders:
    print(f"  {p}")

for p in placeholders:
    if p not in PARAM_CONFIG:
        PARAM_CONFIG[p] = DEFAULT_CONFIG.copy()
        PARAM_CONFIG[p]["label"] = p

current_values = {
    p: PARAM_CONFIG[p]["init"]
    for p in placeholders
}

# ============================================================
# Initial TOPAS simulation
# ============================================================

print("\nRunning initial TOPAS simulation...")

x0, y0 = run_topas(template_text, current_values)

print("Initial simulation loaded.")

# ============================================================
# Plot and sliders
# ============================================================

n_sliders = len(placeholders)
fig_height = max(6, 4 + 0.45 * n_sliders)

fig, ax = plt.subplots(figsize=(10, fig_height))

bottom_margin = 0.10 + 0.045 * n_sliders
plt.subplots_adjust(bottom=bottom_margin)

line, = ax.plot(x0, y0, linewidth=1.0)

ax.set_xlabel("2θ")
ax.set_ylabel("Normalized intensity")
ax.set_xlim(4, 25)
ax.set_ylim(0, max(y0) * 1.1)

title = ax.set_title(title_from_values(current_values), fontsize=8)

sliders = {}

slider_bottom_start = 0.05
slider_spacing = 0.045

# Display sliders in a nicer order
preferred_slider_order = [
    "Phase_1_WP",
    "D_1",
    "PO_CA1",
    "D_2",
    "Z_2",
    "PO_CA2_001",
    "PO_CA2_100",
    "PO_WEIGHT2",
]

ordered_placeholders = [
    p for p in preferred_slider_order if p in placeholders
] + [
    p for p in placeholders if p not in preferred_slider_order
]

for i, name in enumerate(ordered_placeholders):
    cfg = PARAM_CONFIG[name]

    ax_slider = plt.axes([
        0.18,
        slider_bottom_start + i * slider_spacing,
        0.68,
        0.025
    ])

    slider = Slider(
        ax=ax_slider,
        label=cfg.get("label", name),
        valmin=cfg["min"],
        valmax=cfg["max"],
        valinit=cfg["init"],
        valstep=cfg["step"]
    )

    sliders[name] = slider

fig._sliders = list(sliders.values())

is_running = False

# ============================================================
# Slider update function
# ============================================================

def update(val=None):
    global is_running

    if is_running:
        return

    is_running = True

    values = values_from_sliders(sliders)

    print("\nRunning TOPAS with:")
    for name, value in values.items():
        print(f"  {name} = {value:g}")

    try:
        x, y = run_topas(template_text, values)

        line.set_data(x, y)

        ax.set_xlim(4, 25)
        ax.set_ylim(0, max(y) * 1.1)

        title.set_text(title_from_values(values))

        fig.canvas.draw_idle()

    except subprocess.CalledProcessError:
        print("TOPAS failed. Check cellulose_current_run.inp and topas.log.")

    except Exception as e:
        print("Error during update:")
        print(e)

    finally:
        is_running = False


for slider in sliders.values():
    slider.on_changed(update)

plt.show()
