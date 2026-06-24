#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scanning XRD Viewer
===================

A desktop GUI to explore scanning X-ray diffraction (sXRD) datasets such as
those produced at ESRF ID13.

Workflow
--------
1.  Pick the scan .h5 file (Browse... or paste a path).
2.  Choose the scan entry (e.g. "1.1") and confirm the internal HDF5 paths,
    then click "Load".
3.  The mean 1D pattern is plotted on top. Drag a horizontal span on it
    (or use the 2theta-min / 2theta-max boxes) to define the 2theta ROI used to
    build the spatial intensity map.
4.  "Click pixel" mode: click any pixel on the map to show, simultaneously,
    its 1D pattern and its raw 2D Eiger detector frame.
5.  "Polygon" mode: draw a polygon on the map (click vertices, then close it).
    The 1D pattern (mean or sum over the polygon) is shown immediately; press
    "Average 2D over polygon" to also build the averaged detector frame.
6.  Save the current 1D pattern (.xy), everything (.npz), or just the polygon
    vertices (.json / .npy).

Requirements
------------
    python >= 3.8
    PyQt5
    matplotlib >= 3.5
    numpy
    h5py
    hdf5plugin   (needed to decode compressed Eiger frames)

Run
---
    python scanning_xrd_viewer.py
"""

import os
import sys
import json
import traceback

import numpy as np

# hdf5plugin MUST be imported (and its plugin path exported) before h5py opens
# any file that uses bitshuffle/LZ4 compression, which Eiger data typically does.
try:
    import hdf5plugin  # noqa: F401
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGIN_PATH
    _HAS_HDF5PLUGIN = True
except Exception:
    _HAS_HDF5PLUGIN = False

import h5py

from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector, PolygonSelector
from matplotlib.path import Path as MplPath
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib as mpl

# Force a Qt canvas. Try the modern (matplotlib >= 3.5) module first.
try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
except Exception:  # older matplotlib
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )


# ======================================================================
# Plot style + custom colormap (from the reference figure)
# ======================================================================
def apply_plot_style(fontsize=11, family="serif"):
    """Classic (non-bold, serif by default) scientific plot style."""
    mpl.rcParams.update({
        "font.family": family,
        "font.weight": "normal",
        "font.size": fontsize,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "axes.titlesize": fontsize + 1,
        "axes.labelsize": fontsize,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "mathtext.default": "regular",
    })


DIFFRACTION_CMAP = LinearSegmentedColormap.from_list(
    "diffraction",
    ["#f7fbff", "#c7e9f1", "#75b9d6", "#8cc47c", "#f1d46b", "#f07f2f"],
)
try:
    mpl.colormaps.register(DIFFRACTION_CMAP, force=True)
except Exception:
    try:
        mpl.cm.register_cmap("diffraction", DIFFRACTION_CMAP)
    except Exception:
        pass

CMAP_CHOICES = ["viridis", "diffraction", "magma", "inferno", "plasma", "gray"]


# ======================================================================
# Data container
# ======================================================================
class ScanData:
    """Holds one loaded scan: the integrated 1D patterns + the raw 2D frames."""

    def __init__(self):
        self.scan_path = None
        self.h5file = None
        self.x = None                 # 2theta / q axis, shape (n_points,)
        self.patterns_map = None      # (n_rows, n_cols, n_points), float32
        self.n_rows = 0
        self.n_cols = 0
        self.n_points = 0
        self.eiger_data = None        # lazy h5py dataset, NOT loaded into RAM
        self.eiger_ndim = None
        self.eiger_path = None

    # ------------------------------------------------------------------
    def close(self):
        if self.h5file is not None:
            try:
                self.h5file.close()
            except Exception:
                pass
        self.h5file = None
        self.eiger_data = None

    # ------------------------------------------------------------------
    def load(self, scan_path, twotheta_path, intensity_path,
             eiger_path, n_rows, n_cols):
        """Read the 1D data fully, open the Eiger dataset lazily."""
        self.close()
        self.scan_path = scan_path

        # --- read the integrated 1D data fully (needed for fast clicking) ---
        with h5py.File(scan_path, "r") as f:
            if twotheta_path not in f:
                raise KeyError("2theta path not found:\n  %s" % twotheta_path)
            if intensity_path not in f:
                raise KeyError("intensity path not found:\n  %s" % intensity_path)
            x = np.asarray(f[twotheta_path][:])
            patterns = np.asarray(f[intensity_path][:])

        # --- reshape patterns into a (rows, cols, points) map ---
        if patterns.ndim == 2:
            n_pixels, n_points = patterns.shape
            if n_pixels != n_rows * n_cols:
                raise ValueError(
                    "Map shape mismatch: %d rows x %d cols = %d, "
                    "but intensity has %d pixels."
                    % (n_rows, n_cols, n_rows * n_cols, n_pixels)
                )
            patterns_map = patterns.reshape(n_rows, n_cols, n_points)
        elif patterns.ndim == 3:
            n_rows, n_cols = patterns.shape[0], patterns.shape[1]
            n_points = patterns.shape[2]
            patterns_map = patterns
        else:
            raise ValueError("intensity must be 2D or 3D, got %dD." % patterns.ndim)

        self.x = x.astype(np.float64)
        self.patterns_map = patterns_map.astype(np.float32)
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.n_points = int(n_points)

        # --- open the raw 2D detector data lazily (kept open) ---
        self.eiger_data = None
        self.eiger_ndim = None
        self.eiger_path = None
        if eiger_path:
            self.h5file = h5py.File(scan_path, "r")
            if eiger_path in self.h5file:
                ds = self.h5file[eiger_path]
                if ds.ndim in (3, 4):
                    self.eiger_data = ds
                    self.eiger_ndim = ds.ndim
                    self.eiger_path = eiger_path

    # ------------------------------------------------------------------
    def get_raw_frame(self, row, col):
        """Return the raw 2D detector frame for scan pixel (row, col)."""
        if self.eiger_data is None:
            return None
        try:
            if self.eiger_ndim == 3:
                idx = row * self.n_cols + col
                img = self.eiger_data[idx, :, :]
            else:
                img = self.eiger_data[row, col, :, :]
            return np.asarray(img, dtype=float)
        except Exception:
            return None


# ======================================================================
# HDF5 tree inspector dialog
# ======================================================================
class TreeDialog(QtWidgets.QDialog):
    def __init__(self, scan_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDF5 structure")
        self.resize(560, 600)
        layout = QtWidgets.QVBoxLayout(self)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet("font-family: monospace;")
        lines = []
        try:
            with h5py.File(scan_path, "r") as f:
                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        lines.append("%s   [dataset %s %s]"
                                     % (name, obj.shape, obj.dtype))
                    else:
                        lines.append("%s/" % name)
                f.visititems(visit)
        except Exception as e:
            lines = ["Could not read file:\n%s" % e]
        text.setPlainText("\n".join(lines))
        layout.addWidget(text)
        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


# ======================================================================
# Main window
# ======================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scanning XRD Viewer")

        self.data = ScanData()

        # current selection state
        self.sel = {
            "type": None,            # "pixel" | "polygon"
            "row": None, "col": None,
            "verts": None,           # polygon vertices (N, 2) in (col, row)
            "mask": None,
            "roi_rows": None, "roi_cols": None,
            "pattern1d": None,       # current 1D pattern
            "frame2d": None,         # current 2D detector frame
        }

        # map integration range (2theta)
        self.map_xmin = None
        self.map_xmax = None

        # matplotlib artists (created after load). Each panel is its own
        # figure+canvas so they can be resized independently via splitters.
        self.fig_avg = self.fig_map = self.fig_pat = self.fig_det = None
        self.canvas_avg = self.canvas_map = self.canvas_pat = self.canvas_det = None
        self.ax_avg = self.ax_map = self.ax_pat = self.ax_det = None
        self.im_map = self.im_det = None
        self.line_avg = self.line_pat = None
        self.marker = None
        self.span = None
        self.poly = None
        self.cbar_map = self.cbar_det = None

        self._build_ui()
        self._size_to_screen()

        # F11 toggles full screen, Esc leaves it
        QtWidgets.QShortcut(QtGui.QKeySequence("F11"), self,
                            activated=self.toggle_fullscreen)
        QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self,
                            activated=self.exit_fullscreen)

    def _size_to_screen(self):
        scr = QtWidgets.QApplication.primaryScreen()
        if scr is None:
            self.resize(1300, 820)
            return
        ag = scr.availableGeometry()
        self.move(ag.topLeft())
        self.resize(int(ag.width() * 0.92), int(ag.height() * 0.90))

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def exit_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # ---- left control panel (inside a scroll area so it never
        #      forces the window taller than the screen) ----
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(360)
        pv = QtWidgets.QVBoxLayout(panel)
        pv.setAlignment(QtCore.Qt.AlignTop)
        pv.setSpacing(6)
        pv.setContentsMargins(6, 6, 6, 6)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(384)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(panel)
        root.addWidget(left_scroll)

        # File group
        gb_file = QtWidgets.QGroupBox("1. Data file")
        fl = QtWidgets.QFormLayout(gb_file)
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("/path/to/scan.h5")
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self.on_browse)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.path_edit)
        row.addWidget(browse)
        wrow = QtWidgets.QWidget(); wrow.setLayout(row)
        fl.addRow("File:", wrow)

        self.scan_combo = QtWidgets.QComboBox()
        self.scan_combo.setEditable(True)
        self.scan_combo.currentTextChanged.connect(self.on_scan_changed)
        fl.addRow("Scan entry:", self.scan_combo)

        self.twotheta_edit = QtWidgets.QLineEdit("/1.1/eiger_integrate/integrated/2th")
        self.intensity_edit = QtWidgets.QLineEdit("/1.1/eiger_integrate/integrated/intensity")
        self.eiger_edit = QtWidgets.QLineEdit("/1.1/measurement/eiger")
        fl.addRow("2theta path:", self.twotheta_edit)
        fl.addRow("intensity path:", self.intensity_edit)
        fl.addRow("eiger path:", self.eiger_edit)

        self.rows_spin = QtWidgets.QSpinBox()
        self.rows_spin.setRange(1, 1000000); self.rows_spin.setValue(440)
        self.cols_spin = QtWidgets.QSpinBox()
        self.cols_spin.setRange(1, 1000000); self.cols_spin.setValue(160)
        rc = QtWidgets.QHBoxLayout()
        rc.addWidget(QtWidgets.QLabel("rows")); rc.addWidget(self.rows_spin)
        rc.addWidget(QtWidgets.QLabel("cols")); rc.addWidget(self.cols_spin)
        wrc = QtWidgets.QWidget(); wrc.setLayout(rc)
        fl.addRow("Map shape:", wrc)

        brow = QtWidgets.QHBoxLayout()
        inspect = QtWidgets.QPushButton("Inspect HDF5")
        inspect.clicked.connect(self.on_inspect)
        load = QtWidgets.QPushButton("Load")
        load.setStyleSheet("font-weight: bold;")
        load.clicked.connect(self.on_load)
        brow.addWidget(inspect); brow.addWidget(load)
        wb = QtWidgets.QWidget(); wb.setLayout(brow)
        fl.addRow(wb)
        pv.addWidget(gb_file)

        # ROI (2theta) group
        gb_roi = QtWidgets.QGroupBox("2. 2theta ROI for the map")
        rl = QtWidgets.QFormLayout(gb_roi)
        self.xmin_spin = QtWidgets.QDoubleSpinBox()
        self.xmax_spin = QtWidgets.QDoubleSpinBox()
        for s in (self.xmin_spin, self.xmax_spin):
            s.setDecimals(4); s.setRange(-1e6, 1e6); s.setSingleStep(0.1)
        self.xmin_spin.setValue(9.5); self.xmax_spin.setValue(10.5)
        self.xmin_spin.editingFinished.connect(self.on_xrange_spin)
        self.xmax_spin.editingFinished.connect(self.on_xrange_spin)
        rl.addRow("2theta min:", self.xmin_spin)
        rl.addRow("2theta max:", self.xmax_spin)
        rl.addRow(QtWidgets.QLabel("(or drag a span on the top plot)"))

        # view x-range of the top (mean) plot
        self.avgx_auto = QtWidgets.QCheckBox("Top plot: auto x-range")
        self.avgx_auto.setChecked(True)
        self.avgx_auto.toggled.connect(self.on_avgx_changed)
        rl.addRow(self.avgx_auto)
        self.avgx_min = QtWidgets.QDoubleSpinBox()
        self.avgx_max = QtWidgets.QDoubleSpinBox()
        for s in (self.avgx_min, self.avgx_max):
            s.setDecimals(3); s.setRange(-1e6, 1e6); s.setSingleStep(0.5)
            s.setEnabled(False)
        self.avgx_min.setValue(0.0); self.avgx_max.setValue(40.0)
        self.avgx_min.editingFinished.connect(self.on_avgx_changed)
        self.avgx_max.editingFinished.connect(self.on_avgx_changed)
        rl.addRow("view x min:", self.avgx_min)
        rl.addRow("view x max:", self.avgx_max)
        pv.addWidget(gb_roi)

        # Selection group
        gb_sel = QtWidgets.QGroupBox("3. Selection")
        sl = QtWidgets.QVBoxLayout(gb_sel)
        self.mode_pixel = QtWidgets.QRadioButton("Click pixel")
        self.mode_poly = QtWidgets.QRadioButton("Draw polygon")
        self.mode_pixel.setChecked(True)
        self.mode_pixel.toggled.connect(self.on_mode_changed)
        sl.addWidget(self.mode_pixel)
        sl.addWidget(self.mode_poly)

        red_row = QtWidgets.QHBoxLayout()
        red_row.addWidget(QtWidgets.QLabel("Polygon reduction:"))
        self.reduce_combo = QtWidgets.QComboBox()
        self.reduce_combo.addItems(["mean", "sum"])
        red_row.addWidget(self.reduce_combo)
        wred = QtWidgets.QWidget(); wred.setLayout(red_row)
        sl.addWidget(wred)

        self.poly2d_btn = QtWidgets.QPushButton("Average 2D over polygon")
        self.poly2d_btn.clicked.connect(self.on_average_polygon_2d)
        self.poly2d_btn.setEnabled(False)
        sl.addWidget(self.poly2d_btn)

        self.clear_btn = QtWidgets.QPushButton("Clear selection")
        self.clear_btn.clicked.connect(self.on_clear)
        sl.addWidget(self.clear_btn)
        pv.addWidget(gb_sel)

        # Detector display group
        gb_det = QtWidgets.QGroupBox("4. Detector display")
        dl = QtWidgets.QFormLayout(gb_det)
        self.vmin_spin = QtWidgets.QDoubleSpinBox()
        self.vmax_spin = QtWidgets.QDoubleSpinBox()
        for s in (self.vmin_spin, self.vmax_spin):
            s.setDecimals(2); s.setRange(-1e9, 1e9)
        self.vmin_spin.setValue(0); self.vmax_spin.setValue(10)
        self.vmin_spin.valueChanged.connect(self.update_detector_norm)
        self.vmax_spin.valueChanged.connect(self.update_detector_norm)
        dl.addRow("count min:", self.vmin_spin)
        dl.addRow("count max:", self.vmax_spin)
        auto = QtWidgets.QPushButton("Auto max from frame")
        auto.clicked.connect(self.on_auto_vmax)
        dl.addRow(auto)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(CMAP_CHOICES)
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)
        dl.addRow("colormap:", self.cmap_combo)
        pv.addWidget(gb_det)

        # 1D pattern group (x-range + fixed aspect)
        gb_xrange = QtWidgets.QGroupBox("5. 1D pattern")
        xl = QtWidgets.QFormLayout(gb_xrange)
        self.xauto_check = QtWidgets.QCheckBox("Auto x-range")
        self.xauto_check.setChecked(True)
        self.xauto_check.toggled.connect(self.on_patx_changed)
        xl.addRow(self.xauto_check)
        self.patx_min = QtWidgets.QDoubleSpinBox()
        self.patx_max = QtWidgets.QDoubleSpinBox()
        for s in (self.patx_min, self.patx_max):
            s.setDecimals(3); s.setRange(-1e6, 1e6); s.setSingleStep(0.5)
            s.setEnabled(False)
        self.patx_min.setValue(0.0); self.patx_max.setValue(40.0)
        self.patx_min.editingFinished.connect(self.on_patx_changed)
        self.patx_max.editingFinished.connect(self.on_patx_changed)
        xl.addRow("x min:", self.patx_min)
        xl.addRow("x max:", self.patx_max)

        self.aspect_check = QtWidgets.QCheckBox("Fix aspect (w:h)")
        self.aspect_check.setChecked(True)
        self.aspect_check.toggled.connect(self.on_aspect_changed)
        xl.addRow(self.aspect_check)
        self.aspect_w = QtWidgets.QDoubleSpinBox()
        self.aspect_h = QtWidgets.QDoubleSpinBox()
        for s in (self.aspect_w, self.aspect_h):
            s.setRange(0.5, 50); s.setDecimals(1); s.setSingleStep(0.5)
        self.aspect_w.setValue(8.0); self.aspect_h.setValue(9.0)
        self.aspect_w.valueChanged.connect(self.on_aspect_changed)
        self.aspect_h.valueChanged.connect(self.on_aspect_changed)
        ar = QtWidgets.QHBoxLayout()
        ar.addWidget(QtWidgets.QLabel("w")); ar.addWidget(self.aspect_w)
        ar.addWidget(QtWidgets.QLabel("h")); ar.addWidget(self.aspect_h)
        war = QtWidgets.QWidget(); war.setLayout(ar)
        xl.addRow("aspect:", war)
        pv.addWidget(gb_xrange)

        # Plot style group
        gb_style = QtWidgets.QGroupBox("6. Plot style")
        stl = QtWidgets.QFormLayout(gb_style)
        self.fontsize_spin = QtWidgets.QSpinBox()
        self.fontsize_spin.setRange(5, 40); self.fontsize_spin.setValue(11)
        stl.addRow("Font size:", self.fontsize_spin)
        self.font_combo = QtWidgets.QComboBox()
        self.font_combo.addItems(["serif", "Times New Roman", "DejaVu Serif",
                                  "DejaVu Sans", "Arial", "sans-serif"])
        stl.addRow("Font family:", self.font_combo)
        apply_style_btn = QtWidgets.QPushButton("Apply font")
        apply_style_btn.clicked.connect(self.on_apply_style)
        stl.addRow(apply_style_btn)
        reset_btn = QtWidgets.QPushButton("Reset panel layout")
        reset_btn.clicked.connect(self.reset_layout)
        stl.addRow(reset_btn)
        fs_btn = QtWidgets.QPushButton("Full screen (F11)")
        fs_btn.clicked.connect(self.toggle_fullscreen)
        stl.addRow(fs_btn)
        hint = QtWidgets.QLabel("Drag the bars between plots to resize them.")
        hint.setWordWrap(True)
        stl.addRow(hint)
        pv.addWidget(gb_style)

        # Save group
        gb_save = QtWidgets.QGroupBox("7. Save")
        svl = QtWidgets.QFormLayout(gb_save)
        self.savename_edit = QtWidgets.QLineEdit("roi_pattern")
        svl.addRow("Base name:", self.savename_edit)
        b1 = QtWidgets.QPushButton("Save 1D (.xy)")
        b1.clicked.connect(self.save_xy)
        b2 = QtWidgets.QPushButton("Save all (.npz)")
        b2.clicked.connect(self.save_npz)
        b3 = QtWidgets.QPushButton("Save polygon (.json)")
        b3.clicked.connect(self.save_polygon)
        svl.addRow(b1); svl.addRow(b2); svl.addRow(b3)
        pv.addWidget(gb_save)

        pv.addStretch(1)

        # ---- right plotting area: four independent canvases, each in its own
        #      splitter pane so they can be dragged/resized independently ----
        self.fig_avg = self._new_cfig()
        self.fig_map = self._new_cfig()
        self.fig_pat = self._new_cfig()
        self.fig_det = self._new_cfig()

        pane_avg, self.canvas_avg = self._make_canvas_pane(self.fig_avg)
        pane_map, self.canvas_map = self._make_canvas_pane(self.fig_map)
        pane_pat, self.canvas_pat = self._make_canvas_pane(self.fig_pat)
        pane_det, self.canvas_det = self._make_canvas_pane(self.fig_det)

        self.canvas_map.mpl_connect("button_press_event", self.on_click_map)

        self.hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.hsplit.setChildrenCollapsible(False)
        self.hsplit.addWidget(pane_map)
        self.hsplit.addWidget(pane_pat)
        self.hsplit.addWidget(pane_det)

        self.vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vsplit.setChildrenCollapsible(False)
        self.vsplit.addWidget(pane_avg)
        self.vsplit.addWidget(self.hsplit)
        root.addWidget(self.vsplit, 1)
        self.reset_layout()

        self.statusBar().showMessage(
            "Ready." if _HAS_HDF5PLUGIN else
            "WARNING: hdf5plugin not installed - compressed Eiger frames may fail."
        )

        # make every control group foldable
        for g in (gb_file, gb_roi, gb_sel, gb_det, gb_xrange, gb_style, gb_save):
            self._make_collapsible(g)

        # disable everything that needs data until a file is loaded
        self._set_controls_enabled(False)

    # ------------------------------------------------------------------
    def _new_cfig(self):
        """A Figure that auto-manages label/colorbar spacing."""
        f = Figure()
        try:
            f.set_layout_engine("constrained")
        except Exception:
            try:
                f.set_constrained_layout(True)
            except Exception:
                pass
        return f

    def _make_canvas_pane(self, fig):
        """A toolbar + canvas widget that expands to fill its splitter pane."""
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Expanding)
        canvas.setMinimumSize(120, 120)
        tb = NavigationToolbar(canvas, self)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        lay.addWidget(tb)
        lay.addWidget(canvas, 1)
        return w, canvas

    def reset_layout(self):
        """Restore the default panel proportions."""
        self.hsplit.setSizes([240, 560, 380])
        self.vsplit.setSizes([300, 540])

    def _draw_all(self):
        for c in (self.canvas_avg, self.canvas_map, self.canvas_pat, self.canvas_det):
            if c is not None:
                c.draw_idle()

    def _make_collapsible(self, group, expanded=True):
        """Turn a QGroupBox into a fold/unfold section via its title checkbox."""
        group.setCheckable(True)
        group.setChecked(expanded)

        def on_toggle(on, g=group):
            lay = g.layout()
            if lay is None:
                return
            for i in range(lay.count()):
                item = lay.itemAt(i)
                w = item.widget()
                if w is not None:
                    w.setVisible(on)
                else:
                    sub = item.layout()
                    if sub is not None:
                        for j in range(sub.count()):
                            sw = sub.itemAt(j).widget()
                            if sw is not None:
                                sw.setVisible(on)

        group.toggled.connect(on_toggle)
        if not expanded:
            on_toggle(False)

    def _set_controls_enabled(self, on):
        for w in (self.xmin_spin, self.xmax_spin, self.mode_pixel,
                  self.mode_poly, self.reduce_combo, self.clear_btn,
                  self.vmin_spin, self.vmax_spin, self.cmap_combo,
                  self.xauto_check, self.aspect_check, self.avgx_auto):
            w.setEnabled(on)

    # ==================================================================
    # File handling
    # ==================================================================
    def on_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select scan .h5 file", "", "HDF5 files (*.h5 *.hdf5 *.nxs);;All files (*)"
        )
        if path:
            self.path_edit.setText(path)
            self.populate_scan_entries(path)

    def populate_scan_entries(self, path):
        """List top-level scan entries (like '1.1') so the user can pick one."""
        self.scan_combo.blockSignals(True)
        self.scan_combo.clear()
        try:
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
            # NXentry scan keys usually look like '1.1', '2.1', ...
            scan_keys = [k for k in keys if any(c.isdigit() for c in k)]
            scan_keys = scan_keys or keys
            self.scan_combo.addItems(scan_keys)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Open failed", str(e))
        self.scan_combo.blockSignals(False)
        if self.scan_combo.count():
            self.scan_combo.setCurrentIndex(0)
            self.on_scan_changed(self.scan_combo.currentText())

    def on_scan_changed(self, key):
        key = (key or "").strip().strip("/")
        if not key:
            return
        self.twotheta_edit.setText("/%s/eiger_integrate/integrated/2th" % key)
        self.intensity_edit.setText("/%s/eiger_integrate/integrated/intensity" % key)
        self.eiger_edit.setText("/%s/measurement/eiger" % key)

    def on_inspect(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "No file", "Pick a valid .h5 file first.")
            return
        TreeDialog(path, self).exec_()

    def on_load(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "No file", "Pick a valid .h5 file first.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.data.load(
                path,
                self.twotheta_edit.text().strip(),
                self.intensity_edit.text().strip(),
                self.eiger_edit.text().strip(),
                self.rows_spin.value(),
                self.cols_spin.value(),
            )
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self, "Load error", "%s\n\n%s" % (e, traceback.format_exc())
            )
            return
        QtWidgets.QApplication.restoreOverrideCursor()

        # default 2theta ROI: keep current if inside x range, else middle decile
        xlo, xhi = float(self.data.x.min()), float(self.data.x.max())
        for s in (self.xmin_spin, self.xmax_spin):
            s.setRange(xlo, xhi)
        if not (xlo <= self.xmin_spin.value() < self.xmax_spin.value() <= xhi):
            self.xmin_spin.setValue(xlo + 0.45 * (xhi - xlo))
            self.xmax_spin.setValue(xlo + 0.55 * (xhi - xlo))
        self.map_xmin = self.xmin_spin.value()
        self.map_xmax = self.xmax_spin.value()

        # default 1D-pattern manual x-range to the full data extent
        self.patx_min.setValue(xlo)
        self.patx_max.setValue(xhi)
        self.avgx_min.setValue(xlo)
        self.avgx_max.setValue(xhi)

        self._build_plots()
        self._set_controls_enabled(True)
        self.poly2d_btn.setEnabled(False)

        det_msg = (self.data.eiger_path if self.data.eiger_data is not None
                   else "no 2D detector data found")
        self.statusBar().showMessage(
            "Loaded %s  |  map %d x %d x %d  |  detector: %s"
            % (os.path.basename(path), self.data.n_rows, self.data.n_cols,
               self.data.n_points, det_msg)
        )

    # ==================================================================
    # Build the matplotlib panels (called once per load)
    # ==================================================================
    def _build_plots(self):
        for f in (self.fig_avg, self.fig_map, self.fig_pat, self.fig_det):
            f.clear()
        self.ax_avg = self.fig_avg.add_subplot(111)
        self.ax_map = self.fig_map.add_subplot(111)
        self.ax_pat = self.fig_pat.add_subplot(111)
        self.ax_det = self.fig_det.add_subplot(111)

        cmap = self.cmap_combo.currentText()

        # --- mean 1D pattern (top) ---
        avg = self.data.patterns_map.mean(axis=(0, 1))
        (self.line_avg,) = self.ax_avg.plot(self.data.x, avg, color="#222222")
        self.ax_avg.set_title("Mean 1D pattern  (drag to set 2theta ROI)")
        self.ax_avg.set_xlabel("2theta")
        self.ax_avg.set_ylabel("Mean intensity")
        self.ax_avg.grid(alpha=0.3)
        self._apply_avg_xrange()

        # --- spatial intensity map ---
        self.im_map = self.ax_map.imshow(
            self._compute_map(), origin="upper", aspect="equal", cmap=cmap
        )
        self._set_map_clim()
        self.marker = self.ax_map.scatter([], [], s=90, marker="x",
                                          linewidths=2.5, color="red")
        self.ax_map.set_title("Intensity map")
        self.ax_map.set_xticks([]); self.ax_map.set_yticks([])
        self.cbar_map = self.fig_map.colorbar(self.im_map, ax=self.ax_map,
                                              fraction=0.046, pad=0.04)
        self.cbar_map.set_label("Integrated intensity")

        # --- selected 1D pattern ---
        (self.line_pat,) = self.ax_pat.plot([], [], color="#f07f2f", linewidth=1.8)
        self.ax_pat.set_title("Selected 1D pattern")
        self.ax_pat.set_xlabel("2theta"); self.ax_pat.set_ylabel("Intensity")
        self.ax_pat.grid(alpha=0.3)

        # --- raw 2D detector frame ---
        dummy = np.zeros((10, 10))
        self.im_det = self.ax_det.imshow(
            dummy, origin="upper", aspect="equal", cmap=cmap,
            norm=Normalize(self.vmin_spin.value(), self.vmax_spin.value()),
        )
        if self.data.eiger_data is None:
            self.ax_det.set_title("No 2D detector data")
        else:
            self.ax_det.set_title("Raw 2D detector (click a pixel)")
        self.ax_det.set_xticks([]); self.ax_det.set_yticks([])
        self.cbar_det = self.fig_det.colorbar(self.im_det, ax=self.ax_det,
                                              fraction=0.046, pad=0.04)
        self.cbar_det.set_label("Detector counts")

        # --- interactive selectors ---
        self.span = SpanSelector(
            self.ax_avg, self.on_span_select, "horizontal",
            useblit=False, interactive=True,
            props=dict(alpha=0.2, facecolor="red"),
        )
        try:
            self.span.extents = (self.map_xmin, self.map_xmax)
        except Exception:
            pass

        self._create_polygon_selector()
        self._apply_mode()
        self._apply_pattern_aspect()

        for c in (self.canvas_avg, self.canvas_map, self.canvas_pat, self.canvas_det):
            c.draw_idle()

    def _create_polygon_selector(self):
        """(Re)create the polygon selector on the map axis."""
        if self.poly is not None:
            try:
                self.poly.set_active(False)
                self.poly.disconnect_events()
            except Exception:
                pass
        self.poly = PolygonSelector(
            self.ax_map, self.on_polygon_complete, useblit=False,
            props=dict(color="red", linewidth=2, alpha=0.8),
        )
        self.poly.set_active(self.mode_poly.isChecked())

    def on_apply_style(self):
        apply_plot_style(self.fontsize_spin.value(), self.font_combo.currentText())
        if self.data.patterns_map is not None:
            self._build_plots()
            self._redisplay_selection()

    def _redisplay_selection(self):
        """Re-draw the current selection after a full plot rebuild."""
        t = self.sel["type"]
        if t == "pixel" and self.sel["pattern1d"] is not None:
            self.marker.set_offsets([[self.sel["col"], self.sel["row"]]])
            self._show_pattern(self.sel["pattern1d"],
                               "1D pattern  row=%d  col=%d"
                               % (self.sel["row"], self.sel["col"]))
            if self.sel["frame2d"] is not None:
                self._show_detector(self.sel["frame2d"],
                                    "Raw 2D detector  row=%d  col=%d"
                                    % (self.sel["row"], self.sel["col"]))
        elif t == "polygon" and self.sel["pattern1d"] is not None:
            v = self.sel["verts"]
            if v is not None and len(v) >= 3:
                vv = np.vstack([v, v[0]])
                self.ax_map.plot(vv[:, 0], vv[:, 1], color="red", linewidth=2)
            npix = self.sel["roi_rows"].size if self.sel["roi_rows"] is not None else 0
            self._show_pattern(self.sel["pattern1d"], "Polygon 1D (%d pixels)" % npix)
            if self.sel["frame2d"] is not None:
                self._show_detector(self.sel["frame2d"], "Polygon 2D")
        self.canvas_map.draw_idle()

    # ==================================================================
    # Map computation / display
    # ==================================================================
    def _compute_map(self):
        x = self.data.x
        mask = (x >= self.map_xmin) & (x <= self.map_xmax)
        if not np.any(mask):
            # fall back to nearest single channel
            mask = np.zeros_like(x, dtype=bool)
            mask[np.argmin(np.abs(x - 0.5 * (self.map_xmin + self.map_xmax)))] = True
        return self.data.patterns_map[:, :, mask].sum(axis=2)

    def _set_map_clim(self):
        m = self.im_map.get_array()
        try:
            vmin = np.nanpercentile(m, 2)
            vmax = np.nanpercentile(m, 98)
            if vmax <= vmin:
                vmax = vmin + 1
            self.im_map.set_clim(vmin, vmax)
        except Exception:
            pass

    def _refresh_map(self):
        self.map_xmin = min(self.xmin_spin.value(), self.xmax_spin.value())
        self.map_xmax = max(self.xmin_spin.value(), self.xmax_spin.value())
        self.im_map.set_data(self._compute_map())
        self._set_map_clim()
        self.ax_map.set_title("Intensity map  %.3f-%.3f" % (self.map_xmin, self.map_xmax))
        self.canvas_map.draw_idle()

    # ==================================================================
    # 2theta ROI controls
    # ==================================================================
    def on_span_select(self, xmin, xmax):
        if xmax <= xmin:
            return
        self.xmin_spin.blockSignals(True); self.xmax_spin.blockSignals(True)
        self.xmin_spin.setValue(xmin); self.xmax_spin.setValue(xmax)
        self.xmin_spin.blockSignals(False); self.xmax_spin.blockSignals(False)
        self._refresh_map()

    def on_xrange_spin(self):
        try:
            self.span.extents = (self.xmin_spin.value(), self.xmax_spin.value())
        except Exception:
            pass
        self._refresh_map()

    # ==================================================================
    # Mode handling
    # ==================================================================
    def on_mode_changed(self, *_):
        self._apply_mode()

    def _apply_mode(self):
        if self.poly is None:
            return
        poly_mode = self.mode_poly.isChecked()
        self.poly.set_active(poly_mode)
        self.poly2d_btn.setEnabled(poly_mode and self.sel["verts"] is not None
                                   and self.data.eiger_data is not None)
        if poly_mode:
            self.statusBar().showMessage(
                "Polygon mode: click vertices on the map, then click the first "
                "point (or double-click) to close it."
            )
        else:
            self.statusBar().showMessage("Click-pixel mode: click a pixel on the map.")

    # ==================================================================
    # Pixel click
    # ==================================================================
    def on_click_map(self, event):
        if self.mode_poly.isChecked():
            return  # polygon selector handles clicks
        if event.inaxes != self.ax_map:
            return
        if event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata)); row = int(round(event.ydata))
        if not (0 <= row < self.data.n_rows and 0 <= col < self.data.n_cols):
            return

        y = self.data.patterns_map[row, col, :]
        self.marker.set_offsets([[col, row]])
        self._show_pattern(y, "1D pattern  row=%d  col=%d" % (row, col))

        frame = self.data.get_raw_frame(row, col)
        self._show_detector(frame, "Raw 2D detector  row=%d  col=%d" % (row, col))

        self.sel.update(type="pixel", row=row, col=col, verts=None, mask=None,
                        roi_rows=None, roi_cols=None,
                        pattern1d=np.asarray(y), frame2d=frame)
        self.poly2d_btn.setEnabled(False)
        self.statusBar().showMessage(
            "Pixel row=%d col=%d  (flat index %d)" % (row, col, row * self.data.n_cols + col)
        )
        self._draw_all()

    # ==================================================================
    # Polygon selection
    # ==================================================================
    def on_polygon_complete(self, verts):
        if not self.mode_poly.isChecked() or verts is None or len(verts) < 3:
            return
        verts = np.asarray(verts)  # (N, 2) in (x=col, y=row)
        path = MplPath(verts)
        cg, rg = np.meshgrid(np.arange(self.data.n_cols), np.arange(self.data.n_rows))
        pts = np.vstack((cg.ravel(), rg.ravel())).T
        mask = path.contains_points(pts).reshape(self.data.n_rows, self.data.n_cols)
        roi_rows, roi_cols = np.where(mask)
        if roi_rows.size == 0:
            self.statusBar().showMessage("Polygon contains no pixels.")
            return

        roi = self.data.patterns_map[mask, :]
        reduce_ = self.reduce_combo.currentText()
        patt = roi.mean(axis=0) if reduce_ == "mean" else roi.sum(axis=0)

        self.marker.set_offsets(np.empty((0, 2)))
        self._show_pattern(
            patt, "Polygon 1D (%s of %d pixels)" % (reduce_, roi_rows.size)
        )

        self.sel.update(type="polygon", row=None, col=None, verts=verts,
                        mask=mask, roi_rows=roi_rows, roi_cols=roi_cols,
                        pattern1d=np.asarray(patt), frame2d=None)
        self.poly2d_btn.setEnabled(self.data.eiger_data is not None)
        self.statusBar().showMessage(
            "Polygon: %d pixels. Press 'Average 2D over polygon' for the detector frame."
            % roi_rows.size
        )
        self._draw_all()

    def on_average_polygon_2d(self):
        if self.data.eiger_data is None:
            QtWidgets.QMessageBox.information(self, "No detector",
                                              "This file has no raw 2D detector data.")
            return
        if self.sel["roi_rows"] is None:
            return
        rows, cols = self.sel["roi_rows"], self.sel["roi_cols"]
        n = rows.size
        reduce_ = self.reduce_combo.currentText()

        progress = QtWidgets.QProgressDialog(
            "Reading %d detector frames..." % n, "Cancel", 0, n, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)

        acc = None
        used = 0
        for i in range(n):
            if progress.wasCanceled():
                break
            frame = self.data.get_raw_frame(int(rows[i]), int(cols[i]))
            if frame is not None:
                if acc is None:
                    acc = np.zeros_like(frame, dtype=np.float64)
                if acc.shape == frame.shape:
                    acc += frame
                    used += 1
            progress.setValue(i + 1)
            if i % 25 == 0:
                QtWidgets.QApplication.processEvents()
        progress.setValue(n)

        if acc is None or used == 0:
            self.statusBar().showMessage("Could not read any detector frames.")
            return
        result = acc / used if reduce_ == "mean" else acc
        self.sel["frame2d"] = result
        self._show_detector(result, "Polygon 2D (%s of %d frames)" % (reduce_, used))
        self.statusBar().showMessage("Averaged %d detector frames." % used)
        self.canvas_det.draw_idle()

    # ==================================================================
    # Plot helpers
    # ==================================================================
    def _show_pattern(self, y, title):
        self.line_pat.set_data(self.data.x, y)
        self.ax_pat.set_title(title)
        self._apply_pattern_xrange()

    def _apply_pattern_xrange(self):
        """Set the 1D pattern x-limits (auto or manual) and fit y to the view."""
        if self.line_pat is None:
            return
        xdata, ydata = self.line_pat.get_data()
        xdata = np.asarray(xdata); ydata = np.asarray(ydata)
        if xdata.size == 0:
            self.ax_pat.relim(); self.ax_pat.autoscale_view()
            self.canvas_pat.draw_idle()
            return
        if self.xauto_check.isChecked():
            xmin, xmax = float(np.min(xdata)), float(np.max(xdata))
        else:
            xmin = min(self.patx_min.value(), self.patx_max.value())
            xmax = max(self.patx_min.value(), self.patx_max.value())
        self.ax_pat.set_xlim(xmin, xmax)
        sel = (xdata >= xmin) & (xdata <= xmax)
        yv = ydata[sel] if np.any(sel) else ydata
        if yv.size:
            ymin, ymax = float(np.nanmin(yv)), float(np.nanmax(yv))
            if ymax <= ymin:
                ymax = ymin + 1.0
            pad = 0.05 * (ymax - ymin)
            self.ax_pat.set_ylim(ymin - pad, ymax + pad)
        self.canvas_pat.draw_idle()

    def on_patx_changed(self, *_):
        auto = self.xauto_check.isChecked()
        self.patx_min.setEnabled(not auto)
        self.patx_max.setEnabled(not auto)
        self._apply_pattern_xrange()

    def _apply_avg_xrange(self):
        """Set the top (mean) plot x-limits (auto or manual) and fit y to view."""
        if self.line_avg is None:
            return
        xdata, ydata = self.line_avg.get_data()
        xdata = np.asarray(xdata); ydata = np.asarray(ydata)
        if xdata.size == 0:
            return
        if self.avgx_auto.isChecked():
            xmin, xmax = float(np.min(xdata)), float(np.max(xdata))
        else:
            xmin = min(self.avgx_min.value(), self.avgx_max.value())
            xmax = max(self.avgx_min.value(), self.avgx_max.value())
        self.ax_avg.set_xlim(xmin, xmax)
        sel = (xdata >= xmin) & (xdata <= xmax)
        yv = ydata[sel] if np.any(sel) else ydata
        if yv.size:
            ymin, ymax = float(np.nanmin(yv)), float(np.nanmax(yv))
            if ymax <= ymin:
                ymax = ymin + 1.0
            pad = 0.05 * (ymax - ymin)
            self.ax_avg.set_ylim(ymin - pad, ymax + pad)
        self.canvas_avg.draw_idle()

    def on_avgx_changed(self, *_):
        auto = self.avgx_auto.isChecked()
        self.avgx_min.setEnabled(not auto)
        self.avgx_max.setEnabled(not auto)
        self._apply_avg_xrange()

    def _apply_pattern_aspect(self):
        """Lock the 1D pattern box to a fixed width:height ratio, or release it."""
        if self.ax_pat is None:
            return
        if self.aspect_check.isChecked():
            w = max(self.aspect_w.value(), 0.1)
            h = max(self.aspect_h.value(), 0.1)
            self.ax_pat.set_box_aspect(h / w)   # box_aspect = height / width
        else:
            self.ax_pat.set_box_aspect(None)
        self.canvas_pat.draw_idle()

    def on_aspect_changed(self, *_):
        on = self.aspect_check.isChecked()
        self.aspect_w.setEnabled(on)
        self.aspect_h.setEnabled(on)
        self._apply_pattern_aspect()

    def _show_detector(self, frame, title):
        if frame is None:
            self.ax_det.set_title("Detector frame unavailable")
            return
        det_y, det_x = frame.shape
        self.im_det.set_data(frame)
        self.im_det.set_extent([0, det_x, det_y, 0])
        self.ax_det.set_xlim(0, det_x); self.ax_det.set_ylim(det_y, 0)
        self.ax_det.set_title(title)
        self.update_detector_norm()

    def update_detector_norm(self, *_):
        if self.im_det is None:
            return
        vmin = self.vmin_spin.value(); vmax = self.vmax_spin.value()
        if vmax <= vmin:
            vmax = vmin + 1
        self.im_det.set_norm(Normalize(vmin, vmax))
        if self.cbar_det is not None:
            self.cbar_det.update_normal(self.im_det)
        self.canvas_det.draw_idle()

    def on_auto_vmax(self):
        frame = self.sel.get("frame2d")
        if frame is None:
            return
        finite = frame[np.isfinite(frame)]
        if finite.size == 0:
            return
        self.vmin_spin.blockSignals(True)
        self.vmin_spin.setValue(0.0)
        self.vmin_spin.blockSignals(False)
        self.vmax_spin.setValue(float(np.nanmax(finite)))  # triggers norm update

    def on_cmap_changed(self, name):
        if self.im_map is not None:
            self.im_map.set_cmap(name)
        if self.im_det is not None:
            self.im_det.set_cmap(name)
        self.canvas_map.draw_idle(); self.canvas_det.draw_idle()

    # ==================================================================
    # Clear
    # ==================================================================
    def on_clear(self):
        self.marker.set_offsets(np.empty((0, 2)))
        self.line_pat.set_data([], [])
        self.ax_pat.set_title("Selected 1D pattern")
        if self.im_det is not None:
            self.im_det.set_data(np.zeros((10, 10)))
            self.im_det.set_extent([0, 10, 10, 0])
            self.ax_det.set_title("Raw 2D detector (click a pixel)")
        self.sel.update(type=None, row=None, col=None, verts=None, mask=None,
                        roi_rows=None, roi_cols=None, pattern1d=None, frame2d=None)
        self._create_polygon_selector()
        self._apply_mode()
        self.statusBar().showMessage("Selection cleared.")
        self._draw_all()

    # ==================================================================
    # Saving
    # ==================================================================
    def _base_path(self):
        name = self.savename_edit.text().strip() or "roi_pattern"
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose save folder", os.getcwd())
        if not folder:
            return None
        return os.path.join(folder, name)

    def save_xy(self):
        if self.sel["pattern1d"] is None:
            QtWidgets.QMessageBox.information(self, "Nothing to save",
                                              "Select a pixel or polygon first.")
            return
        base = self._base_path()
        if base is None:
            return
        path = base + ".xy"
        np.savetxt(path, np.column_stack([self.data.x, self.sel["pattern1d"]]),
                   fmt="%.8f %.8f")
        self.statusBar().showMessage("Saved %s" % path)

    def save_npz(self):
        if self.sel["pattern1d"] is None:
            QtWidgets.QMessageBox.information(self, "Nothing to save",
                                              "Select a pixel or polygon first.")
            return
        base = self._base_path()
        if base is None:
            return
        path = base + ".npz"
        np.savez(
            path,
            type=str(self.sel["type"]),
            x=self.data.x,
            pattern1d=self.sel["pattern1d"],
            frame2d=(self.sel["frame2d"] if self.sel["frame2d"] is not None
                     else np.array([])),
            row=-1 if self.sel["row"] is None else self.sel["row"],
            col=-1 if self.sel["col"] is None else self.sel["col"],
            polygon_vertices=(self.sel["verts"] if self.sel["verts"] is not None
                              else np.array([])),
            roi_mask=(self.sel["mask"] if self.sel["mask"] is not None
                      else np.array([])),
            map_xmin=self.map_xmin, map_xmax=self.map_xmax,
            scan_path=str(self.data.scan_path),
        )
        self.statusBar().showMessage("Saved %s" % path)

    def save_polygon(self):
        if self.sel["verts"] is None:
            QtWidgets.QMessageBox.information(self, "No polygon",
                                              "Draw a polygon first.")
            return
        base = self._base_path()
        if base is None:
            return
        verts = np.asarray(self.sel["verts"])
        np.save(base + "_polygon.npy", verts)
        with open(base + "_polygon.json", "w") as fh:
            json.dump({
                "scan_path": str(self.data.scan_path),
                "n_rows": self.data.n_rows, "n_cols": self.data.n_cols,
                "vertices_col_row": verts.tolist(),
                "map_xmin": self.map_xmin, "map_xmax": self.map_xmax,
                "n_pixels": int(self.sel["roi_rows"].size)
                if self.sel["roi_rows"] is not None else 0,
            }, fh, indent=2)
        self.statusBar().showMessage(
            "Saved %s_polygon.npy and .json" % os.path.basename(base))

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self.data.close()
        super().closeEvent(event)


def main():
    apply_plot_style(11, "serif")
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
