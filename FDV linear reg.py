import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import csv
import os

# --- Sample data (units: billions) ---
#Exemple:
"""
data = {
    "exchange": ["Hyperliquid", "Aster", "Jupiter", "Avantis", "Apex", "Drift", "dydx", "gmx", "gains_net"],
    "OI_B":     [7.27400,   2.46800,   0.30152,   0.03245,  0.05073,  0.31095, 0.08567, 0.12114,    0.01431],
    "FDV_B":    [24.53800,  5.71500,   1.31200,   0.25612,  0.20975,  0.15205, 0.16340, 0.08336,    0.03424]
}
df = pd.DataFrame(data)
"""
data = {
    "exchange": ["Hyperliquid", "Aster", "Jupiter", "Avantis", "Apex", "Drift", "dydx", "gmx", "gains_net"],
    "OI_B":     [,   ,  ,   ,  ,  , , ,    ],
    "FDV_B":    [,  ,   ,   ,  ,  , , ,    ]
}
df = pd.DataFrame(data)

# ---- fitting helpers ----
def fit_origin(x, y):
    """Fit FDV = slope * OI (no intercept)."""
    slope = np.sum(x * y) / np.sum(x * x)
    pred = slope * x
    residuals = y - pred
    SSE = np.sum(residuals**2)
    SST = np.sum((y - np.mean(y))**2)
    R2 = 1 - SSE / SST if SST > 0 else np.nan
    return slope, pred, R2

# ---- Draggable annotations manager ----
class DraggableAnnotationManager:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.annotations = []
        self._dragging = None
        self._cid_pick = fig.canvas.mpl_connect('pick_event', self._on_pick)
        self._cid_motion = fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self._cid_release = fig.canvas.mpl_connect('button_release_event', self._on_release)
        self._cid_key = fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._orig_positions = {}

    def register(self, annotation, name=None, picker_tolerance=6):
        annotation.set_picker(picker_tolerance)
        label_name = name if name is not None else f"label_{len(self.annotations)+1}"
        self.annotations.append((annotation, label_name))
        # Store original position (for reset functionality)
        self._orig_positions[label_name] = annotation.get_position()

    def _on_pick(self, event):
        artist = event.artist
        for ann, name in self.annotations:
            if artist == ann:
                self._dragging = (ann, name)
                ann.set_fontweight('bold')
                self.fig.canvas.draw_idle()
                break

    def _on_motion(self, event):
        if self._dragging is None: return
        ann, name = self._dragging
        if event.inaxes != self.ax: return
        if event.xdata is None or event.ydata is None: return
        ann.set_position((event.xdata, event.ydata))
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self._dragging is None: return
        ann, name = self._dragging
        ann.set_fontweight('normal')
        self._dragging = None
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == 'r':
            # Reset to calculated defaults
            for ann, name in self.annotations:
                orig = self._orig_positions.get(name)
                if orig is not None: ann.set_position(orig)
            print("Reset label positions to defaults.")
            self.fig.canvas.draw_idle()
        elif event.key == 's':
            # Save current positions
            rows = []
            for ann, name in self.annotations:
                x, y = ann.get_position()
                rows.append((name, float(x), float(y)))
            with open("label_positions_saved.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['name', 'x_data', 'y_data'])
                writer.writerows(rows)
            print("Saved label positions to 'label_positions_saved.csv'.")

# ---- Helper: Load previously saved positions ----
def load_saved_positions(filename="label_positions_saved.csv"):
    """Returns a dict {name: (x, y)} from the CSV if it exists."""
    positions = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    positions[row['name']] = (float(row['x_data']), float(row['y_data']))
            print(f"Loaded {len(positions)} saved label positions.")
        except Exception as e:
            print(f"Could not load saved positions: {e}")
    return positions

# ---- Helper: Add labels (smartly checks for saved positions) ----
def add_draggable_labels(ax, mgr, data_rows, saved_positions, log_y, label_offset, x_range, max_y_grid, template):
    """
    Adds labels. If a label is found in 'saved_positions', it uses that coordinate.
    Otherwise, it calculates the default offset.
    """
    for row in data_rows:
        name = row['name']
        xi = row['x']
        anchor_y = row['y'] # The point the arrow points to (data point)
        pred_val = row['pred']

        # 1. Check if we have a saved position for this label
        if name in saved_positions:
            x_text, y_text = saved_positions[name]
            # When loading saved data coords, we use them directly
        else:
            # 2. Otherwise, calculate default position
            if log_y:
                y_text = anchor_y * (1 + label_offset[1])
                x_text = xi + label_offset[0] * x_range
            else:
                y_text = anchor_y + label_offset[1] * max_y_grid
                x_text = xi + label_offset[0] * x_range

        ann = ax.annotate(
            template.format(name=name, pred=pred_val),
            xy=(xi, anchor_y),      # Arrow tip (always locked to data)
            xytext=(x_text, y_text), # Text box position (draggable/loadable)
            textcoords='data',
            arrowprops=dict(arrowstyle='-|>', alpha=0.9, linewidth=0.8, connectionstyle="arc3,rad=0.12"),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9, linewidth=0),
            fontsize=9,
            zorder=10
        )
        mgr.register(ann, name=name)

# ---- interactive plotting function ----
def plot_fdv_vs_oi_interactive(
    df,
    new_oi_inputs=None,
    oi_scale='B',
    log_y=True,
    show_new_prediction_labels=True,
    show_df_prediction_labels=True,
    prediction_label_template="{name}: {pred:.2f}B",
    label_offset=(0.02, 0.02),
    savefile=None
):
    # 1. Load saved positions (if any)
    saved_positions = load_saved_positions()

    x = df["OI_B"].values
    y = df["FDV_B"].values

    # Fit model (Origin only)
    slope, df['pred_fit'], R2 = fit_origin(x, y)
    model_label = f"FDV = {slope:.4f} × OI  (R²={R2:.4f})"

    # Grid for curve
    x_grid = np.linspace(max(np.min(x), 1e-12), np.max(x)*1.02, 300)
    y_grid = slope * x_grid

    # Plot - INCREASED FIGSIZE HERE
    fig, ax = plt.subplots(figsize=(14, 9)) 
    
    # --- Plot observed & fit but DO NOT call legend() yet ---
    sc_obs = ax.scatter(x, y, label='Observed data', color='tab:blue', zorder=3)
    ln_fit, = ax.plot(x_grid, y_grid, label='Fit curve', color='tab:orange', linewidth=1.5, zorder=2)

    # 2. Annotate Predictions (and plot green X with label so legend informs what the green X is)
    predictions_df = None
    if new_oi_inputs:
        if isinstance(new_oi_inputs, dict):
            labels = list(new_oi_inputs.keys())
            values = np.array(list(new_oi_inputs.values()), dtype=float)
        else:
            values = np.array(new_oi_inputs, dtype=float)
            labels = [f"P{i+1}" for i in range(len(values))]

        values_in_B = values / 1e9 if oi_scale == 'raw' else values
        new_preds = slope * values_in_B

        predictions_df = pd.DataFrame({
            "label": labels,
            "OI_in_B": values_in_B,
            "Predicted_FDV_B": new_preds
        })

        # Plot predictions as green X and give that scatter a label so it appears in legend:
        sc_pred = ax.scatter(
            predictions_df["OI_in_B"],
            predictions_df["Predicted_FDV_B"],
            marker='X',
            s=80,
            color='green',
            label='Predictive FDV (green X)',
            zorder=4
        )
    else:
        sc_pred = None

    # --- Build custom legend so the green X entry appears and (optionally) appears first ---
    # Order: predictive X (if present), observed, fit
    legend_handles = []
    legend_labels = []

    if sc_pred is not None:
        legend_handles.append(sc_pred)
        legend_labels.append('Predictive FDV (green X)')

    legend_handles.append(sc_obs)
    legend_labels.append('Observed data')

    legend_handles.append(ln_fit)
    legend_labels.append('Fit curve')

    ax.legend(handles=legend_handles, labels=legend_labels, loc='best', fontsize=9)
    # ------------------------------------------------------------------------------

    ax.set_xlabel("Open Interest (billions $)")
    ax.set_ylabel("FDV (billions $)")
    ax.set_title(f"FDV vs Open Interest\n{model_label}")
    ax.grid(True, which='major', alpha=0.4)

    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:g}"))

    mgr = DraggableAnnotationManager(fig, ax)
    
    # Ranges for label offsets
    x_range = np.max(x_grid) - np.min(x_grid)
    max_y_grid = np.max(y_grid) if len(y_grid) > 0 else np.max(y)

    # 3. Annotate Observed Data (draggable)
    if show_df_prediction_labels:
        rows_to_annotate = []
        for _, row in df.iterrows():
            rows_to_annotate.append({
                'name': row['exchange'],
                'x': row['OI_B'],
                'y': row['FDV_B'], 
                'pred': row['pred_fit']
            })
        add_draggable_labels(ax, mgr, rows_to_annotate, saved_positions, log_y, label_offset, x_range, max_y_grid, prediction_label_template)

    # 4. Annotate Predictions (draggable)
    if predictions_df is not None and show_new_prediction_labels:
        rows_to_annotate = []
        for _, row in predictions_df.iterrows():
            rows_to_annotate.append({
                'name': row['label'],
                'x': row['OI_in_B'],
                'y': row['Predicted_FDV_B'],
                'pred': row['Predicted_FDV_B']
            })
        add_draggable_labels(ax, mgr, rows_to_annotate, saved_positions, log_y, label_offset, x_range, max_y_grid, prediction_label_template)

    if savefile:
        fig.savefig(savefile, dpi=150, bbox_inches='tight')

    print("\nINTERACTIVE CONTROLS:")
    print(" - Click & Drag labels to declutter.")
    print(" - Press 's' to SAVE positions (so they stay there next time).")
    print(" - Press 'r' to RESET to default positions.")
    plt.show()

    return {"predictions_df": predictions_df}

# ---- Main ----
if __name__ == "__main__":
    new_inputs_dict = {
        "Lighter": 1720000000,
        "Extended": 119750000,
        "Paradex": 535608000,
        "edgeX": 700401000,
        "Variational": 326740000,
        "Ostium": 178955000,
        "Pacifica": 57562000
    }

    out = plot_fdv_vs_oi_interactive(
        df,
        new_oi_inputs=new_inputs_dict,
        oi_scale='raw',
        show_df_prediction_labels=True
    )

    if out["predictions_df"] is not None:
        print("\nPredictions:")
        print(out["predictions_df"].to_string(index=False))
