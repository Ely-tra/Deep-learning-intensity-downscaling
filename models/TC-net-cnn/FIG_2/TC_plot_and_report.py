"""
===============================================================================
AUTHOR:      Minh Khanh Luong
DATE:        (2025-11-09)
===============================================================================
DESCRIPTION:
    This script generates comparative plots for tropical cyclone metrics 
    (VMAX, PMIN, or RMW) using four data sources plus a linear 
    regression baseline. It reads precomputed CNN text reports and MERRA2-
    derived reference fields, computes linear model predictions from 
    training/testing data, and produces a two-panel figure showing:

        (a) Boxplots comparing all five series:
            [MERRA2, Linear Regression, CNN, CNN_ALL, Truth]
        (b) Scatter plots of predictions vs truth.

    The output figure filename includes RMSE and MAE values for each model.

USAGE EXAMPLES:
    # Default (VMAX)
    python TC_plot_and_report.py --metric vmax

    # PMIN
    python plot_five_with_linear_baseline.py --metric pmin

    # RMW (radius of maximum wind, plotted in nautical miles)
    python plot_five_with_linear_baseline.py --metric rmw --name RMW_rand

INPUT FILES:
    • ref_x (.npy/.npz)     : Reference 4D array (N, 5, lon, lat) 
                              → channels: [u_sfc, v_sfc, slp, lat, lon]
    • train_x / train_y     : Training data (N, 13, H, W) and (N, 3)
    • test_x / test_y       : Testing data (M, 13, H, W) and (M, 3)
    • text_report/*.txt     : CNN output files (pred,true per line) for each metric:
                                - VMAX_CNN_RAND.txt / VMAX_ALL_RAND.txt
                                - PMIN_CNN_RAND.txt / PMIN_ALL_RAND.txt
                                - RMW_CNN_RAND.txt  / RMW_ALL_RAND.txt

OUTPUTS:
    • A PNG figure saved in "reports/" with RMSE and MAE encoded in the filename.
      Example: fig_RMW_rand_rmse1.23_1.56_1.78_mae0.89_1.02_1.11.png

METRIC UNITS:
    • VMAX → knots (kt)
    • PMIN → millibar / hPa (mb)
    • RMW  → nautical miles (nm)

NOTES:
    - The MERRA2-based reference statistics (VMAX, PMIN, RMW) are derived using
      a 6°×5° window centered on the storm and converted to the units above.
    - The script uses NumPy and TensorFlow for metrics, and matplotlib for plotting.
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

# ---------- Defaults ----------
DEFAULTS = {
    "ref_x":  "randomdata/temp/ref_x_hhffxxkk.npy",
    "train_x": "randomdata/temp/train_x_hhffxxkk.npy",
    "train_y": "randomdata/temp/train_y_hhffxxkk.npy",
    "test_x":  "randomdata/temp/test_x_hhffxxkk.npy",
    "test_y":  "randomdata/temp/test_y_hhffxxkk.npy",

    "vm":  "text_report/VMAX_CNN_RAND.txt",
    "vma": "text_report/VMAX_ALL_RAND.txt",
    "pm":  "text_report/PMIN_CNN_RAND.txt",
    "pma": "text_report/PMIN_ALL_RAND.txt",
    "rm":  "text_report/RMW_CNN_RAND.txt",
    "rma": "text_report/RMW_ALL_RAND.txt",

    "name": "FIG",
    "unit": "knots",      # will be overridden by metric-specific defaults if not set
    "report_dir": "reports",
}

def build_parser():
    p = argparse.ArgumentParser(
        description="Plot five-series figure with linear baseline and save with RMSE/MAE in filename."
    )
    # Which metric to plot
    p.add_argument(
        "--metric",
        choices=["vmax", "pmin", "rmw"],
        default="vmax",
        help="Target metric to plot (controls which column of stats and which text reports to read).",
    )

    # Text reports (pred,true per line under header)
    p.add_argument("--vm",  default=DEFAULTS["vm"],  help="VMAX single-model text report")
    p.add_argument("--vma", default=DEFAULTS["vma"], help="VMAX ALL-model text report")
    p.add_argument("--pm",  default=DEFAULTS["pm"],  help="PMIN single-model text report")
    p.add_argument("--pma", default=DEFAULTS["pma"], help="PMIN ALL-model text report")
    p.add_argument("--rm",  default=DEFAULTS["rm"],  help="RMW single-model text report")
    p.add_argument("--rma", default=DEFAULTS["rma"], help="RMW ALL-model text report")

    # Ref array for MERRA-derived stats
    p.add_argument("--ref-x",  default=DEFAULTS["ref_x"],  help="Path to ref_x .npy/.npz with shape (N,5,lon,lat)")

    # Linear baseline data
    p.add_argument("--train-x", default=DEFAULTS["train_x"])
    p.add_argument("--train-y", default=DEFAULTS["train_y"])
    p.add_argument("--test-x",  default=DEFAULTS["test_x"])
    p.add_argument("--test-y",  default=DEFAULTS["test_y"])

    # Figure metadata
    p.add_argument("--name",       default=DEFAULTS["name"],       help="Figure base name (added to filename).")
    p.add_argument("--unit",       default=DEFAULTS["unit"],       help="Y-axis unit label.")
    p.add_argument("--report-dir", default=DEFAULTS["report_dir"], help="Output directory to save figure.")
    return p

# =================== Metrics ===================
def root_mean_squared_error(y_true, y_pred):
    m = tf.keras.metrics.RootMeanSquaredError()
    m.update_state(y_true, y_pred)
    return m.result().numpy()

def MAE(y_true, y_pred):
    m = tf.keras.metrics.MeanAbsoluteError()
    m.update_state(y_true, y_pred)
    return m.result().numpy()

# =================== Plotting ===================
def plot_five_with_linear(arrays, labels, name, unit, y_pred, report_directory=None):
    """
    arrays: list/tuple of 4 arrays [A, B, C, D]
      A: MERRA-derived target (from ref_x stats)
      B: CNN (single) prediction series
      C: CNN_ALL prediction series
      D: Truth series
    labels: list/tuple of 4 labels matching A,B,C,D
    y_pred: 1D numpy array (linear regression predictions) to insert as the 2nd box
    """
    if len(arrays) != 4 or len(labels) != 4:
        raise ValueError("Provide exactly four base arrays and four labels (A,B,C,D).")

    # Unpack and ensure numpy
    A, B, C, D = (np.asarray(x) for x in arrays)
    y_pred = np.asarray(y_pred)

    # Build 5-series order: [A, y_pred, B, C, D]
    series = [A, y_pred, B, C, D]
    lab5   = [labels[0], "Linear\nRegression", labels[1], labels[2], labels[3]]

    # Global mask (kept as in your original; if you want per-series NaN masking, we can switch)
    mask = np.ones_like(series[0], dtype=bool)
    for s in series:
        if s.shape != series[0].shape:
            raise ValueError("All arrays must be the same length along the sample axis.")
        mask &= (s != -1)
    series = [s[mask] for s in series]
    A, YLIN, B, C, D = series

    # Metrics for all three predictors vs truth
    rmse_lin = root_mean_squared_error(D, YLIN)
    mae_lin  = MAE(D, YLIN)
    rmse1    = root_mean_squared_error(D, B)
    mae1     = MAE(D, B)
    rmse2    = root_mean_squared_error(D, C)
    mae2     = MAE(D, C)

    colors = ['g', 'y', 'b', 'r', 'k']
    fig, axs = plt.subplots(1, 2, figsize=(14, 6),
                            gridspec_kw={'width_ratios': [1.2, 1]})

    # (a) boxplots
    bps = axs[0].boxplot([A, YLIN, B, C, D], patch_artist=True)
    for patch, color in zip(bps['boxes'], colors):
        patch.set_facecolor(color)
    axs[0].set_xticklabels(lab5, fontsize=14, rotation=0)
    axs[0].set_ylabel(unit, fontsize=16)
    axs[0].grid(True)
    axs[0].text(0.95, 0.05, '(a)', transform=axs[0].transAxes,
                fontsize=16, va='bottom', ha='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # (b) scatter: B, C, YLIN vs D
    for arr, lbl, col in zip([B, C, YLIN],
                             [lab5[2], lab5[3], "Linear Regression"],
                             [colors[2], colors[3], colors[1]]):
        axs[1].scatter(D, arr, c=col, label=lbl, alpha=0.7, edgecolors='none')

    mn, mx = D.min(), D.max()
    axs[1].plot([mn, mx], [mn, mx], 'r-', alpha=0.5)
    axs[1].set_xlabel(f"{lab5[4]}", fontsize=16)  # truth label
    axs[1].set_ylabel("Prediction", fontsize=16)
    axs[1].grid(True)
    axs[1].legend(fontsize=12, loc='upper left')
    axs[1].text(0.95, 0.05, '(b)', transform=axs[1].transAxes,
                fontsize=16, va='bottom', ha='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    axs[1].tick_params(labelsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save with metrics in filename
    if report_directory is not None:
        os.makedirs(report_directory, exist_ok=True)
        def fmt(x): return f"{x:.3f}"
        fname = os.path.join(
            report_directory,
            f"fig_{name}_rmse{fmt(rmse_lin)}_{fmt(rmse1)}_{fmt(rmse2)}"
            f"_mae{fmt(mae_lin)}_{fmt(mae1)}_{fmt(mae2)}.png"
        )
        plt.savefig(fname, dpi=300)
        print(f"Saved figure to: {fname}")
    else:
        plt.show()

# =================== Parsing helpers & data prep ===================
def extract_predictions(filename):
    """Read lines after 'Predictions vs Actual Values:' as 'pred,true' floats."""
    pred, true = [], []
    in_data_block = False
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Predictions vs Actual Values:"):
                in_data_block = True
                continue
            if in_data_block:
                if not line:
                    break
                try:
                    pval, tval = map(float, line.split(','))
                    pred.append(pval)
                    true.append(tval)
                except ValueError:
                    continue
    return np.array(pred), np.array(true)

def haversine(lat1, lon1, lat2, lon2, radius_km=6371.0):
    """Great-circle distance in nautical miles (inputs in degrees)."""
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    Δφ     = φ2 - φ1
    Δλ     = np.radians(lon2 - lon1)
    a      = np.sin(Δφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(Δλ/2)**2
    d_km   = 2 * radius_km * np.arcsin(np.sqrt(a))
    return d_km  # nm

def _load_array(path: str) -> np.ndarray:
    """Load .npy or .npz (single-array) file."""
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        with np.load(path) as z:
            if len(z.files) != 1:
                raise ValueError(f"{path} contains multiple arrays; please provide a single-array .npz.")
            return z[z.files[0]]
    raise ValueError(f"Unsupported file type for {path}; expected .npy or .npz")

def train_and_eval_linear_models(x_train_path, y_train_path, x_test_path, y_test_path):
    """
    Average X over last two dims -> (N, 13); add bias; solve least squares to 3 targets.
    Returns y_pred_test, y_true_test with shape (M, 3).
    """
    X_tr = _load_array(x_train_path)
    Y_tr = _load_array(y_train_path)
    X_te = _load_array(x_test_path)
    Y_te = _load_array(y_test_path)

    if X_tr.ndim != 4 or X_te.ndim != 4:
        raise ValueError("X_train and X_test must be 4D arrays like (N, 13, H, W).")
    if Y_tr.ndim != 2 or Y_te.ndim != 2:
        raise ValueError("Y_train and Y_test must be 2D arrays like (N, 3).")
    if X_tr.shape[1] != 13 or X_te.shape[1] != 13:
        raise ValueError("Expected 13 channels in X.")
    if Y_tr.shape[1] != 3 or Y_te.shape[1] != 3:
        raise ValueError("Targets must have 3 columns.")

    Xtr_red = X_tr.mean(axis=(-2, -1))
    Xte_red = X_te.mean(axis=(-2, -1))
    N, M = Xtr_red.shape[0], Xte_red.shape[0]
    Xtr_aug = np.concatenate([np.ones((N,1), dtype=Xtr_red.dtype), Xtr_red], axis=1)
    Xte_aug = np.concatenate([np.ones((M,1), dtype=Xte_red.dtype), Xte_red], axis=1)
    B, *_ = np.linalg.lstsq(Xtr_aug, Y_tr, rcond=None)
    Y_pred_te = Xte_aug @ B
    return Y_pred_te, Y_te

# =================== CLI ===================
def extract_wind_slp_stats(ref_x):
    """
    ref_x: np.ndarray, shape (N,5,lon,lat)
      channel 0 = u_sfc
      channel 1 = v_sfc
      channel 2 = slp
      channel 3 = latitudes (deg), same shape (lon,lat)
      channel 4 = longitudes (deg), same shape (lon,lat)
    Returns: np.ndarray shape (N,3) of [max_wind, min_slp, distance_nm]
    """
    N, _, nlon, nlat = ref_x.shape
    # pixel radii: exactly ±3° in each direction
    lon_res = 0.5
    lat_res = 0.625
    i_radius = int(np.ceil(3.0 / lon_res))    # 3°/0.5° → 6 pixels east/west
    j_radius = int(np.ceil(3.0 / lat_res))    # 3°/0.625° → 4 pixels north/south

    # Pre‑extract static lat/lon grids (identical for every sample)
    lat_grid = ref_x[0,3]   # shape (nlon,nlat)
    lon_grid = ref_x[0,4]

    out = np.zeros((N,3), dtype=float)  # [max_wind, min_slp, dist_km]

    for idx in range(N):
        u   = ref_x[idx,0]
        v   = ref_x[idx,1]
        slp = ref_x[idx,2]

        # 1) wind speed magnitude
        wind = np.sqrt(u*u + v*v)
        # — if the whole wind field is NaN, set everything to 1
        if np.all(np.isnan(wind)):
            out[idx] = [-1, -1, -1]
            continue
        # 2) max‐wind location
        flat_idx = np.nanargmax(wind)
        i0, j0  = np.unravel_index(flat_idx, wind.shape)
        max_w   = wind[i0,j0]

     # define a (2*i_radius+1)×(2*j_radius+1) window centered on (i0,j0)
        i1 = max(0,               i0 - i_radius)
        i2 = min(nlon, i0 + i_radius + 1)  # +1 to include the upper bound
        j1 = max(0,               j0 - j_radius)
        j2 = min(nlat, j0 + j_radius + 1)

        sub_slp = slp[i1:i2, j1:j2]
        # 4) find min‐SLP in that window
        sub_flat = np.nanargmin(sub_slp)
        di, dj   = np.unravel_index(sub_flat, sub_slp.shape)
        i_min    = i1 + di
        j_min    = j1 + dj
        min_p    = slp[i_min,j_min]

        # 5) compute great‐circle distance
        lat0, lon0 = lat_grid[i0,j0], lon_grid[i0,j0]
        latm, lonm = lat_grid[i_min,j_min], lon_grid[i_min,j_min]
        dist_km    = haversine(lat0, lon0, latm, lonm)
        # convert speed m/s → knots, and pressure Pa → bar
        max_w_kt   = max_w   * 1.94384
        min_p_bar  = min_p   / 1e2
        
        out[idx] = [max_w_kt, min_p_bar, dist_km*0.539957]

    return out
def _load_array(path: str) -> np.ndarray:
    """Load .npy or .npz (single-array) file."""
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        with np.load(path) as z:
            if len(z.files) != 1:
                raise ValueError(f"{path} contains multiple arrays; please provide a single-array .npz.")
            return z[z.files[0]]
    raise ValueError(f"Unsupported file type for {path}; expected .npy or .npz")



def main():
    args = build_parser().parse_args()

    # 1) Load ref_x and compute stats
    ref_x = _load_array(args.ref_x)
    stats = extract_wind_slp_stats(ref_x)  # [:,0]=VMAX, [:,1]=PMIN, [:,2]=RMW (as in your current code)

    # 2) Load linear baseline and choose the column index
    y_pred_lin, y_true_lin = train_and_eval_linear_models(args.train_x, args.train_y, args.test_x, args.test_y)

    metric = args.metric.lower()
    if metric == "vmax":
        idx = 0
        if not (args.vm and args.vma):
            raise ValueError("For vmax, please provide --vm and --vma")
        B_pred, B_true = extract_predictions(args.vm)   # CNN single
        C_pred        = extract_predictions(args.vma)[0]  # CNN ALL (pred only)
        label_name = "VMAX"
        unit = args.unit or "knots"
    elif metric == "pmin":
        idx = 1
        if not (args.pm and args.pma):
            raise ValueError("For pmin, please provide --pm and --pma")
        B_pred, B_true = extract_predictions(args.pm)
        C_pred        = extract_predictions(args.pma)[0]
        label_name = "PMIN"
        unit = args.unit or "mb"
    else:  # rmw
        idx = 2
        if not (args.rm and args.rma):
            raise ValueError("For rmw, please provide --rm and --rma")
        B_pred, B_true = extract_predictions(args.rm)
        C_pred        = extract_predictions(args.rma)[0]
        label_name = "RMW"
        unit = args.unit or "nm"

    # 3) Assemble the four base series [A, B, C, D]
    A = stats[:, idx]               # MERRA-derived
    B = B_pred                      # CNN (single)
    C = C_pred                      # CNN (all)
    D = B_true                      # Truth (from the same file as B)

    # 4) Choose the linear prediction column matching the metric
    y_pred_col = y_pred_lin[:, idx]

    # 5) Plot + save
    plot_five_with_linear(
        arrays=[A, B, C, D],
        labels=['MERRA2', 'CNN', 'CNN_ALL', 'Truth'],
        name=f"{label_name}_{args.name}",
        unit=unit,
        y_pred=y_pred_col,
        report_directory=args.report_dir
    )

if __name__ == "__main__":
    main()
