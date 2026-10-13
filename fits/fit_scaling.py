#!/usr/bin/env python
"""Scaling-law fits behind the paper's fit tables (SoloWeb 7x7 sweep; Denglisch + TS evals).
Reads the shipped per-run metrics (fits/data/scaling_points.csv; see figures/export_eval_data.py
for provenance) and reproduces the published fits exactly.

Protocol (documented in the paper's appendix): one point per (size, budget) cell with seed
replicates averaged; cells with D/N_ne < 2 dropped (N_ne = non-embedding params; 46 cells
remain); the regressor N is total parameters; L = E + A N^-alpha + B D^-beta fitted with a
Huber loss (f_scale 1e-2) on log residuals via bounded trust-region least squares; 300-fold
nonparametric bootstrap percentile CIs on the exponents. ARE% is a held-out extrapolation
check: refit without the 817M cells and report the mean |log Lhat - log L| x100 on them
(log-space, which approximates relative BPB error for small errors)."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "data" / "scaling_points.csv")
df = df[df["size"] != "gpt2mL24"]  # fig-4 anchor, not part of the 7x7 sweep
df = df.rename(columns={"mono_avg_bpb": "mono_avg", "dgl_switch_span_bpb": "dgl_span",
                        "dgl_switch_point_bpb": "dgl_point", "dgl_ED_span_bpb": "dgl_ED_span",
                        "dgl_DE_span_bpb": "dgl_DE_span", "ts_switch_span_bpb": "ts_span"})
METRICS = ["mono_avg", "dgl_span", "dgl_ED_span", "dgl_DE_span", "dgl_point", "ts_span"]
cell = df.groupby(["size", "N_total", "N_ne", "D"], as_index=False)[METRICS].mean()
c = cell[cell.D / cell.N_ne >= 2].reset_index(drop=True)
N, D = c.N_total.to_numpy(float), c.D.to_numpy(float)
big = (c["size"] == "gpt2xlL25").to_numpy()
print(f"cells: {len(c)} (dropped {len(cell)-len(c)} with D/N_ne < 2)")


def fit(N, D, y):
    def resid(th):
        E, a, al, b, be = th
        return np.log(np.maximum(E + np.exp(a)*N**(-al) + np.exp(b)*D**(-be), 1e-9)) - np.log(y)
    x0 = [0.5, 2, .2, 3, .2]; lo = [0, -35, .02, -35, .02]; hi = [y.min(), 35, 2, 35, 2]
    return least_squares(resid, x0, bounds=(lo, hi), loss="huber", f_scale=1e-2, max_nfev=40000).x


def pred(th, N, D):
    E, a, al, b, be = th
    return E + np.exp(a)*N**(-al) + np.exp(b)*D**(-be)


rng = np.random.default_rng(0)
print("| metric | E | alpha [CI] | beta [CI] | ARE% | RMSE(logL) |")
print("|---|---|---|---|---|---|")
for mc in METRICS:
    y = c[mc].to_numpy(float)
    th = fit(N, D, y)
    rmse = float(np.sqrt(np.mean((np.log(pred(th, N, D)) - np.log(y))**2)))
    th2 = fit(N[~big], D[~big], y[~big])
    are = float(np.mean(np.abs(np.log(pred(th2, N[big], D[big])) - np.log(y[big]))) * 100)
    bt = np.full((300, 2), np.nan)
    for k in range(300):
        i = rng.integers(0, len(y), len(y))
        try:
            tb = fit(N[i], D[i], y[i]); bt[k] = [tb[2], tb[4]]
        except Exception:
            pass
    alo, ahi = np.nanpercentile(bt[:, 0], [2.5, 97.5]); blo, bhi = np.nanpercentile(bt[:, 1], [2.5, 97.5])
    print(f"| {mc} | {th[0]:.2f} | {th[2]:.3f} [{alo:.2f},{ahi:.2f}] | "
          f"{th[4]:.3f} [{blo:.2f},{bhi:.2f}] | {are:.1f} | {rmse:.4f} |")
