#!/usr/bin/env python
"""Generation plots round 3 (oct10): multiword-switch (>=2-word minority run) metric ONLY.
Outputs (also copied by hand to latex/appendix_plots/):
  gen_grid_mw      -- single-panel (N,D) map of multiword-switch share (T=0.6, TS prompts)
  gen_inout_violin -- multiword share by input-prompt switch count: violin over the 245
                      per-(model,seed) shares per bin, jittered points colored by size
If sw7_nopropn_runs.csv exists, uses the PROPN-excluded run metric;
otherwise falls back to ts_max_minority_run (includes proper nouns; caption notes it)."""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

TS = Path(os.environ["CS_ROOT"]) / "generation_out"
HERE = TS
LAB = {"t11": "11M", "t22": "22M", "gpt2sL4": "52M", "gpt2sL12": "108M",
       "gpt2mL16": "232M", "gpt2lL24": "511M", "gpt2xlL25": "817M"}
NTOT = {"11M": 10.84e6, "22M": 22.2e6, "52M": 51.4e6, "108M": 108.1e6,
        "232M": 232.3e6, "511M": 510.7e6, "817M": 816.5e6}
DMAP = {"D0p5B": .5e9, "D1B": 1e9, "D2p74B": 2.74e9, "D5p5B": 5.5e9,
        "D11B": 11e9, "D24B": 24e9, "D52B": 52e9}
ORDER = ["11M", "22M", "52M", "108M", "232M", "511M", "817M"]
CB = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(ORDER)))
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "legend.fontsize": 8.5,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5})

KEYS = ["run", "set", "prompt_id", "temp", "seed"]
d = pd.read_csv(TS / "sw7_temp_sweep_tagged.csv",
                usecols=KEYS + ["ts_max_minority_run"], low_memory=False)
np_file = TS / "sw7_nopropn_runs.csv"
if np_file.exists():
    npd = pd.read_csv(np_file, usecols=KEYS + ["ts_max_minority_run_np"], low_memory=False)
    d = d.merge(npd, on=KEYS, how="left")
    d["mw"] = (d.ts_max_minority_run_np >= 2).astype(float)
    SRC = "nopropn"
else:
    d["mw"] = (d.ts_max_minority_run >= 2).astype(float)
    SRC = "incl-propn (retag pending)"
print("multiword source:", SRC)
d["size"] = d["run"].str.extract(r"sw7_([a-z0-9L]+)_D")[0].map(LAB)
d["D"] = d["run"].str.extract(r"_(D[0-9p]+B)")[0].map(DMAP)
t6 = d[(d.temp == 0.6) & d["set"].str.startswith("ts")].copy()

# ---------- single-panel (N, D) map of multiword share ----------
cells = t6.groupby(["size", "D"])["mw"].mean().reset_index()
cells["lN"] = np.log10(cells["size"].map(NTOT)); cells["lD"] = np.log10(cells.D)
fig, ax = plt.subplots(figsize=(4.6, 3.7))
z = cells.mw.to_numpy(float) * 100
tri = Triangulation(cells.lN.to_numpy(), cells.lD.to_numpy())
cf = ax.tricontourf(tri, z, levels=12, cmap="viridis", alpha=.9)
fig.colorbar(cf, ax=ax, fraction=.046, pad=.04, label="% of completions")
ax.scatter(cells.lN, cells.lD, s=10, color="k", alpha=.75, zorder=3)
ax.set_xticks(sorted(cells.lN.unique())); ax.set_xticklabels(ORDER, rotation=45)
ax.set_yticks(sorted(cells.lD.unique()))
ax.set_yticklabels([f"{v/1e9:g}B" for v in sorted(cells.D.unique())])
ax.set_xlabel("$N$ (parameters)"); ax.set_ylabel("$D$ (tokens)")
ax.set_title("Completions with a $\\geq$2-word switch")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"gen_grid_mw.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig); print("wrote gen_grid_mw")

# ---------- violin: multiword share by input-prompt switch count ----------
t6["n_in"] = t6["set"].str.extract(r"ts(\d)sw")[0].astype(int)
per = t6.groupby(["n_in", "size", "run", "seed"])["mw"].mean().reset_index()
fig, ax = plt.subplots(figsize=(5.6, 3.6))
data = [per[per.n_in == k]["mw"].to_numpy() * 100 for k in (1, 2, 3, 4)]
vp = ax.violinplot(data, positions=[1, 2, 3, 4], widths=0.8, showmedians=True,
                   showextrema=False)
for b in vp["bodies"]:
    b.set_facecolor("0.82"); b.set_alpha(.8)
vp["cmedians"].set_color("0.25")
rng = np.random.default_rng(0)
for c, sz in zip(CB, ORDER):
    sub = per[per["size"] == sz]
    ax.scatter(sub.n_in + rng.uniform(-0.22, 0.22, len(sub)), sub.mw * 100,
               s=7, color=c, alpha=.65, label=sz, zorder=3, edgecolors="none")
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("switches in input prompt")
ax.set_ylabel("% completions with $\\geq$2-word switch")
ax.grid(alpha=.3, axis="y")
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.17),
          handletextpad=0.1, columnspacing=0.8, markerscale=1.6)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"gen_inout_violin.{ext}", dpi=200, bbox_inches="tight")
print("wrote gen_inout_violin")
