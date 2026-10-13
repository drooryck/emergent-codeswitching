#!/usr/bin/env python
"""Reproduce the paper's Figures 1-4 from the shipped eval CSVs in figures/data/ (no cluster
runs needed). Each CSV holds final-checkpoint metrics of the trained models; see
export_eval_data.py for provenance. Outputs PDF+PNG into figures/out/.
Usage:  python make_figures.py            (all four)
        python make_figures.py 2 4        (a subset)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "out"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9})
SIZES = [("gpt2sL4", "52M", "#0072B2"), ("gpt2sL12", "108M", "#E69F00"),
         ("gpt2mL16", "232M", "#009E73"), ("gpt2mL24", "334M", "#D55E00")]
MONO_T = "Average German and English BPB ($\\downarrow$)"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def fig1():
    """Emergence ladder: mono + Denglisch switch BPB vs fraction of Chinchilla data."""
    df = pd.read_csv(DATA / "fig1_emergence_points.csv")
    lad = df[df.kind == "ladder"]
    eng = df[df.kind == "engonly_ref"].mean(numeric_only=True)
    ts = df[df.kind == "tsonly_ref"].mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, col, title in [(axes[0], "mono_avg_bpb", MONO_T),
                           (axes[1], "dgl_switch_span_bpb", "Switch BPB ($\\downarrow$)")]:
        for tag, nm, c in SIZES:
            a = lad[lad["size"] == tag].groupby("f")[col].agg(["mean", "std"]).reset_index().sort_values("f")
            ax.errorbar(a.f, a["mean"], yerr=a["std"].fillna(0), fmt="-o", color=c, ms=4,
                        mec="k", mew=.3, capsize=2.5, lw=1.5, label=nm)
        ax.set_xscale("log")
        ax.set_xticks(list({1/20: 0, 1/10: 0, 1/5: 0, 1/3: 0, 1/2: 0, 1: 0}.keys()))
        ax.set_xticklabels(["1/20", "1/10", "1/5", "1/3", "1/2", "1"])
        ax.minorticks_off(); ax.grid(alpha=.25)
        ax.set_xlabel("fraction of Chinchilla-optimal data")
        refs = [(eng.f, eng[col], "^", "English-only (52M)", None)]
        if col == "dgl_switch_span_bpb":
            refs.append((ts.f, ts[col], "*", "switch-specialist (52M)", "bl"))
        for x, y, mk, name, pos in refs:
            ax.axhline(y, ls=":", lw=1.1, color="0.4", zorder=1)
            ax.plot([x], [y], marker=mk, ms=11 if mk == "*" else 7.5, color="0.15",
                    mec="k", mew=.4, ls="none", zorder=4)
            if pos == "bl":
                ax.annotate(name, (1/20, y), textcoords="offset points", xytext=(-2, -11),
                            ha="left", fontsize=7.5, color="0.45", style="italic")
            else:
                ax.annotate(name, (x, y), textcoords="offset points", xytext=(-4, 6),
                            ha="right", fontsize=8.5, color="0.25", style="italic")
        ax.set_title(title)
    y0, y1 = axes[0].get_ylim(); axes[0].set_ylim(y0, y1 + 0.22 * (y1 - y0))
    axes[0].legend(ncol=2, loc="upper right", framealpha=.9, handlelength=1.1,
                   fontsize=8, handletextpad=0.5, borderpad=.3, labelspacing=.3, columnspacing=.9)
    save(fig, "fig1_emergence")


def _landscape_panel(fig, ax, g, zcol, title, legend=False):
    z = g[zcol].to_numpy(float)
    tri = Triangulation(np.log10(g.N_total), np.log10(g.D))
    cf = ax.tricontourf(tri, z, levels=12, cmap="viridis_r", alpha=.9)
    fig.colorbar(cf, ax=ax, fraction=.046, pad=.04)
    ax.scatter(np.log10(g.N_total), np.log10(g.D), s=10, color="k", alpha=.75, zorder=3)
    xx = np.array(ax.get_xlim() if ax.has_data() else [7, 9.2])
    xx = np.array([np.log10(g.N_total.min()) - .12, np.log10(g.N_total.max()) + .12])
    for k, st, lab in [(20, "-", "$D/N=20$"), (1, "--", "$D/N=1$"), (100, ":", "$D/N=100$")]:
        ax.plot(xx, xx + np.log10(k), "r", ls=st, lw=1.2 if k == 20 else .9,
                alpha=1 if k == 20 else .6, label=lab)
    lNs = sorted(np.log10(g.N_total.unique())); lDs = sorted(np.log10(g.D.unique()))
    labmap = {10839552: "11M", 22167552: "22M", 51393024: "52M", 108096000: "108M",
              232261632: "232M", 333031424: "334M", 510661120: "511M", 816523200: "817M"}
    ax.set_xlim(xx); ax.set_ylim(min(lDs) - .12, max(lDs) + .12)
    ax.set_xticks(lNs); ax.set_xticklabels([labmap[v] for v in sorted(g.N_total.unique())], rotation=45)
    ax.set_yticks(lDs); ax.set_yticklabels([f"{v/1e9:g}B" for v in sorted(g.D.unique())])
    ax.set_xlabel("$N$ (parameters)"); ax.set_ylabel("$D$ (tokens)"); ax.set_title(title)
    if legend:
        ax.legend(loc="lower right", framealpha=.9, handlelength=1.6, fontsize=8)


def fig2():
    """(N,D) scaling landscape over the 7x7 SoloWeb sweep."""
    df = pd.read_csv(DATA / "fig2_points.csv")
    df = df[df["size"] != "gpt2mL24"]  # 334M exists only as the fig-4 zero-dose anchor, not in the 7x7
    g = df.groupby(["N_total", "D"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    _landscape_panel(fig, axes[0], g, "mono_avg_bpb", MONO_T, legend=True)
    _landscape_panel(fig, axes[1], g, "dgl_switch_span_bpb", "Switch BPB ($\\downarrow$)")
    save(fig, "fig2_landscape")


def fig3():
    """Language-composition grid at 108M: per-language mono + directional switch BPB."""
    df = pd.read_csv(DATA / "fig3_grid_points.csv")
    g = df.groupby(["D_en", "D_de"], as_index=False).mean(numeric_only=True)
    panels = [("mono_eng_bpb", "English BPB ($\\downarrow$)"),
              ("mono_deu_bpb", "German BPB ($\\downarrow$)"),
              ("dgl_DE_span_bpb", "D$\\rightarrow$E switch BPB ($\\downarrow$)"),
              ("dgl_ED_span_bpb", "E$\\rightarrow$D switch BPB ($\\downarrow$)")]
    labs = [f"{v/1e9:g}B" for v in sorted(g.D_en.unique())]
    lx, ly = np.log10(g.D_en), np.log10(g.D_de)
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 3.6))
    for i, (ax, (col, title)) in enumerate(zip(axes, panels)):
        cf = ax.tricontourf(Triangulation(lx, ly), g[col].to_numpy(float), levels=12, cmap="viridis_r", alpha=.9)
        fig.colorbar(cf, ax=ax, fraction=.046, pad=.04)
        ax.scatter(lx, ly, s=10, color="k", alpha=.75, zorder=3)
        lo, hi = lx.min() - .1, lx.max() + .1
        ax.plot([lo, hi], [lo, hi], "r-", lw=1.2, label="$D_{en}=D_{de}$")
        ax.set_xlim(lo, hi); ax.set_ylim(ly.min() - .1, ly.max() + .1)
        ax.set_xticks(sorted(lx.unique())); ax.set_xticklabels(labs, rotation=45)
        ax.set_yticks(sorted(ly.unique())); ax.set_yticklabels(labs)
        ax.set_xlabel("$D_{en}$ (English tokens)"); ax.set_title(title)
        if i == 0:
            ax.set_ylabel("$D_{de}$ (German tokens)"); ax.legend(loc="lower right", framealpha=.9, fontsize=8)
    save(fig, "fig3_composition_grid")


def fig4():
    """Code-switched data substitution at fixed 2.74B budget: mono | switch | ACS."""
    df = pd.read_csv(DATA / "fig4_dose_points.csv")
    X0 = 1e6
    panels = [("mono_avg_bpb", MONO_T), ("dgl_switch_span_bpb", "Switch BPB ($\\downarrow$)"),
              ("acs_acc_bpb", "ACS accuracy ($\\uparrow$)")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    for i, (ax, (col, title)) in enumerate(zip(axes, panels)):
        for _, nm, c in SIZES:
            sub = df[df["size"] == nm].sort_values("dcs")
            x = np.where(sub.dcs == 0, X0, sub.dcs)
            ax.plot(x, sub[col], "-o", ms=4.5, lw=1.8, color=c, label=nm if i == 2 else None)
        ax.set_xscale("log"); ax.grid(alpha=.25, lw=.5)
        ax.set_xticks([X0, 1e7, 1e8, 1e9]); ax.set_xticklabels(["0", "$10^7$", "$10^8$", "$10^9$"])
        ax.minorticks_off(); ax.set_xlabel("CS tokens in pool ($D_{cs}$)"); ax.set_title(title)
    axes[0].set_ylabel("BPB"); axes[2].set_ylabel("accuracy")
    axes[2].axhline(0.5, ls=":", color="k", lw=1.2)
    axes[2].text(.97, 0.502, "random guess", transform=axes[2].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=9)
    axes[2].legend(loc="upper left", frameon=False)
    save(fig, "fig4_substitution")


if __name__ == "__main__":
    want = sys.argv[1:] or ["1", "2", "3", "4"]
    for k in want:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4}[k]()
