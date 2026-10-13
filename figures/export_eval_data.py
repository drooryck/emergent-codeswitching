#!/usr/bin/env python
"""Provenance script: exports the final-checkpoint eval metrics behind Figures 1-4 (and the
scaling fits) from the cluster run directories into the CSVs shipped in figures/data/ and
fits/data/. Needs the original exp12/runs tree; end users don't run this -- they get the CSVs."""
import glob, re
from pathlib import Path
import pandas as pd

E12 = Path(__import__("os").environ["CS_ROOT"])
OUT = Path(__file__).resolve().parent / "data"
FOUT = Path(__file__).resolve().parent.parent / "fits" / "data"
NNE = {"t11": 3159552, "t22": 10647552, "gpt2sL4": 28353024, "gpt2sL12": 85056000,
       "gpt2mL16": 201541632, "gpt2mL24": 302311424, "gpt2lL24": 472261120, "gpt2xlL25": 768523200}
EMB = {"t11": 30000*256, "t22": 30000*384, "gpt2sL4": 30000*768, "gpt2sL12": 30000*768,
       "gpt2mL16": 30000*1024, "gpt2mL24": 30000*1024, "gpt2lL24": 30000*1280, "gpt2xlL25": 30000*1600}
DM = {"D0p5B": .5e9, "D1B": 1e9, "D2p74B": 2.74e9, "D5p5B": 5.5e9, "D11B": 11e9, "D24B": 24e9, "D52B": 52e9}
FT = {"f1o20": 1/20, "f1o10": 1/10, "f1o5": 1/5, "f1o4": 1/4, "f1o3": 1/3, "f1o2": 1/2, "f1x": 1.0}
MET = {"mono_avg_bpb": "monoweb_avg_bpb",
       "dgl_switch_span_bpb": "denglisch_manual_switch_token_span_bpb",
       "dgl_switch_point_bpb": "denglisch_manual_switch_point_bpb",
       "dgl_ED_span_bpb": "denglisch_manual_E_to_D_switch_token_span_bpb",
       "dgl_DE_span_bpb": "denglisch_manual_D_to_E_switch_token_span_bpb",
       "ts_switch_span_bpb": "tongueswitcher_switch_token_span_bpb",
       "mono_eng_bpb": "monoweb_eng_overall_bpb", "mono_deu_bpb": "monoweb_deu_overall_bpb"}

def last(d, strict=True):
    f = glob.glob(f"{d}/dynamics_packed/*.csv")
    if not f or not glob.glob(f"{d}/models_packed/*/final"):
        return None
    df = pd.read_csv(f[0])
    if not len(df) or (strict and int(df.step.iloc[0]) != 0):
        return None
    return df.iloc[-1]

def met(r, keys=MET):
    return {k: float(r[c]) for k, c in keys.items() if c in r.index}

# --- fig 1: emergence ladder (swp0) + references ---
rows = []
for d in glob.glob(str(E12/"runs/exp12_swp0_*_s*")):
    m = re.search(r"swp0_(gpt2[a-z]+L\d+)_(f[0-9a-z]+)_s(\d+)$", Path(d).name)
    r = last(d)
    if m and m.group(2) in FT and r is not None:
        rows.append(dict(kind="ladder", size=m.group(1), f=FT[m.group(2)], seed=int(m.group(3)), **met(r)))
for d, kind, f in [("exp12_swref_engonly_gpt2sL4_s1", "engonly_ref", 1.0)] + \
                  [(f"exp12_tsonly_gpt2sL4_s{s}", "tsonly_ref", 981707264/1.028e9) for s in (1, 2, 3)]:
    r = last(str(E12/"runs"/d))
    if r is not None:
        rows.append(dict(kind=kind, size="gpt2sL4", f=f, seed=0, **met(r)))
pd.DataFrame(rows).to_csv(OUT/"fig1_emergence_points.csv", index=False)

# --- fig 2 + fits: sw7 sweep ---
rows = []
for d in glob.glob(str(E12/"runs/exp12_sw7_*_s*")):
    m = re.search(r"sw7_([a-z0-9L]+)_(D[0-9p]+B)_s(\d+)$", Path(d).name)
    r = last(d)
    if m and m.group(1) in NNE and r is not None:
        rows.append(dict(size=m.group(1), N_total=NNE[m.group(1)]+EMB[m.group(1)], N_ne=NNE[m.group(1)],
                         D=DM[m.group(2)], seed=int(m.group(3)), **met(r)))
sw7 = pd.DataFrame(rows)
sw7.to_csv(OUT/"fig2_points.csv", index=False)
FOUT.mkdir(exist_ok=True); sw7.to_csv(FOUT/"scaling_points.csv", index=False)

# --- fig 3: 108M language-composition grid (swp2b + swfact cs0 + sw7 diagonal) ---
rows = []
SW7D = {"D1B": (0.5e9, 0.5e9), "D2p74B": (1.37e9, 1.37e9), "D11B": (5.5e9, 5.5e9)}
for g in ("exp12_swp2b_gpt2sL12_en*_de*_s*", "exp12_swfact_gpt2sL12_en*_cs0_s*"):
    for d in glob.glob(str(E12/"runs"/g)):
        m = re.search(r"_en([0-9p]+)B_de([0-9p]+)B", Path(d).name)
        r = last(d)
        if m and r is not None:
            rows.append(dict(D_en=float(m.group(1).replace("p","."))*1e9,
                             D_de=float(m.group(2).replace("p","."))*1e9, **met(r)))
for dt, (a, b) in SW7D.items():
    for d in glob.glob(str(E12/"runs"/f"exp12_sw7_gpt2sL12_{dt}_s*")):
        r = last(d)
        if r is not None:
            rows.append(dict(D_en=a, D_de=b, **met(r)))
pd.DataFrame(rows).to_csv(OUT/"fig3_grid_points.csv", index=False)

# --- fig 4: substitution ladder (from the canonical points CSV incl. post-hoc ACS) ---
_fig4_src = Path(__import__("os").environ.get(
    "FIG4_POINTS_SRC", str(E12 / "export_src" / "fig_dose_soloweb_dgl_points.csv")))
pts = pd.read_csv(_fig4_src)
pts = pts[pts.budget == "D2p74B"][["size","dcs","monoweb_avg_bpb",
        "denglisch_manual_switch_token_span_bpb","acs_acc_bpb"]].rename(columns={
        "monoweb_avg_bpb":"mono_avg_bpb",
        "denglisch_manual_switch_token_span_bpb":"dgl_switch_span_bpb"})
# the canonical points CSV predates the 334M ladder; append those rows from run dynamics
CSMAP = {"cs9p6M": 9.6e6, "cs24M": 24e6, "cs48M": 48e6, "cs96M": 96e6,
         "cs240M": 240e6, "cs480M": 480e6, "csFULL": 961.657e6}
extra = []
for dcs, key in [(0.0, None)] + [(v, k) for k, v in CSMAP.items()]:
    run = ("exp12_sw7_gpt2mL24_D2p74B_s1" if key is None
           else f"exp12_swdose_gpt2mL24_D2p74B_{key}_s1")
    r = last(str(E12/"runs"/run))
    if r is None:
        continue
    acs = float(r["acs_acc_bpb"]) if "acs_acc_bpb" in r.index and pd.notna(r["acs_acc_bpb"]) else None
    if acs is None:
        f = E12/"acs_posthoc"/f"{run}.csv"
        if f.exists():
            acs = float(pd.read_csv(f)["acs_acc_bpb"].iloc[0])
    extra.append(dict(size="334M", dcs=dcs, mono_avg_bpb=float(r["monoweb_avg_bpb"]),
                      dgl_switch_span_bpb=float(r["denglisch_manual_switch_token_span_bpb"]),
                      acs_acc_bpb=acs))
pts = pd.concat([pts, pd.DataFrame(extra)], ignore_index=True)
pts.to_csv(OUT/"fig4_dose_points.csv", index=False)
print("exported:", [p.name for p in OUT.iterdir()], "+ fits/data/scaling_points.csv")
