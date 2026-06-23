#!/usr/bin/env python3
"""
analyze_cmaes_vs_eso.py
=======================
Honest head-to-head analysis of CMA-ES vs LBM-ESO from a sweep CSV.

INPUT  : a CSV with columns (names auto-detected, case-insensitive):
           bc / tag, vol_eso, dp_eso, vol_cmaes, dp_cmaes
OUTPUT : a folder `cmaes_comparison_analysis/` containing
           - summary.txt          (all statistics, plain text)
           - master_clean.csv      (per-BC table with derived columns)
           - plot_dp_scatter.png   (dp_cmaes vs dp_eso, y=x reference)
           - plot_pareto.png       (vol-vs-dp cloud, the key figure)
           - plot_dp_gap_hist.png  (distribution of dp gap %)
           - plot_vol_compare.png  (volume per method)
           - plot_win_breakdown.png(domination categories)

KEY IDEA (why this is not a naive comparison):
  CMA-ES frequently lands at LOWER volume than ESO. Because dp and volume
  trade off (less open channel -> higher dp), a raw "who has lower dp" count
  is misleading. This script classifies every BC into:
     DOMINATES   : strictly better on BOTH dp AND volume (unambiguous win)
     DOMINATED   : strictly worse on both (unambiguous loss)
     TRADE-OFF   : better on one, worse on the other (NOT comparable by a
                   single number — report honestly as a trade)
  and reports the naive dp-win-rate SEPARATELY, clearly labelled, so the
  report cannot be accused of hiding the volume confound.

USAGE:
  python analyze_cmaes_vs_eso.py --csv your_results.csv
  python analyze_cmaes_vs_eso.py --csv your_results.csv --out cmaes_comparison_analysis
"""
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ──────────────────────────────────────────────────────────────────────────────
# COLUMN AUTO-DETECTION
# ──────────────────────────────────────────────────────────────────────────────
def find_col(cols, *keywords):
    """Find the column whose lowercased name contains ALL given keywords."""
    low = {c: c.lower() for c in cols}
    for c, lc in low.items():
        if all(k in lc for k in keywords):
            return c
    return None


def load_and_standardize(csv_path):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    bc_col       = (find_col(cols, "bc") or find_col(cols, "tag")
                    or find_col(cols, "config") or cols[0])
    vol_eso_col  = find_col(cols, "vol", "eso")
    dp_eso_col   = find_col(cols, "dp", "eso")
    vol_cma_col  = find_col(cols, "vol", "cma")
    dp_cma_col   = find_col(cols, "dp", "cma")

    missing = [n for n, c in [
        ("vol_eso", vol_eso_col), ("dp_eso", dp_eso_col),
        ("vol_cmaes", vol_cma_col), ("dp_cmaes", dp_cma_col)]
        if c is None]
    if missing:
        print("ERROR: could not auto-detect these columns:", missing)
        print("Columns found in CSV:", cols)
        sys.exit(1)

    out = pd.DataFrame({
        "bc":        df[bc_col].astype(str),
        "vol_eso":   pd.to_numeric(df[vol_eso_col],  errors="coerce"),
        "dp_eso":    pd.to_numeric(df[dp_eso_col],   errors="coerce"),
        "vol_cmaes": pd.to_numeric(df[vol_cma_col],  errors="coerce"),
        "dp_cmaes":  pd.to_numeric(df[dp_cma_col],   errors="coerce"),
    })

    print(f"Detected columns:")
    print(f"  bc        <- {bc_col}")
    print(f"  vol_eso   <- {vol_eso_col}")
    print(f"  dp_eso    <- {dp_eso_col}")
    print(f"  vol_cmaes <- {vol_cma_col}")
    print(f"  dp_cmaes  <- {dp_cma_col}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# DERIVED METRICS + DOMINATION CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────
def add_derived(df, vol_tol=0.005):
    """vol_tol: volumes within this are treated as 'equal' for domination."""
    df = df.copy()

    # dp gap: positive = CMA-ES WORSE (higher dp) than ESO
    df["dp_gap_pct"] = 100.0 * (df["dp_cmaes"] - df["dp_eso"]) / df["dp_eso"]
    df["vol_diff"]   = df["vol_cmaes"] - df["vol_eso"]   # +ve = CMA-ES uses MORE

    def classify(row):
        dlo = row["dp_cmaes"] < row["dp_eso"]    # CMA-ES lower dp (better)
        dhi = row["dp_cmaes"] > row["dp_eso"]    # CMA-ES higher dp (worse)
        # volume: lower is better (less material). treat near-equal as tie.
        vlo = row["vol_cmaes"] < row["vol_eso"] - vol_tol   # CMA-ES less vol (better)
        vhi = row["vol_cmaes"] > row["vol_eso"] + vol_tol   # CMA-ES more vol (worse)
        veq = not vlo and not vhi

        # CMA-ES dominates: better-or-equal on both, strictly better on one
        if (dlo and (vlo or veq)) or (vlo and (dlo or not dhi)):
            return "CMAES_DOMINATES"
        if (dhi and (vhi or veq)) or (vhi and (dhi or not dlo)):
            return "ESO_DOMINATES"
        return "TRADE_OFF"

    df["category"] = df.apply(classify, axis=1)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY TEXT
# ──────────────────────────────────────────────────────────────────────────────
def build_summary(df):
    n = len(df)
    L = []
    A = L.append

    A("=" * 70)
    A("  CMA-ES vs LBM-ESO — COMPARISON SUMMARY")
    A("=" * 70)
    A(f"  Total boundary conditions analysed: {n}")
    A("")

    # --- naive dp comparison (clearly labelled as confounded) ---
    cma_lower_dp = int((df["dp_cmaes"] < df["dp_eso"]).sum())
    eso_lower_dp = int((df["dp_cmaes"] > df["dp_eso"]).sum())
    A("  ── NAIVE pressure-drop comparison (NOT volume-matched) ──")
    A(f"    CMA-ES lower dp : {cma_lower_dp}/{n}  ({100*cma_lower_dp/n:.0f}%)")
    A(f"    ESO    lower dp : {eso_lower_dp}/{n}  ({100*eso_lower_dp/n:.0f}%)")
    A(f"    [CAUTION] CMA-ES often reaches lower dp at LOWER volume, so this")
    A(f"    count alone overstates the win. See domination analysis below.")
    A("")

    # --- dp gap distribution ---
    g = df["dp_gap_pct"].dropna()
    A("  ── dp gap distribution (positive = CMA-ES higher/worse) ──")
    A(f"    mean   : {g.mean():+.1f}%")
    A(f"    median : {g.median():+.1f}%")
    A(f"    IQR    : [{g.quantile(.25):+.1f}%, {g.quantile(.75):+.1f}%]")
    A(f"    min    : {g.min():+.1f}%   (CMA-ES's biggest win)")
    A(f"    max    : {g.max():+.1f}%   (CMA-ES's biggest loss)")
    A("")

    # --- volume comparison ---
    vd = df["vol_diff"].dropna()
    A("  ── volume comparison (CMA-ES vol minus ESO vol) ──")
    A(f"    mean vol_diff : {vd.mean():+.4f}  (negative = CMA-ES uses LESS material)")
    A(f"    CMA-ES lower volume : {int((df['vol_diff']<0).sum())}/{n}")
    A(f"    ESO    lower volume : {int((df['vol_diff']>0).sum())}/{n}")
    A(f"    mean vol_eso   : {df['vol_eso'].mean():.4f}")
    A(f"    mean vol_cmaes : {df['vol_cmaes'].mean():.4f}")
    A("")

    # --- DOMINATION analysis (the honest headline) ---
    cats = df["category"].value_counts().to_dict()
    nd = cats.get("CMAES_DOMINATES", 0)
    ne = cats.get("ESO_DOMINATES", 0)
    nt = cats.get("TRADE_OFF", 0)
    A("  ── DOMINATION analysis (volume-aware, the HONEST headline) ──")
    A(f"    CMA-ES dominates (better dp AND volume) : {nd}/{n}  ({100*nd/n:.0f}%)")
    A(f"    ESO    dominates (better dp AND volume) : {ne}/{n}  ({100*ne/n:.0f}%)")
    A(f"    Trade-off (better on one, worse other)  : {nt}/{n}  ({100*nt/n:.0f}%)")
    A(f"    -> Only DOMINATION is an unambiguous win. Trade-offs must be")
    A(f"       reported as trades, not wins, since dp and volume trade off.")
    A("")

    # --- paired significance on dp ---
    A("  ── paired significance test on dp ──")
    if HAVE_SCIPY and len(g) > 5:
        try:
            stat, p = wilcoxon(df["dp_cmaes"].values, df["dp_eso"].values)
            verdict = ("CMA-ES significantly lower dp" if g.median() < 0
                       else "ESO significantly lower dp" if g.median() > 0
                       else "no difference")
            A(f"    Wilcoxon signed-rank p = {p:.4f}")
            A(f"    (paired dp_cmaes vs dp_eso) -> {verdict if p < 0.05 else 'NOT significant'}")
        except Exception as e:
            A(f"    Wilcoxon failed: {e}")
    else:
        A("    scipy unavailable or too few samples; skipping.")
    A("")

    # --- the trade-off cases, detailed (these need human judgement) ---
    trades = df[df["category"] == "TRADE_OFF"]
    if len(trades):
        A(f"  ── TRADE-OFF cases ({len(trades)}) — CMA-ES trades dp for volume ──")
        A(f"    (typically: CMA-ES lower volume but slightly higher dp)")
        sub = trades.copy()
        sub = sub.reindex(sub["dp_gap_pct"].abs().sort_values(ascending=False).index)
        for _, r in sub.head(10).iterrows():
            A(f"      {r['bc']:<22} dp_gap={r['dp_gap_pct']:+6.1f}%  "
              f"vol_diff={r['vol_diff']:+.4f}  "
              f"(cma dp={r['dp_cmaes']:.4f}/vol={r['vol_cmaes']:.3f}  "
              f"eso dp={r['dp_eso']:.4f}/vol={r['vol_eso']:.3f})")
        if len(sub) > 100:
            A(f"      ... and {len(sub)-10} more (see master_clean.csv)")
    A("")
    A("=" * 70)
    A("  RECOMMENDED REPORT FRAMING")
    A("=" * 70)
    A(f"  Of {n} BCs, CMA-ES strictly dominates ESO (lower dp AND lower-or-equal")
    A(f"  volume) on {nd} ({100*nd/n:.0f}%). On the {nt} trade-off cases it")
    A(f"  achieves lower volume at modestly higher dp (median gap {g.median():+.1f}%).")
    A(f"  This is an honest, volume-aware comparison — not a raw dp count.")
    A("=" * 70)
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────────────────
def plot_dp_scatter(df, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    lo = min(df["dp_eso"].min(), df["dp_cmaes"].min())
    hi = max(df["dp_eso"].max(), df["dp_cmaes"].max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            "k--", alpha=0.6, label="y = x (tie)")
    # colour by who wins on dp
    cma_win = df["dp_cmaes"] < df["dp_eso"]
    ax.scatter(df.loc[cma_win, "dp_eso"], df.loc[cma_win, "dp_cmaes"],
               c="#2a9d8f", s=40, alpha=0.8, edgecolor="k", linewidth=0.3,
               label=f"CMA-ES lower dp ({int(cma_win.sum())})")
    ax.scatter(df.loc[~cma_win, "dp_eso"], df.loc[~cma_win, "dp_cmaes"],
               c="#e76f51", s=40, alpha=0.8, edgecolor="k", linewidth=0.3,
               label=f"ESO lower dp ({int((~cma_win).sum())})")
    ax.set_xlabel("dp  (ESO)")
    ax.set_ylabel("dp  (CMA-ES)")
    ax.set_title("Pressure drop: CMA-ES vs ESO\n(points below dashed line = CMA-ES wins)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_aspect("equal", "box")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_pareto(df, path):
    """THE key figure: vol on x, dp on y. Lower-left is better.
    Shows whether CMA-ES's wins are genuine (below ESO at same vol) or
    just trade-offs (further left at higher dp)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["vol_eso"], df["dp_eso"], c="#e76f51", s=45, alpha=0.75,
               edgecolor="k", linewidth=0.3, label="ESO", marker="s")
    ax.scatter(df["vol_cmaes"], df["dp_cmaes"], c="#2a9d8f", s=45, alpha=0.75,
               edgecolor="k", linewidth=0.3, label="CMA-ES", marker="o")
    # connect each BC's pair with a faint line so the trade is visible
    for _, r in df.iterrows():
        ax.plot([r["vol_eso"], r["vol_cmaes"]], [r["dp_eso"], r["dp_cmaes"]],
                color="gray", alpha=0.25, linewidth=0.6, zorder=0)
    ax.axvline(0.20, color="navy", linestyle=":", alpha=0.5,
               label="target vol = 0.20")
    ax.set_xlabel("Volume (designable fluid fraction)")
    ax.set_ylabel("Pressure drop  dp")
    ax.set_title("Volume vs Pressure-drop trade-off\n"
                 "lower-left = better; lines connect the same BC")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_dp_gap_hist(df, path):
    g = df["dp_gap_pct"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(g, bins=25, color="#4a7", edgecolor="k", alpha=0.8)
    ax.axvline(0, color="k", linestyle="--", label="tie")
    ax.axvline(g.median(), color="red", linestyle="-",
               label=f"median {g.median():+.1f}%")
    ax.set_xlabel("dp gap %   (negative = CMA-ES lower dp = better)")
    ax.set_ylabel("number of BCs")
    ax.set_title("Distribution of dp gap (CMA-ES vs ESO)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_vol_compare(df, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["vol_eso"], bins=20, alpha=0.55, label="ESO vol",
            color="#e76f51", edgecolor="k")
    ax.hist(df["vol_cmaes"], bins=20, alpha=0.55, label="CMA-ES vol",
            color="#2a9d8f", edgecolor="k")
    ax.axvline(0.20, color="navy", linestyle=":", label="target 0.20")
    ax.set_xlabel("Volume (designable fluid fraction)")
    ax.set_ylabel("number of BCs")
    ax.set_title("Volume distribution by method")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_win_breakdown(df, path):
    order = ["CMAES_DOMINATES", "TRADE_OFF", "ESO_DOMINATES"]
    labels = ["CMA-ES\ndominates", "Trade-off", "ESO\ndominates"]
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    counts = [int((df["category"] == c).sum()) for c in order]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="k")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, c + 0.5, str(c),
                ha="center", fontweight="bold")
    ax.set_ylabel("number of BCs")
    ax.set_title("Volume-aware outcome breakdown\n"
                 "(only domination is an unambiguous win)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="input CSV path")
    ap.add_argument("--out", default="cmaes_comparison_analysis",
                    help="output folder")
    ap.add_argument("--vol_tol", type=float, default=0.002,
                    help="volumes within this are 'equal' for domination")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_and_standardize(args.csv)
    before = len(df)
    df = df.dropna(subset=["vol_eso", "dp_eso", "vol_cmaes", "dp_cmaes"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing/NA values "
              f"(e.g. CMA-ES failed on those BCs).")
    if len(df) == 0:
        print("No valid rows after dropping NA. Check the CSV.")
        sys.exit(1)

    df = add_derived(df, vol_tol=args.vol_tol)

    # save the clean per-BC table
    df.to_csv(os.path.join(args.out, "master_clean.csv"), index=False)

    # summary text
    summary = build_summary(df)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print("\n" + summary + "\n")

    # plots
    plot_dp_scatter(df,   os.path.join(args.out, "plot_dp_scatter.png"))
    plot_pareto(df,       os.path.join(args.out, "plot_pareto.png"))
    plot_dp_gap_hist(df,  os.path.join(args.out, "plot_dp_gap_hist.png"))
    plot_vol_compare(df,  os.path.join(args.out, "plot_vol_compare.png"))
    plot_win_breakdown(df,os.path.join(args.out, "plot_win_breakdown.png"))

    print(f"All outputs written to: {args.out}/")
    print("  summary.txt, master_clean.csv, and 5 PNG plots")
    if dropped:
        print(f"  NOTE: {dropped} BCs excluded (CMA-ES produced no valid result).")
        print(f"        Report this feasibility rate honestly.")


if __name__ == "__main__":
    main()
'''

import os
import pandas as pd
import matplotlib.pyplot as plt

# --- edit this to your CSV path ---
CSV_PATH = "eso_cmaes.csv"          # columns: tag, vol_eso, dp_eso, vol_cmaes, dp_cmaes
OUT_PATH = "cmaes_scatter.png"

os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

df = pd.read_csv(CSV_PATH).dropna(subset=["dp_eso", "dp_cmaes"])

# axis range so the parity line spans the data
m = max(df.dp_eso.max(), df.dp_cmaes.max())
lo = min(df.dp_eso.min(), df.dp_cmaes.min())
pad = (m - lo) * 0.05

# who wins on dp (below the line = CMA-ES lower dp)
cma_win = df.dp_cmaes < df.dp_eso
n_cma   = int(cma_win.sum())
n_eso   = int((~cma_win).sum())

fig, ax = plt.subplots(figsize=(5.2, 5.2))

# parity / separating line
ax.plot([lo - pad, m + pad], [lo - pad, m + pad],
        "k--", lw=1, alpha=0.7, label="parity ($\\Delta p$ equal)")

# points, coloured by which method wins
ax.scatter(df.dp_eso[cma_win],  df.dp_cmaes[cma_win],
           s=22, alpha=0.7, color="#2a9d8f", edgecolor="k", linewidth=0.3,
           label=f"CMA-ES lower ($n={n_cma}$)")
ax.scatter(df.dp_eso[~cma_win], df.dp_cmaes[~cma_win],
           s=22, alpha=0.7, color="#e76f51", edgecolor="k", linewidth=0.3,
           label=f"ESO lower ($n={n_eso}$)")

ax.set_xlabel(r"$\Delta p$  (LBM-ESO)")
ax.set_ylabel(r"$\Delta p$  (CMA-ES)")
ax.set_aspect("equal", "box")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"saved {OUT_PATH}  |  CMA-ES lower dp on {n_cma}/{len(df)}, ESO lower on {n_eso}/{len(df)}")