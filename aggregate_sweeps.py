
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ======================================================================
# Geometry constants — MUST match how the sweep was run.
# ======================================================================
Nx = Ny = 64
WALL = 4
PORT_HEIGHT = 6          # height-6 ports (matches latgrad / diversity / fixed harness)

_WALLMAP = {"l": "left", "r": "right", "t": "top", "b": "bottom"}


# ======================================================================
# Tag -> ports
# ======================================================================
def parse_ports_from_tag(tag):
    """'il39_il52_ot18' -> list of port dicts with slice ranges."""
    ports = []
    if not tag:
        return ports
    for tok in str(tag).split("_"):
        if len(tok) < 3:
            continue
        ptype = "inlet" if tok[0] == "i" else "outlet" if tok[0] == "o" else None
        wall = _WALLMAP.get(tok[1])
        if ptype is None or wall is None:
            continue
        try:
            center = int(tok[2:])
        except ValueError:
            continue
        lo = max(WALL, center - PORT_HEIGHT // 2)
        hi = min(Ny - WALL, center + PORT_HEIGHT // 2)
        ports.append({"type": ptype, "wall": wall,
                      "range": slice(lo, hi), "center": center})
    return ports


# ======================================================================
# Column auto-detection (headers drift across runs)
# ======================================================================
def find_col(cols, *must_contain, exclude=()):
    for c in cols:
        cl = c.lower()
        if all(k in cl for k in must_contain) and not any(e in cl for e in exclude):
            return c
    return None


def resolve_columns(df):
    cols = list(df.columns)
    return {
        "tag":     find_col(cols, "tag") or find_col(cols, "ports_desc") or cols[0],
        "eso_dp":  find_col(cols, "eso", "dp"),
        "eso_vol": find_col(cols, "eso", "vol"),
        "lg_dp":   find_col(cols, "grad", "dp") or find_col(cols, "latent", "dp") or find_col(cols, "latgrad", "dp"),
        "lg_vol":  find_col(cols, "grad", "vol") or find_col(cols, "latent", "vol") or find_col(cols, "latgrad", "vol"),
    }


# ======================================================================
# (A) TABLE
# ======================================================================
def build_master_table(root, out_csv):
    pattern = os.path.join(root, "sweep_results", "best", "2026*", "sweep_results.csv")
    csvs = sorted(glob.glob(pattern))
    if not csvs:
        print(f"[table] no CSVs matched: {pattern}")
        print("        (check --root; it should CONTAIN sweep_results/)")
        return None

    print(f"[table] found {len(csvs)} run CSV(s):")
    for c in csvs:
        print(f"        {c}")

    frames = []
    for csv_path in csvs:
        run_id = os.path.basename(os.path.dirname(csv_path))
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[table] could not read {csv_path}: {e}")
            continue
        if df.empty:
            continue

        cmap = resolve_columns(df)
        if not frames:
            print("[table] auto-detected columns (from first CSV):")
            for k, v in cmap.items():
                flag = "" if v else "   <-- NOT FOUND"
                print(f"        {k:8s} -> {v}{flag}")

        def col(key):
            c = cmap[key]
            return pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series([np.nan] * len(df))

        out = pd.DataFrame()
        out["run"]     = [run_id] * len(df)
        out["tag"]     = df[cmap["tag"]] if cmap["tag"] in df else ""
        out["eso_vol"] = col("eso_vol")
        out["eso_dp"]  = col("eso_dp")
        out["lg_vol"]  = col("lg_vol")
        out["lg_dp"]   = col("lg_dp")
        out["dp_gap_pct"] = 100.0 * (out["lg_dp"] - out["eso_dp"]) / out["eso_dp"]
        out["vol_diff"]   = out["lg_vol"] - out["eso_vol"]
        out["winner"] = np.where(out["lg_dp"] < out["eso_dp"], "latgrad",
                          np.where(out["lg_dp"] > out["eso_dp"], "eso", "tie"))
        frames.append(out)

    if not frames:
        print("[table] nothing aggregated.")
        return None

    master = pd.concat(frames, ignore_index=True)
    master.to_csv(out_csv, index=False)
    print(f"\n[table] master table -> {out_csv}  ({len(master)} rows)")

    valid = master.dropna(subset=["dp_gap_pct"])
    if len(valid):
        print("\n" + "=" * 64)
        print("SUMMARY (gap = (LG-ESO)/ESO ; +ve = latgrad worse)")
        print("=" * 64)
        print(f"  configs compared         : {len(valid)}")
        print(f"  mean dp gap              : {valid['dp_gap_pct'].mean():+.1f}%")
        print(f"  median dp gap            : {valid['dp_gap_pct'].median():+.1f}%")
        print(f"  latgrad wins (lower dp)  : {(valid['winner']=='latgrad').sum()}/{len(valid)}")
        print(f"  mean volume diff (LG-ESO): {valid['vol_diff'].mean():+.4f}")
        print("  (vol diff > 0 => latgrad used more material — confound to disclose)")
        print("=" * 64)
        show = valid[["run", "tag", "eso_vol", "eso_dp", "lg_vol", "lg_dp",
                      "dp_gap_pct", "winner"]]
        with pd.option_context("display.max_rows", None, "display.width", 160,
                               "display.float_format", lambda v: f"{v:.4f}"):
            print("\n", show.to_string(index=False))
    return master


# ======================================================================
# (B) PLOTS
# ======================================================================
def _load_design(design_dir):
    if not os.path.isdir(design_dir):
        return None, None
    npys = sorted(glob.glob(os.path.join(design_dir, "*.npy")))
    if npys:
        try:
            return "npy", np.load(npys[0])
        except Exception:
            pass
    pngs = sorted(glob.glob(os.path.join(design_dir, "*.png")))
    if pngs:
        return "png", pngs[0]
    return None, None


def _blank_border(arr):
    """Zero the 4-cell wall frame so neither method shows a stray black wall.
    Border is not part of the design and does not affect dp/vol."""
    a = arr.copy().astype(float)
    W = WALL
    a[:W, :] = 0.0; a[-W:, :] = 0.0
    a[:, :W] = 0.0; a[:, -W:] = 0.0
    return a


def _draw_ports(ax, ports):
    for p in ports:
        r = p["range"]; wall = p["wall"]
        color = "green" if p["type"] == "inlet" else "red"
        if   wall == "left":   ax.plot([0, 0],          [r.start, r.stop], color=color, lw=4)
        elif wall == "right":  ax.plot([Nx-1, Nx-1],    [r.start, r.stop], color=color, lw=4)
        elif wall == "bottom": ax.plot([r.start, r.stop], [0, 0],          color=color, lw=4)
        elif wall == "top":    ax.plot([r.start, r.stop], [Ny-1, Ny-1],    color=color, lw=4)


def _show(ax, kind, data, title, ports):
    if kind == "npy":
        arr = _blank_border(data)
        ax.imshow(arr.T, cmap="gray_r", origin="lower",
                  vmin=0, vmax=1, interpolation="nearest")
        _draw_ports(ax, ports)
    elif kind == "png":
        ax.imshow(mpimg.imread(data))   # PNG already styled; ports baked in
    else:
        ax.text(0.5, 0.5, "no design", ha="center", va="center",
                color="crimson", transform=ax.transAxes)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _tag_for_cfg(cfg_dir):
    """Read the tag for this cfg from the run's CSV, matched by config index."""
    csv_path = os.path.join(os.path.dirname(cfg_dir), "sweep_results.csv")
    cfg_id = os.path.basename(cfg_dir)
    m = re.search(r"(\d+)", cfg_id)
    if not (m and os.path.exists(csv_path)):
        return None
    try:
        df = pd.read_csv(csv_path)
        if "tag" not in df.columns:
            return None
        idx = int(m.group(1))
        key = df["config_idx"] if "config_idx" in df.columns else (df.index + 1)
        row = df[key == idx]
        if len(row):
            return str(row.iloc[0]["tag"])
    except Exception:
        return None
    return None


def make_side_by_side(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cfg_dirs = sorted(glob.glob(os.path.join(root, "sweep_results", "best", "2026*", "cfg*")))
    if not cfg_dirs:
        print(f"[plots] no cfg dirs matched: {os.path.join(root,'sweep_results', 'best','2026*','cfg*')}")
        return

    print(f"[plots] found {len(cfg_dirs)} cfg dir(s).")
    made = 0
    for cfg_dir in cfg_dirs:
        run_id = os.path.basename(os.path.dirname(cfg_dir))
        cfg_id = os.path.basename(cfg_dir)

        tag = _tag_for_cfg(cfg_dir)
        ports = parse_ports_from_tag(tag)

        eso_kind, eso_data = _load_design(os.path.join(cfg_dir, "lbm_eso"))
        lg_kind,  lg_data  = _load_design(os.path.join(cfg_dir, "latent_grad"))
        if eso_kind is None and lg_kind is None:
            print(f"[plots] skip {run_id}/{cfg_id}: no designs in either subdir")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
        _show(axes[0], eso_kind, eso_data, "LBM ESO", ports)
        _show(axes[1], lg_kind,  lg_data,  "Latent Gradient", ports)
        ttl = f"{run_id} / {cfg_id}" + (f"  ({tag})" if tag else "")
        fig.suptitle(ttl, fontsize=11)

        # shared legend
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([0],[0],color="green",lw=4,label="inlet"),
                            Line2D([0],[0],color="red",lw=4,label="outlet")],
                   loc="lower center", ncol=2, fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(os.path.join(out_dir, f"{run_id}_{cfg_id}_compare.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        made += 1

    print(f"[plots] wrote {made} comparison image(s) -> {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="dir that CONTAINS sweep_results/")
    ap.add_argument("--out_csv", default="master_results.csv")
    ap.add_argument("--out_plots", default="comparison_plots")
    ap.add_argument("--only-table", action="store_true")
    ap.add_argument("--only-plots", action="store_true")
    args = ap.parse_args()

    print("ASSUMED LAYOUT:")
    print("  runs : sweep_results/best/2026*/")
    print("  csv  : <run>/sweep_results.csv")
    print("  cfgs : <run>/cfg*/{lbm_eso,latent_grad}/<design.npy or .png>")
    print(f"  PORT_HEIGHT={PORT_HEIGHT}, WALL={WALL}, grid={Nx}x{Ny}")
    print(f"  root : {os.path.abspath(args.root)}\n")

    if not args.only_plots:
        build_master_table(args.root, args.out_csv)
    if not args.only_table:
        print()
        make_side_by_side(args.root, args.out_plots)


if __name__ == "__main__":
    main()