
import argparse
import json
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Nx, Ny = 64, 64
WALL   = 4


def ports_to_tag(ports):
    return "_".join(f"{p['type'][0]}{p['wall'][0]}{p['center']}" for p in ports)


def ports_to_desc(ports):
    return "  |  ".join(f"{p['type']}@{p['wall']}:{p['center']}" for p in ports)


def parse_port_args(raw_args):
    """Convert [['inlet','top','16'], ...] → list of port dicts."""
    PORT_HEIGHT = int(0.10 * Ny)
    ports = []
    for p in raw_args:
        ptype, wall, center = p[0], p[1], int(p[2])
        lo = max(WALL, center - PORT_HEIGHT // 2)
        hi = min(Ny - WALL, center + PORT_HEIGHT // 2)
        ports.append({
            "type":   ptype,
            "wall":   wall,
            "range":  slice(lo, hi),
            "center": center,
        })
    return ports


_LBM_CONFIGS = {
    # config_id : (sorted inlet walls, sorted outlet walls)
    0: (["bottom", "left"],  ["top"]),
    1: (["right",  "right"], ["top"]),
    2: (["top",    "top"],   ["left"]),
}


def detect_lbm_config(ports):
    inlet_walls  = sorted(p["wall"] for p in ports if p["type"] == "inlet")
    outlet_walls = sorted(p["wall"] for p in ports if p["type"] == "outlet")
    for cfg_id, (iw, ow) in _LBM_CONFIGS.items():
        if inlet_walls == sorted(iw) and outlet_walls == sorted(ow):
            return cfg_id
    return None          # no exact match


# ── subprocess runners ────────────────────────────────────────────────────────

def run_lbm(ports, target_volume, lbm_config, output_dir):

    print("\n[comparison] Ports passed to LBM:")
    for p in ports:
        print(f"   {p['type']:6s} @ {p['wall']:6s}  center={p['center']}")

    PORT_HEIGHT = 4   # must match lbm_multiple.py

    cmd = [
        sys.executable, "lbm_multiple.py",
        "--target_volume", str(target_volume),
    ]

    # Add each port exactly like user CLI
    for p in ports:
        cmd += [
            "--port",
            p["type"],
            p["wall"],
            str(p["center"]),
            str(PORT_HEIGHT),
        ]

    print("\n[comparison] LBM command:")
    print(" ".join(cmd))
    _run_subprocess("LBM ESO", cmd)
 
    lbm_out = "results/best_topology.npy"
    if not os.path.exists(lbm_out):
        print(f"[comparison] LBM ESO: expected output not found at {lbm_out}")
        return None
 
    arr = np.load(lbm_out).copy()
 

    W = WALL  # = 4
 
    # zero the full border frame
    arr[:W, :]   = 0.0
    arr[-W:, :]  = 0.0
    arr[:, :W]   = 0.0
    arr[:, -W:]  = 0.0
 
    # re-open port slots — those border cells are fluid (orifice), not solid
    for p in ports:
        r    = p["range"]
        wall = p["wall"]
        if wall == "left":    arr[:W,  r] = 1.0
        elif wall == "right": arr[-W:, r] = 1.0
        elif wall == "bottom":arr[r,  :W] = 1.0
        elif wall == "top":   arr[r,  -W:] = 1.0
    file_name = f"lbm_final_np_{ports_to_tag(ports)}.npy"
    dst = os.path.join(output_dir, file_name)
    np.save(dst, arr)
    vol = float((arr > 0.5).mean())
    print(f"[comparison] LBM ESO result saved → {dst}  shape={arr.shape}  vol={vol:.3f}")
    return arr


def run_gradient_opt(ports, vae_path, n_restarts, n_steps, lr,
                     lambda_volume, lambda_binary, temp_start, temp_end,
                     output_dir):
    cmd = [sys.executable, "gradient_opt_1.py"]
    for p in ports:
        cmd += ["--port", p["type"], p["wall"], str(p["center"])]
    cmd += [
        "--vae_path",      vae_path,
        "--n_restarts",    str(n_restarts),
        "--n_steps",       str(n_steps),
        "--lr",            str(lr),
        "--lambda_volume", str(lambda_volume),
        "--lambda_binary", str(lambda_binary),
        "--temp_start",    str(temp_start),
        "--temp_end",      str(temp_end),
    ]

    _run_subprocess("Latent Gradient", cmd)

    tag     = ports_to_tag(ports) + f"_lv{lambda_volume}"
    src     = f"latgrad_result_{tag}.npz"
    if not os.path.exists(src):
        print(f"[comparison] Latent Gradient: expected output not found at {src}")
        return None

    data = np.load(src, allow_pickle=True)
    arr  = data["best_design"]
    file_name = f"latgrad_final_np_{tag}.npy"
    dst  = os.path.join(output_dir, file_name)
    np.save(dst, arr)
    print(f"[comparison] Latent Gradient result saved → {dst}  shape={arr.shape}")
    return arr


def run_cmaes(ports, vae_path, lambda_volume, max_gen, popsize, sigma0,
              output_dir):
    cmd = [sys.executable, "cmaes_mit_data_multiple.py"]
    for p in ports:
        cmd += ["--port", p["type"], p["wall"], str(p["center"])]
    cmd += [
        "--vae_path",      vae_path,
        "--lambda_volume", str(lambda_volume),
        "--max_gen",       str(max_gen),
        "--popsize",       str(popsize),
        "--sigma0",        str(sigma0),
    ]

    _run_subprocess("CMA-ES", cmd)

    tag  = ports_to_tag(ports) + f"_lv{lambda_volume}"
    src  = f"cmaes_result_{tag}.npz"
    if not os.path.exists(src):
        print(f"[comparison] CMA-ES: expected output not found at {src}")
        return None

    data = np.load(src, allow_pickle=True)
    arr  = data["best_design"]
    file_name = f"cmaes_final_np_{tag}.npy"
    dst  = os.path.join(output_dir, file_name)
    np.save(dst, arr)
    print(f"[comparison] CMA-ES result saved → {dst}  shape={arr.shape}")
    return arr


def _run_subprocess(label, cmd):
    print(f"\n{'='*70}")
    print(f"[comparison] Starting: {label}")
    print(f"             CMD: {' '.join(cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[comparison] WARNING: {label} exited with code {result.returncode}")



def _draw_ports(ax, ports):
    """Overlay port markers on an existing imshow axes."""
    in_idx = out_idx = 0
    inlet_colors  = ["#00cc44", "#88ee00"]
    outlet_colors = ["#dd2222", "#ff7700"]

    for p in ports:
        r    = p["range"]
        wall = p["wall"]
        if p["type"] == "inlet":
            color = inlet_colors[in_idx % len(inlet_colors)]; in_idx += 1
        else:
            color = outlet_colors[out_idx % len(outlet_colors)]; out_idx += 1

        lw = 4
        if wall == "left":
            ax.plot([0, 0], [r.start, r.stop], color=color, linewidth=lw)
        elif wall == "right":
            ax.plot([Nx-1, Nx-1], [r.start, r.stop], color=color, linewidth=lw)
        elif wall == "bottom":
            ax.plot([r.start, r.stop], [0, 0], color=color, linewidth=lw)
        elif wall == "top":
            ax.plot([r.start, r.stop], [Ny-1, Ny-1], color=color, linewidth=lw)

from new_generate_dataset_multiple import build_bc_masks
def compute_vol(arr, ports):
    masks = build_bc_masks(Nx, Ny, WALL, ports)
    is_designable = (
        ~masks["solid_mask"] &
        ~masks["fluid_mask"] &
        ~masks["orifice_mask"]
    ).cpu().numpy()
    return float((arr > 0.5)[is_designable].mean())

def save_comparison_plot(results, ports, output_dir, tag):
    valid = {name: arr for name, arr in results.items() if arr is not None}
    if not valid:
        print("[comparison] No valid results to plot.")
        return

    labels = {
        "lbm_eso":     "LBM ESO",
        "latent_grad": "Latent Gradient",
        "cmaes":       "CMA-ES",
    }

    n   = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5), squeeze=False)
    axes = axes[0]

    for ax, (name, arr) in zip(axes, valid.items()):
        binary = (arr > 0.5).astype(float)
        ax.imshow(binary.T, cmap="gray_r", origin="lower", vmin=0, vmax=1)
        _draw_ports(ax, ports)
        vol = binary.mean()
        vol = compute_vol(binary, ports)
        ax.set_title(f"{labels.get(name, name)}\nvol={vol:.3f}", fontsize=10)
        ax.axis("off")

    # shared legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#00cc44", label="inlet"),
        Patch(facecolor="#dd2222", label="outlet"),
        Patch(facecolor="#333333", label="fluid material"),
        Patch(facecolor="white",   edgecolor="gray", label="void"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(f"Optimizer Comparison\n{ports_to_desc(ports)}", fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = os.path.join(output_dir, f"comparison_{tag}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[comparison] Side-by-side plot saved → {out_path}")



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run LBM ESO, Latent Gradient Opt, and CMA-ES with one shared "
            "port config and collect all final designs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Port specification (shared by all three methods) ──────────────────
    parser.add_argument(
        "--port", action="append", nargs=3,
        metavar=("TYPE", "WALL", "CENTER"), required=True,
        help=(
            "Add a port. TYPE=inlet|outlet, WALL=left|right|top|bottom, "
            "CENTER=integer.  Repeat for multiple ports."
        ),
    )

    # ── Shared ────────────────────────────────────────────────────────────
    parser.add_argument("--vae_path",   default="vae_best_new.pth")
    parser.add_argument("--output_dir", default="comparison_results",
                        help="Where to save final_np arrays and comparison plot.")

    # ── LBM ESO ───────────────────────────────────────────────────────────
    lbm = parser.add_argument_group("LBM ESO (lbm_multiport.py)")
    lbm.add_argument("--target_volume", type=float, default=0.20)
    lbm.add_argument(
        "--lbm_config", type=int, default=None, choices=[0, 1, 2],
        help=(
            "Force lbm_multiport.py config index (0/1/2). "
            "Auto-detected from port walls when omitted."
        ),
    )
    lbm.add_argument("--skip_lbm",  action="store_true",
                     help="Skip the LBM ESO run entirely.")

    # ── Latent Gradient ───────────────────────────────────────────────────
    grad = parser.add_argument_group("Latent Gradient (gradient_opt_2.py)")
    grad.add_argument("--n_restarts",         type=int,   default=1)
    grad.add_argument("--n_steps",            type=int,   default=200)
    grad.add_argument("--lr",                 type=float, default=0.05)
    grad.add_argument("--lambda_volume_grad", type=float, default=0.5,
                      help="lambda_volume passed to gradient_opt.py")
    grad.add_argument("--lambda_binary",      type=float, default=2.0)
    grad.add_argument("--temp_start",         type=float, default=1.0)
    grad.add_argument("--temp_end",           type=float, default=0.1)
    grad.add_argument("--skip_grad",  action="store_true",
                      help="Skip the Latent Gradient run entirely.")

    # ── CMA-ES ────────────────────────────────────────────────────────────
    cma = parser.add_argument_group("CMA-ES (cmaes_mit_data_multiple.py)")
    cma.add_argument("--lambda_volume_cmaes", type=float, default=0.4,
                     help="lambda_volume passed to cmaes_mit_data_multiple.py")
    cma.add_argument("--max_gen",  type=int,   default=50)
    cma.add_argument("--popsize",  type=int,   default=24)
    cma.add_argument("--sigma0",   type=float, default=0.5)
    cma.add_argument("--skip_cmaes", action="store_true",
                     help="Skip the CMA-ES run entirely.")

    args = parser.parse_args()

    # ── Parse ports ───────────────────────────────────────────────────────
    ports = parse_port_args(args.port)
    tag   = ports_to_tag(ports)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"[comparison] Ports : {ports_to_desc(ports)}")
    print(f"[comparison] Tag   : {tag}")
    print(f"[comparison] Output: {args.output_dir}")
    print("=" * 70)

    # ── Run each method ───────────────────────────────────────────────────
    results = {}

    if not args.skip_lbm:
        results["lbm_eso"] = run_lbm(
            ports, args.target_volume, args.lbm_config, args.output_dir
        )

    if not args.skip_grad:
        results["latent_grad"] = run_gradient_opt(
            ports, args.vae_path,
            args.n_restarts, args.n_steps, args.lr,
            args.lambda_volume_grad, args.lambda_binary,
            args.temp_start, args.temp_end,
            args.output_dir,
        )

    if not args.skip_cmaes:
        results["cmaes"] = run_cmaes(
            ports, args.vae_path,
            args.lambda_volume_cmaes, args.max_gen,
            args.popsize, args.sigma0,
            args.output_dir,
        )

    # ── Save combined npz ─────────────────────────────────────────────────
    save_dict = {
        "ports_json": json.dumps([
            {"type": p["type"], "wall": p["wall"], "center": p["center"]}
            for p in ports
        ])
    }
    for name, arr in results.items():
        if arr is not None:
            save_dict[f"final_np_{name}"] = arr

    combined_path = os.path.join(args.output_dir, f"comparison_{tag}.npz")
    np.savez(combined_path, **save_dict)
    print(f"\n[comparison] Combined npz saved → {combined_path}")
    print(f"             Keys: {[k for k in save_dict if k != 'ports_json']}")

    # ── Side-by-side plot ─────────────────────────────────────────────────
    save_comparison_plot(results, ports, args.output_dir, tag)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("[comparison] SUMMARY")
    print(f"{'='*70}")
    col_w = 22
    print(f"  {'Method':<{col_w}} {'Status':<12} {'Volume':>8}")
    print(f"  {'-'*col_w} {'-'*12} {'-'*8}")
    for name, arr in results.items():
        if arr is None:
            print(f"  {name:<{col_w}} {'FAILED':12} {'—':>8}")
        else:
            vol = float((arr > 0.5).mean())
            vol = compute_vol(arr, ports)
            print(f"  {name:<{col_w}} {'OK':12} {vol:>8.3f}")

    print(f"\n  Output dir : {args.output_dir}/")
    print(f"  Combined   : comparison_{tag}.npz")
    print(f"  Plot       : comparison_{tag}.png")


if __name__ == "__main__":
    main()