

import argparse
import glob
import json
import os
import heapq
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Nx, Ny = 64, 64
WALL   = 4

METHOD_KEYS = {
    "lbm_eso":     "LBM ESO",
    "latent_grad": "Latent Gradient",
    "cmaes":       "CMA-ES",
}


def parse_port_args(raw_args):
    PORT_HEIGHT = int(0.10 * Ny)
    ports = []
    for p in raw_args:
        ptype, wall, center = p[0], p[1], int(p[2])
        lo = max(WALL, center - PORT_HEIGHT // 2)
        hi = min(Ny - WALL, center + PORT_HEIGHT // 2)
        ports.append({"type": ptype, "wall": wall,
                       "range": slice(lo, hi), "center": center})
    return ports


def ports_to_desc(ports):
    return "  |  ".join(f"{p['type']}@{p['wall']}:{p['center']}" for p in ports)


def get_port_cells(port):
    r = port["range"]
    wall = port["wall"]
    if wall == "left":   return [(WALL,        y) for y in range(r.start, r.stop)]
    if wall == "right":  return [(Nx-WALL-1,   y) for y in range(r.start, r.stop)]
    if wall == "bottom": return [(x, WALL)        for x in range(r.start, r.stop)]
    if wall == "top":    return [(x, Ny-WALL-1)   for x in range(r.start, r.stop)]


def get_port_center_coords(port):
    c = port["center"]
    wall = port["wall"]
    if wall == "left":   return (WALL,        c)
    if wall == "right":  return (Nx-WALL-1,   c)
    if wall == "bottom": return (c, WALL)
    if wall == "top":    return (c, Ny-WALL-1)


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_tortuosity(binary, ports):
    """
    For every inlet-outlet pair: tau = shortest fluid path / straight-line distance.
    Uses Dijkstra on the binary grid.
    """
    inlets  = [p for p in ports if p["type"] == "inlet"]
    outlets = [p for p in ports if p["type"] == "outlet"]
    results = []

    for inlet in inlets:
        for outlet in outlets:
            ix, iy = get_port_center_coords(inlet)
            ox, oy = get_port_center_coords(outlet)
            L_straight = np.sqrt((ix - ox)**2 + (iy - oy)**2)

            goal_set = set(map(tuple, get_port_cells(outlet)))
            dist = {}
            heap = []
            for cell in get_port_cells(inlet):
                cell = tuple(cell)
                dist[cell] = 0
                heapq.heappush(heap, (0, cell))

            L_actual = None
            while heap:
                d, node = heapq.heappop(heap)
                if node in goal_set:
                    L_actual = d
                    break
                if d > dist.get(node, float("inf")):
                    continue
                x, y = node
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nx_, ny_ = x+dx, y+dy
                    if 0 <= nx_ < Nx and 0 <= ny_ < Ny and binary[nx_, ny_]:
                        nd = d + 1
                        if nd < dist.get((nx_, ny_), float("inf")):
                            dist[(nx_, ny_)] = nd
                            heapq.heappush(heap, (nd, (nx_, ny_)))

            if L_actual is not None and L_straight > 0:
                results.append({
                    "inlet_wall":  inlet["wall"],
                    "outlet_wall": outlet["wall"],
                    "L_actual":    L_actual,
                    "L_straight":  L_straight,
                    "tortuosity":  L_actual / L_straight,
                })

    return results


def compute_channel_width(binary, ports, n_samples=20):
    """
    Sample perpendicular slices along the straight line between each
    inlet-outlet pair centre. Returns (mean_width, std_width) in pixels.
    """
    inlets  = [p for p in ports if p["type"] == "inlet"]
    outlets = [p for p in ports if p["type"] == "outlet"]
    widths  = []

    for inlet in inlets:
        for outlet in outlets:
            p0 = np.array(get_port_center_coords(inlet),  dtype=float)
            p1 = np.array(get_port_center_coords(outlet), dtype=float)

            flow_dir = p1 - p0
            flow_dir = flow_dir / (np.linalg.norm(flow_dir) + 1e-8)
            perp     = np.array([-flow_dir[1], flow_dir[0]])

            for t in np.linspace(0.1, 0.9, n_samples):
                px, py = (p0 + t * (p1 - p0)).astype(int)
                if not (0 <= px < Nx and 0 <= py < Ny):
                    continue
                if not binary[px, py]:
                    continue

                width = 1
                for sign in [1, -1]:
                    for step in range(1, 20):
                        qx = int(px + sign * step * perp[0])
                        qy = int(py + sign * step * perp[1])
                        if 0 <= qx < Nx and 0 <= qy < Ny and binary[qx, qy]:
                            width += 1
                        else:
                            break
                widths.append(width)

    if not widths:
        return None, None
    return float(np.mean(widths)), float(np.std(widths))


def compute_utilization(binary, ports):
    """
    Fraction of fluid material that lies on any inlet->outlet flow path.
    Computed as intersection of BFS forward from inlets and BFS backward from outlets.
    """
    total_fluid = int(binary.sum())
    if total_fluid == 0:
        return 0.0

    inlets  = [p for p in ports if p["type"] == "inlet"]
    outlets = [p for p in ports if p["type"] == "outlet"]

    def bfs(start_cells):
        visited = set()
        q = deque()
        for cell in start_cells:
            cell = tuple(cell)
            if binary[cell[0], cell[1]]:
                visited.add(cell)
                q.append(cell)
        while q:
            x, y = q.popleft()
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx_, ny_ = x+dx, y+dy
                if (0 <= nx_ < Nx and 0 <= ny_ < Ny
                        and binary[nx_, ny_]
                        and (nx_, ny_) not in visited):
                    visited.add((nx_, ny_))
                    q.append((nx_, ny_))
        return visited

    forward  = bfs([c for p in inlets  for c in get_port_cells(p)])
    backward = bfs([c for p in outlets for c in get_port_cells(p)])
    return len(forward & backward) / total_fluid


def run_metrics(density_np, ports, label):
    binary = (density_np > 0.5).astype(np.uint8)
    vol    = float(binary.mean())

    torts           = compute_tortuosity(binary, ports)
    mean_tau        = float(np.mean([t["tortuosity"] for t in torts])) if torts else None
    mean_w, std_w   = compute_channel_width(binary, ports)
    util            = compute_utilization(binary, ports)

    print(f"\n{'='*54}")
    print(f"  {label}")
    print(f"{'='*54}")
    print(f"  Volume fraction     : {vol:.4f}")
    if torts:
        for t in torts:
            print(f"  Tortuosity ({t['inlet_wall']:<6}→{t['outlet_wall']:<6}): "
                  f"{t['tortuosity']:.4f}  "
                  f"(path={t['L_actual']}px  straight={t['L_straight']:.1f}px)")
        print(f"  Mean tortuosity     : {mean_tau:.4f}")
    if mean_w is not None:
        print(f"  Mean channel width  : {mean_w:.2f} px")
        print(f"  Width std dev       : {std_w:.2f} px")
    print(f"  Material utilisation: {util:.4f}")

    return {
        "volume":           vol,
        "tortuosity_pairs": torts,
        "mean_tortuosity":  mean_tau,
        "mean_width":       mean_w,
        "width_std":        std_w,
        "utilization":      util,
    }



def _draw_ports(ax, ports):
    in_colors  = ["#00cc44", "#88ee00"]
    out_colors = ["#dd2222", "#ff7700"]
    ii = oi = 0
    for p in ports:
        r, wall = p["range"], p["wall"]
        if p["type"] == "inlet":
            color = in_colors[ii % len(in_colors)];   ii += 1
        else:
            color = out_colors[oi % len(out_colors)]; oi += 1
        lw = 4
        if wall == "left":    ax.plot([0, 0],         [r.start, r.stop], color=color, lw=lw)
        elif wall == "right": ax.plot([Nx-1, Nx-1],   [r.start, r.stop], color=color, lw=lw)
        elif wall == "bottom":ax.plot([r.start, r.stop], [0, 0],          color=color, lw=lw)
        elif wall == "top":   ax.plot([r.start, r.stop], [Ny-1, Ny-1],    color=color, lw=lw)


def save_plot(designs, metrics, ports, save_dir, tag):
    keys   = list(designs.keys())
    labels = [METHOD_KEYS.get(k, k) for k in keys]
    n      = len(keys)

    fig = plt.figure(figsize=(5*n, 9))
    gs  = fig.add_gridspec(2, n, height_ratios=[3, 2], hspace=0.4, wspace=0.3)

    # --- topology row ---
    for col, key in enumerate(keys):
        ax  = fig.add_subplot(gs[0, col])
        arr = (designs[key] > 0.5).astype(float)
        ax.imshow(arr.T, cmap="gray_r", origin="lower", vmin=0, vmax=1)
        _draw_ports(ax, ports)
        m   = metrics[key]
        tau = f"{m['mean_tortuosity']:.3f}" if m["mean_tortuosity"] else "N/A"
        ax.set_title(f"{labels[col]}\nvol={m['volume']:.3f}  τ={tau}", fontsize=9)
        ax.axis("off")

    # --- metric bar row ---
    ax_bar = fig.add_subplot(gs[1, :])
    metric_specs = [
        ("mean_tortuosity", "Mean tortuosity",      "steelblue"),
        ("utilization",     "Material utilisation", "seagreen"),
        ("mean_width",      "Channel width (px)",   "darkorange"),
    ]
    x       = np.arange(n)
    n_met   = len(metric_specs)
    bar_w   = 0.22
    offsets = np.linspace(-(n_met-1)/2, (n_met-1)/2, n_met) * bar_w

    for i, (mkey, mlabel, mcolor) in enumerate(metric_specs):
        vals = [metrics[k].get(mkey) or 0.0 for k in keys]
        bars = ax_bar.bar(x + offsets[i], vals, bar_w,
                          label=mlabel, color=mcolor, alpha=0.82)
        for bar, v in zip(bars, vals):
            if v:
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_ylabel("Value")
    ax_bar.set_title("Metric comparison across methods")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor="#00cc44", label="inlet"),
            Patch(facecolor="#dd2222", label="outlet"),
            Patch(facecolor="#333",    label="fluid"),
            Patch(facecolor="white", edgecolor="gray", label="void"),
        ],
        loc="lower center", ncol=4, fontsize=8,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.01),
    )
    plt.suptitle(f"Comparison  —  {ports_to_desc(ports)}", fontsize=9)

    out = os.path.join(save_dir, f"analysis_{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analyse] Plot saved → {out}")


# ── summary table ─────────────────────────────────────────────────────────────

def print_table(metrics):
    cw = 20
    print(f"\n{'='*78}")
    print("  SUMMARY")
    print(f"{'='*78}")
    print(f"  {'Method':<{cw}} {'Volume':>8} {'Tortuosity':>12} "
          f"{'Width(px)':>11} {'Width σ':>9} {'Utilisation':>13}")
    print(f"  {'-'*cw} {'-'*8} {'-'*12} {'-'*11} {'-'*9} {'-'*13}")
    for key, label in METHOD_KEYS.items():
        if key not in metrics:
            continue
        m = metrics[key]
        tau = f"{m['mean_tortuosity']:.4f}" if m["mean_tortuosity"] is not None else "N/A"
        mw  = f"{m['mean_width']:.2f}"      if m["mean_width"]      is not None else "N/A"
        ws  = f"{m['width_std']:.2f}"       if m["width_std"]        is not None else "N/A"
        print(f"  {label:<{cw}} {m['volume']:>8.4f} {tau:>12} {mw:>11} {ws:>9} {m['utilization']:>13.4f}")
    print(f"{'='*78}")

def ports_to_tag(ports):
    return "_".join(f"{p['type'][0]}{p['wall'][0]}{p['center']}" for p in ports)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--port", action="append", nargs=3,
        metavar=("TYPE", "WALL", "CENTER"), required=True,
        help="Same format as comparison.py. Repeat for multiple ports.",
    )
    parser.add_argument(
        "--results_dir", default="comparison_results",
        help="Folder containing comparison_*.npz files (default: comparison_results)",
    )
    args = parser.parse_args()

    ports    = parse_port_args(args.port)
    save_dir = args.results_dir

    # ------------------------------------------------------------
    # Build expected filename from ports (same logic as comparison.py)
    # ------------------------------------------------------------
    tag = ports_to_tag(ports)
    npz_path = os.path.join(args.results_dir, f"comparison_{tag}.npz")

    if not os.path.exists(npz_path):
        print(f"[analyse] Expected file not found:")
        print(f"           {npz_path}")
        print("\nDid you run comparison.py with the same ports?")
        return

    print(f"[analyse] Loading file:")
    print(f"           {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    designs = {}
    for key in METHOD_KEYS:
        npz_key = f"final_np_{key}"
        if npz_key in data:
            designs[key] = data[npz_key]
            print(f"[analyse] Loaded {METHOD_KEYS[key]}")
            
    if not designs:
        print("[analyse] No final_np_* arrays found — nothing to analyse.")
        return

    print(f"\n[analyse] Ports  : {ports_to_desc(ports)}")
    print(f"[analyse] Methods: {[METHOD_KEYS[k] for k in designs]}")

    # run metrics on each design
    metrics = {}
    for key, arr in designs.items():
        metrics[key] = run_metrics(arr, ports, label=METHOD_KEYS[key])

    print_table(metrics)

    # tag from first file name for the output plot name
    tag = ports_to_tag(ports)
    save_plot(designs, metrics, ports, save_dir, tag)

    print("\n[analyse] Done.")


if __name__ == "__main__":
    main()