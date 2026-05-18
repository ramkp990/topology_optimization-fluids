"""
skeleton_angles.py

Loads designs from a comparison npz, skeletonizes each topology,
finds the junction (branch point), walks each arm to the junction,
computes angles between arms, and saves an annotated plot.

Usage
-----
python skeleton_angles.py \
    --port inlet left   14 \
    --port inlet bottom 52 \
    --port outlet right 37 \
    --results_dir comparison_results
"""

import argparse
import os
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from skimage.morphology import skeletonize

# ── constants ─────────────────────────────────────────────────────────────────
Nx, Ny = 64, 64
WALL   = 4

METHOD_KEYS = {
    "lbm_eso":     "LBM ESO",
    "latent_grad": "Latent Gradient",
    "cmaes":       "CMA-ES",
}


# ── port helpers ──────────────────────────────────────────────────────────────

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


def ports_to_tag(ports):
    return "_".join(f"{p['type'][0]}{p['wall'][0]}{p['center']}" for p in ports)


def ports_to_desc(ports):
    return "  |  ".join(f"{p['type']}@{p['wall']}:{p['center']}" for p in ports)


def get_port_cells(port):
    r = port["range"]
    wall = port["wall"]
    if wall == "left":   return [(WALL,        y) for y in range(r.start, r.stop)]
    if wall == "right":  return [(Nx-WALL-1,   y) for y in range(r.start, r.stop)]
    if wall == "bottom": return [(x, WALL)        for x in range(r.start, r.stop)]
    if wall == "top":    return [(x, Ny-WALL-1)   for x in range(r.start, r.stop)]


# ── skeleton helpers ──────────────────────────────────────────────────────────

def skel_neighbours(skel, x, y):
    """8-connected skeleton neighbours of (x,y)."""
    nb = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx_, ny_ = x+dx, y+dy
            if 0 <= nx_ < Nx and 0 <= ny_ < Ny and skel[nx_, ny_]:
                nb.append((nx_, ny_))
    return nb


def find_branch_points(skel):
    """Pixels with 3+ skeleton neighbours = junction."""
    return [(x, y) for x, y in zip(*np.where(skel))
            if len(skel_neighbours(skel, x, y)) >= 3]


def nearest_skel_point(skel, cells):
    """Closest skeleton pixel to any cell in the list."""
    skel_pts = list(zip(*np.where(skel)))
    if not skel_pts:
        return None
    best, best_d = None, float("inf")
    for cx, cy in cells:
        for sx, sy in skel_pts:
            d = (cx-sx)**2 + (cy-sy)**2
            if d < best_d:
                best_d = d
                best = (sx, sy)
    return best


def walk_path(skel, start, goal):
    """
    BFS along skeleton pixels from start to goal.
    Returns ordered list of (x,y) or None if unreachable.
    """
    if start == goal:
        return [start]
    visited = {start}
    queue   = deque([(start, [start])])
    while queue:
        (x, y), path = queue.popleft()
        for nx_, ny_ in skel_neighbours(skel, x, y):
            if (nx_, ny_) == goal:
                return path + [(nx_, ny_)]
            if (nx_, ny_) not in visited:
                visited.add((nx_, ny_))
                queue.append(((nx_, ny_), path + [(nx_, ny_)]))
    return None


def smooth_path(path, window=5):
    """Moving-average smooth on (x,y) path to reduce pixelation noise."""
    if len(path) < window:
        return path
    xs = np.array([p[0] for p in path], dtype=float)
    ys = np.array([p[1] for p in path], dtype=float)
    k  = np.ones(window) / window
    xs = np.convolve(xs, k, mode="valid")
    ys = np.convolve(ys, k, mode="valid")
    return list(zip(xs, ys))


def arm_direction(path, near_junction=True, n=8):
    """
    Direction vector of an arm near the junction end (near_junction=True)
    or port end.  Returns unit vector pointing AWAY from junction.
    """
    if len(path) < 2:
        return None
    seg = path[:min(n, len(path))] if near_junction else path[-min(n, len(path)):]
    dx  = seg[-1][0] - seg[0][0]
    dy  = seg[-1][1] - seg[0][1]
    norm = np.hypot(dx, dy)
    if norm < 1e-8:
        return None
    # near_junction=True: path runs port→junction, so seg[0] is near junction,
    # seg[-1] is further out.  Direction pointing away from junction = seg[-1]-seg[0].
    return np.array([dx / norm, dy / norm])


def angle_deg(v1, v2):
    """Angle in degrees between two unit vectors."""
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))))


# ── per-design analysis ───────────────────────────────────────────────────────

def analyse_design(binary, ports, label):
    """
    Skeletonize, find junction, walk arms, compute junction angles.
    Returns a result dict.
    """
    skel = skeletonize(binary.astype(bool))

    branch_pts = find_branch_points(skel)
    print(f"\n  [{label}]  branch points found: {len(branch_pts)}")

    # pick junction = branch point with most neighbours (most central)
    junction = None
    if branch_pts:
        junction = max(branch_pts,
                       key=lambda p: len(skel_neighbours(skel, p[0], p[1])))
        print(f"  [{label}]  junction at {junction}")

    # nearest skeleton pixel to each port
    port_anchors = {}
    for p in ports:
        cells  = get_port_cells(p)
        anchor = nearest_skel_point(skel, cells)
        key    = f"{p['type']}@{p['wall']}:{p['center']}"
        port_anchors[key] = anchor
        print(f"  [{label}]  {key} anchor → {anchor}")

    arms = []
    if junction is not None:
        for p in ports:
            key    = f"{p['type']}@{p['wall']}:{p['center']}"
            anchor = port_anchors[key]
            if anchor is None:
                continue
            raw_path = walk_path(skel, anchor, junction)
            if raw_path is None:
                print(f"  [{label}]  WARNING: no skeleton path from {key} to junction")
                continue
            # smooth then get direction near junction (first few pixels after junction)
            smoothed = smooth_path(raw_path, window=5)
            # direction points away from junction toward port
            vec = arm_direction(smoothed, near_junction=True, n=10)
            arms.append({"port": p, "key": key,
                          "raw_path": raw_path, "smoothed": smoothed,
                          "direction": vec})

    # compute pairwise angles at junction
    junction_angles = []
    for i in range(len(arms)):
        for j in range(i+1, len(arms)):
            v1 = arms[i]["direction"]
            v2 = arms[j]["direction"]
            if v1 is None or v2 is None:
                continue
            ang = angle_deg(v1, v2)
            # turn angle = how much flow has to turn = 180 - junction_angle
            turn = 180.0 - ang
            junction_angles.append({
                "arm1":        arms[i]["key"],
                "arm2":        arms[j]["key"],
                "junction_angle": ang,
                "turn_angle":  turn,
            })
            print(f"  [{label}]  {arms[i]['key']}  ↔  {arms[j]['key']}: "
                  f"junction angle={ang:.1f}°  turn={turn:.1f}°")

    return {
        "label":           label,
        "skeleton":        skel,
        "branch_points":   branch_pts,
        "junction":        junction,
        "port_anchors":    port_anchors,
        "arms":            arms,
        "junction_angles": junction_angles,
    }


# ── plot ──────────────────────────────────────────────────────────────────────

def _draw_ports(ax, ports):
    in_colors  = ["#00cc44", "#88ee00"]
    out_colors = ["#dd2222", "#ff7700"]
    ii = oi = 0
    for p in ports:
        r, wall = p["range"], p["wall"]
        color = in_colors[ii%2] if p["type"] == "inlet" else out_colors[oi%2]
        if p["type"] == "inlet": ii += 1
        else: oi += 1
        lw = 4
        if wall == "left":    ax.plot([0,0],         [r.start,r.stop], color=color, lw=lw)
        elif wall == "right": ax.plot([Nx-1,Nx-1],   [r.start,r.stop], color=color, lw=lw)
        elif wall == "bottom":ax.plot([r.start,r.stop],[0,0],           color=color, lw=lw)
        elif wall == "top":   ax.plot([r.start,r.stop],[Ny-1,Ny-1],     color=color, lw=lw)


def save_plot(designs, analyses, ports, save_dir, tag):
    keys = [k for k in METHOD_KEYS if k in designs]
    n    = len(keys)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5*n, 6), squeeze=False)
    axes = axes[0]

    for ax, key in zip(axes, keys):
        arr    = designs[key]
        binary = (arr > 0.5).astype(float)
        res    = analyses[key]
        skel   = res["skeleton"]

        # topology
        ax.imshow(binary.T, cmap="gray_r", origin="lower", vmin=0, vmax=1)

        # skeleton overlay
        sy, sx = np.where(skel.T)
        ax.scatter(sx, sy, c="red", s=3, zorder=3)

        # arm paths
        colors_arm = ["#ff9900", "#00ccff", "#cc00ff"]
        for ci, arm in enumerate(res["arms"]):
            rp = arm["raw_path"]
            if rp:
                px = [p[0] for p in rp]
                py = [p[1] for p in rp]
                ax.plot(px, py, color=colors_arm[ci % len(colors_arm)],
                        lw=1.5, zorder=4, alpha=0.7)

        # junction marker
        if res["junction"]:
            jx, jy = res["junction"]
            ax.scatter([jx], [jy], c="yellow", s=150,
                       marker="*", zorder=6, edgecolors="black", linewidths=0.5)

        # port markers
        _draw_ports(ax, ports)

        # angle annotations
        angle_lines = []
        for ang in res["junction_angles"]:
            a1_short = ang["arm1"].split("@")[1].split(":")[0]  # e.g. "left"
            a2_short = ang["arm2"].split("@")[1].split(":")[0]
            angle_lines.append(
                f"{a1_short}↔{a2_short}\n"
                f"  junction={ang['junction_angle']:.1f}°\n"
                f"  turn    ={ang['turn_angle']:.1f}°"
            )

        title = f"{METHOD_KEYS[key]}\n" + "\n".join(angle_lines)
        ax.set_title(title, fontsize=7, family="monospace")
        ax.axis("off")

    # legend
    legend_handles = [
        mpatches.Patch(facecolor="#00cc44", label="inlet"),
        mpatches.Patch(facecolor="#dd2222", label="outlet"),
        Line2D([0],[0], color="red",    lw=1.5, label="skeleton"),
        Line2D([0],[0], marker="*",     color="yellow", lw=0,
               markersize=10, markeredgecolor="black", label="junction"),
        Line2D([0],[0], color="#ff9900",lw=1.5, label="arm paths"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(f"Skeleton Analysis — Junction Angles\n{ports_to_desc(ports)}",
                 fontsize=9, y=1.01)
    plt.tight_layout()

    out = os.path.join(save_dir, f"skeleton_{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[skeleton] Plot saved → {out}")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(analyses):
    print(f"\n{'='*72}")
    print("  JUNCTION ANGLE SUMMARY")
    print(f"  (junction angle: higher = smoother flow path)")
    print(f"  (turn angle    : lower  = less aggressive bend)")
    print(f"{'='*72}")
    for key, label in METHOD_KEYS.items():
        if key not in analyses:
            continue
        res = analyses[key]
        print(f"\n  {label}")
        if not res["junction_angles"]:
            print("    no junction found (single-path topology)")
            continue
        for ang in res["junction_angles"]:
            better = "✓ smooth" if ang["turn_angle"] < 45 else \
                     "~ moderate" if ang["turn_angle"] < 80 else "✗ sharp"
            print(f"    {ang['arm1']}")
            print(f"    {ang['arm2']}")
            print(f"      junction angle : {ang['junction_angle']:6.1f}°")
            print(f"      turn angle     : {ang['turn_angle']:6.1f}°  {better}")
    print(f"\n{'='*72}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", action="append", nargs=3,
                        metavar=("TYPE", "WALL", "CENTER"), required=True)
    parser.add_argument("--results_dir", default="comparison_results")
    args = parser.parse_args()

    ports = parse_port_args(args.port)
    tag   = ports_to_tag(ports)

    # load npz
    npz_path = os.path.join(args.results_dir, f"comparison_{tag}.npz")
    if not os.path.exists(npz_path):
        print(f"[skeleton] File not found: {npz_path}")
        return

    print(f"[skeleton] Loading {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    designs = {}
    for key in METHOD_KEYS:
        k = f"final_np_{key}"
        if k in data:
            designs[key] = data[k]
            print(f"[skeleton] Loaded {METHOD_KEYS[key]}")

    if not designs:
        print("[skeleton] No designs found in npz.")
        return

    print(f"\n[skeleton] Ports: {ports_to_desc(ports)}")

    # run analysis
    analyses = {}
    for key, arr in designs.items():
        binary = (arr > 0.5).astype(np.uint8)
        analyses[key] = analyse_design(binary, ports, label=METHOD_KEYS[key])

    print_summary(analyses)
    save_plot(designs, analyses, ports, args.results_dir, tag)
    print("\n[skeleton] Done.")


if __name__ == "__main__":
    main()