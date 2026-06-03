
import argparse, os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vae_fluid_multiple import FluidVAE, make_bc_mask, port_cells
from new_generate_dataset_multiple import build_bc_masks

LATENT_DIM = 32
THRESHOLD  = 0.5
Nx, Ny     = 64, 64
WALL       = 4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
WALL_THICKNESS = 4
PORT_HEIGHT    = int(0.10 * Ny)   # 6
WALLS          = ["left", "right", "top", "bottom"]


# ═══════════════════════════════════════════════════════════════════════════════
# PORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def make_slot(center, height=PORT_HEIGHT):
    return slice(int(center - height // 2), int(center + height // 2))

def random_center(wall):
    margin   = WALL_THICKNESS + PORT_HEIGHT // 2 + 2
    axis_max = Ny if wall in ("left", "right") else Nx
    return int(np.random.randint(margin, axis_max - margin))

def sample_ports(n, port_type):
    ports, attempts = [], 0
    while len(ports) < n and attempts < 50:
        wall   = np.random.choice(WALLS)
        center = random_center(wall)
        ok = all(
            abs(center - (p["range"].start + p["range"].stop) // 2) >= PORT_HEIGHT + 5
            for p in ports if p["wall"] == wall
        )
        if ok:
            ports.append({"type": port_type, "wall": wall,
                          "range": make_slot(center), "center": center})
        attempts += 1
    return ports

def ports_overlap(a, b, gap=5):
    if a["wall"] != b["wall"]:
        return False
    return (a["range"].start - gap) < b["range"].stop and \
           b["range"].start < (a["range"].stop + gap)

def generate_valid_ports(n_inlets=1, seed=None, max_attempts=200):
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        np.random.seed(int(rng.integers(0, 2**31)))
        ports = sample_ports(n_inlets, "inlet") + sample_ports(1, "outlet")

        overlap   = any(ports_overlap(ports[i], ports[j])
                        for i in range(len(ports))
                        for j in range(i + 1, len(ports)))
        same_wall = bool({p["wall"] for p in ports if p["type"] == "outlet"} &
                         {p["wall"] for p in ports if p["type"] == "inlet"})

        if not overlap and not same_wall:
            return ports
    raise RuntimeError(f"Could not sample valid ports in {max_attempts} attempts.")

def ports_to_tag(ports):
    return "_".join(f"{p['type'][0]}{p['wall'][0]}{p['center']}" for p in ports)

def ports_to_desc(ports):
    return " | ".join(f"{p['type']}@{p['wall']}:{p['center']}" for p in ports)


# ---------- feasibility (connectivity + volume), structural only ----------
from collections import deque
def _bfs(binary, starts):
    seen = set(map(tuple, starts)); q = deque(starts)
    while q:
        x, y = q.popleft()
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
            a, b = x+dx, y+dy
            if 0 <= a < Nx and 0 <= b < Ny and binary[a, b] and (a, b) not in seen:
                seen.add((a, b)); q.append((a, b))
    return seen

def connected(design, ports):
    b = (design > THRESHOLD).astype(np.uint8)
    outs = set()
    for p in ports:
        if p["type"] == "outlet":
            outs |= set(map(tuple, port_cells(p)))
    for p in ports:
        if p["type"] == "inlet":
            if not (_bfs(b, [tuple(c) for c in port_cells(p)]) & outs):
                return False
    return True


def decode_binary(vae, z, bc_t):
    with torch.no_grad():
        logits = vae.decode(z, bc_t)
        return (torch.sigmoid(logits[0, 0]) > THRESHOLD).float().cpu().numpy()


# ============================================================
# PART 1 — z-SWEEP : the honest diagnostic
# ============================================================
def z_sweep(vae, bc_t, ports, is_desig, dims=tuple(range(LATENT_DIM)), steps=7, span=3.0,
            out="diversity_zsweep.png"):
    """Fix BC, vary ONE z dim at a time over [-span, span]. If topology is frozen,
    every row looks identical → z carries no structure."""
    vals = np.linspace(-span, span, steps)
    fig, axes = plt.subplots(len(dims), steps, figsize=(1.6*steps, 1.6*len(dims)))
    for i, d in enumerate(dims):
        for j, v in enumerate(vals):
            z = torch.zeros(1, LATENT_DIM, device=DEVICE)
            z[0, d] = v
            dz = decode_binary(vae, z, bc_t)
            vol = float(dz[is_desig].mean())
            ax = axes[i, j]
            ax.imshow(dz.T, cmap="gray_r", origin="lower", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(f"{v:+.1f}", fontsize=7)
            if j == 0: ax.set_ylabel(f"z[{d}]", fontsize=7)
            ax.text(2, 2, f"{vol:.2f}", fontsize=5, color="red")
    plt.suptitle("z-sweep — each ROW varies one latent dim (BC fixed). "
                 "Frozen rows ⇒ z carries no topology.", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"[zsweep] saved → {out}")

    # quantify how much each dim moves the *designable* region (mean abs pixel change)
    print("[zsweep] per-dim structural sensitivity (mean |Δ| over designable cells):")
    base = decode_binary(vae, torch.zeros(1, LATENT_DIM, device=DEVICE), bc_t)
    for d in dims:
        z = torch.zeros(1, LATENT_DIM, device=DEVICE); z[0, d] = span
        dz = decode_binary(vae, z, bc_t)
        change = np.abs(dz[is_desig] - base[is_desig]).mean()
        print(f"    z[{d:2d}] : {change:.4f}")


# ============================================================
# PART 2 + 3 — SAMPLE many z, keep feasible, count STRUCTURALLY distinct
# ============================================================
def iou(a, b, is_desig):
    A = (a > THRESHOLD)[is_desig]; B = (b > THRESHOLD)[is_desig]
    inter = np.logical_and(A, B).sum(); union = np.logical_or(A, B).sum()
    return inter / union if union > 0 else 1.0

def count_distinct(designs, is_desig, iou_thresh=0.85):
    """Greedy clustering: a design joins an existing cluster if IoU>thresh with its
    representative; else it starts a new cluster. #clusters = #distinct topologies.
    Higher iou_thresh = stricter (more clusters)."""
    reps = []
    for d in designs:
        if all(iou(d, r, is_desig) <= iou_thresh for r in reps):
            reps.append(d)
    return reps

def sample_diversity(vae, bc_t, ports, is_desig, n_samples=40, sigma=1.0,
                     iou_thresh=0.85, out="diversity_samples.png"):
    feas = []
    for _ in range(n_samples):
        z = (torch.randn(1, LATENT_DIM, device=DEVICE) * sigma)
        dz = decode_binary(vae, z, bc_t)
        if connected(dz, ports):
            feas.append(dz)
    print(f"[sample] {len(feas)}/{n_samples} feasible (connected)")

    if not feas:
        print("[sample] no feasible designs — cannot assess diversity"); return

    reps = count_distinct(feas, is_desig, iou_thresh)
    print(f"[sample] DISTINCT feasible topologies (IoU>{iou_thresh}): {len(reps)}")
    print(f"         ESO produces 1 design for this BC by construction.")

    # pairwise IoU stats over feasible set (lower mean = more diverse)
    if len(feas) > 1:
        ious = [iou(feas[i], feas[j], is_desig)
                for i in range(len(feas)) for j in range(i+1, len(feas))]
        print(f"[sample] pairwise IoU over feasible set: "
              f"mean={np.mean(ious):.3f}  min={np.min(ious):.3f}  max={np.max(ious):.3f}")
        print(f"         (IoU≈1 everywhere ⇒ all the SAME topology ⇒ no real diversity)")

    # show up to 12 distinct representatives
    tag = ports_to_tag(ports)
    k = min(12, len(reps))
    cols = 4; rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows), squeeze=False)
    for idx in range(rows*cols):
        ax = axes[idx//cols][idx%cols]; ax.axis("off")
        if idx < k:
            d = reps[idx]
            ax.imshow(d.T, cmap="gray_r", origin="lower", vmin=0, vmax=1)
            for p in ports:
                r = p["range"]; col = "green" if p["type"]=="inlet" else "red"
                if p["wall"]=="left":   ax.plot([0,0],[r.start,r.stop],col,lw=3)
                elif p["wall"]=="right":ax.plot([Nx-1,Nx-1],[r.start,r.stop],col,lw=3)
                elif p["wall"]=="bottom":ax.plot([r.start,r.stop],[0,0],col,lw=3)
                elif p["wall"]=="top":  ax.plot([r.start,r.stop],[Ny-1,Ny-1],col,lw=3)
            ax.set_title(f"distinct #{idx+1}  vol={d[is_desig].mean():.2f}", fontsize=8)
    plt.suptitle(f"Distinct feasible topologies for one BC  "
                 f"({len(reps)} found / {len(feas)} feasible / {n_samples} sampled)",
                 fontsize=10)
    plt.tight_layout()
    #plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    # save the plot with tag in filename
    base, ext = os.path.splitext(out)
    out = f"{base}_{tag}{ext}"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"[sample] saved → {out}")


def main():
    ap = argparse.ArgumentParser()
    #ap.add_argument("--port", action="append", nargs=3,
    #                metavar=("TYPE","WALL","CENTER"), required=True)
    ap.add_argument("--vae_path", default="vae_best_new.pth")
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="std of z sampling; try 1.0, also 1.5/2.0 to probe wider")
    ap.add_argument("--iou_thresh", type=float, default=0.85,
                    help="IoU above which two designs count as the SAME topology")
    #ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ports = []
    # for t, w_, c in args.port:
    #     c = int(c)
    #     lo = max(WALL, c - PORT_HEIGHT//2); hi = min(Ny - WALL, c + PORT_HEIGHT//2)
    #     ports.append({"type": t, "wall": w_, "range": slice(lo, hi), "center": c})
    for i in range(10):
        ports = generate_valid_ports(n_inlets=2)
        vae = FluidVAE(latent_dim=LATENT_DIM).to(DEVICE)
        vae.load_state_dict(torch.load(args.vae_path, map_location=DEVICE, weights_only=True))
        vae.eval()

        bc_t = torch.FloatTensor(make_bc_mask(ports)).unsqueeze(0).to(DEVICE)
        masks = build_bc_masks(Nx, Ny, WALL, ports)
        is_desig = (~masks["solid_mask"] & ~masks["fluid_mask"]
                    & ~masks["orifice_mask"]).cpu().numpy()
        #torch.manual_seed(args.seed)
        #np.random.seed(args.seed)
        print("="*60)
        print("BC: " + " | ".join(f"{p['type']}@{p['wall']}:{p['center']}" for p in ports))
        print("="*60)

        z_sweep(vae, bc_t, ports, is_desig)
        sample_diversity(vae, bc_t, ports, is_desig,
                        n_samples=args.n_samples, sigma=args.sigma,
                        iou_thresh=args.iou_thresh)

if __name__ == "__main__":
    main()