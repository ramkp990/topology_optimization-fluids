"""
LBM Topology Optimization — Multi-Port Dataset Generator
=========================================================
Up to 2 inlets (left wall) + 2 outlets (right wall).
Number of ports per run is sampled randomly (1 or 2 each).
Produces 3 HDF5 files per batch run:
  dataset_final_runN.h5         — one best design per optimization
  dataset_intermediate_runN.h5  — trajectory snapshots (any inlet_y > any outlet_y)
  dataset_all_runN.h5           — every feasible snapshot
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import h5py
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
# =========================================================
# FIXED SIMULATION PARAMETERS
# =========================================================
Nx = 64
Ny = 64
WALL_THICKNESS  = 4
PORT_HEIGHT     = int(0.10 * Ny)      # 6 cells tall per slot

timesteps_no_grad = 2000
timesteps_grad    = 300
tau   = 0.6
omega = 1.0 / tau

REMOVAL_FRACTION = 0.1
ALPHA_MAX        = 100.0
MAX_ESO_ITERS    = 30

LEFT_NOSLIP_X  = WALL_THICKNESS
RIGHT_NOSLIP_X = Nx - WALL_THICKNESS - 1
BOT_NOSLIP_Y   = WALL_THICKNESS
TOP_NOSLIP_Y   = Ny - WALL_THICKNESS - 1
P_IN_X         = LEFT_NOSLIP_X
P_OUT_X        = RIGHT_NOSLIP_X

inlet_interior_x  = slice(1, WALL_THICKNESS)           # x = 1,2,3
outlet_interior_x = slice(Nx - WALL_THICKNESS, Nx - 1) # x = 60,61,62

# =========================================================
# DATASET / RUN PARAMETERS  ← edit these
# =========================================================
NUM_DESIGNS_TARGET = 10000   # overall goal across all runs
BATCH_SIZE         = 3      # designs to collect per script execution
VOLUME_THRESHOLD   = 0.25    # upper bound for random target volume
CHECKPOINT_FILE    = "./data/new/generation_checkpoint.json"
OUTPUT_DIR         = "./data/new"

SAVE_FINAL_ONLY   = True
SAVE_INTERMEDIATE = True   # trajectory snapshots where any inlet_y > any outlet_y
SAVE_ALL_FEASIBLE = True

# =========================================================
# D2Q9 CONSTANTS
# =========================================================
c = torch.tensor(
    [[0,0],[1,0],[0,1],[-1,0],[0,-1],
     [1,1],[-1,1],[-1,-1],[1,-1]],
    dtype=torch.int64, device=device
)
c_float = c.float()
w = torch.tensor(
    [4/9, 1/9, 1/9, 1/9, 1/9,
     1/36, 1/36, 1/36, 1/36],
    device=device
)

# Re-created fresh for every optimization run
topology = None


# =========================================================
# SLOT HELPER
# =========================================================
def make_slot(center, height=PORT_HEIGHT):
    """Clamp slot so it never encroaches on the wall bands."""
    lo = max(WALL_THICKNESS, int(center - height // 2))
    hi = min(Ny - WALL_THICKNESS, int(center + height // 2))
    return slice(lo, hi)


# =========================================================
# MASK CREATION  — accepts 1-2 inlet centers, 1-2 outlet centers
# =========================================================
def create_masks(inlet_centers, outlet_centers):
    """
    inlet_centers  : list of 1 or 2 Y positions on the LEFT wall
    outlet_centers : list of 1 or 2 Y positions on the RIGHT wall
    Returns a dict used by apply_bcs / simulate / feasibility checks.
    """
    inlet_ranges  = [make_slot(c) for c in inlet_centers]
    outlet_ranges = [make_slot(c) for c in outlet_centers]

    # solid wall bands
    solid = torch.zeros(Nx, Ny, device=device, dtype=torch.bool)
    solid[0:WALL_THICKNESS, :]  = True
    solid[-WALL_THICKNESS:, :]  = True
    solid[:, 0:WALL_THICKNESS]  = True
    solid[:, -WALL_THICKNESS:]  = True
    for r in inlet_ranges:
        solid[0:WALL_THICKNESS, r] = False
    for r in outlet_ranges:
        solid[-WALL_THICKNESS:, r] = False

    # orifice cells  (inside wall bands at the slots)
    orifice = torch.zeros(Nx, Ny, device=device, dtype=torch.bool)
    for r in inlet_ranges:
        orifice[0:WALL_THICKNESS, r] = True
    for r in outlet_ranges:
        orifice[-WALL_THICKNESS:, r] = True

    # Zou-He boundary cells  (x=0 for inlets, x=Nx-1 for outlets)
    fluid = torch.zeros(Nx, Ny, device=device, dtype=torch.bool)
    for r in inlet_ranges:
        fluid[0, r] = True
    for r in outlet_ranges:
        fluid[Nx - 1, r] = True

    # y-rows where left/right no-slip applies
    left_ns = torch.ones(Ny, dtype=torch.bool, device=device)
    for r in inlet_ranges:
        left_ns[r] = False

    right_ns = torch.ones(Ny, dtype=torch.bool, device=device)
    for r in outlet_ranges:
        right_ns[r] = False

    # adjacency protection: optimizer never removes a cell adjacent to any orifice
    dilated = F.max_pool2d(
        orifice.float().unsqueeze(0).unsqueeze(0),
        kernel_size=3, stride=1, padding=1
    )[0, 0].bool()

    return {
        "inlet_centers":      inlet_centers,
        "outlet_centers":     outlet_centers,
        "inlet_ranges":       inlet_ranges,
        "outlet_ranges":      outlet_ranges,
        "solid_mask":         solid,
        "orifice_mask":       orifice,
        "fluid_mask":         fluid,
        "left_noslip_ymask":  left_ns,
        "right_noslip_ymask": right_ns,
        "fluid_dilated":      dilated,
    }


# =========================================================
# CORE LBM
# =========================================================
def equilibrium(rho, u):
    cu   = torch.einsum("ia,xya->xyi", c_float, u)
    usqr = (u ** 2).sum(-1, keepdim=True)
    return rho.unsqueeze(-1) * w * (1 + 3*cu + 4.5*cu**2 - 1.5*usqr)


def streaming(f):
    f_out = torch.empty_like(f)
    for i in range(9):
        f_out[:, :, i] = torch.roll(
            f[:, :, i],
            shifts=(c[i, 0].item(), c[i, 1].item()),
            dims=(0, 1)
        )
    return f_out


# =========================================================
# BOUNDARY CONDITIONS  (multi-port, all slots)
# =========================================================
def apply_bcs(f, u_in, masks):
    f  = f.clone()
    IR = masks["inlet_ranges"]
    OR = masks["outlet_ranges"]

    # global top/bottom no-slip
    f[:, BOT_NOSLIP_Y, 2] = f[:, BOT_NOSLIP_Y, 4]
    f[:, BOT_NOSLIP_Y, 5] = f[:, BOT_NOSLIP_Y, 7]
    f[:, BOT_NOSLIP_Y, 6] = f[:, BOT_NOSLIP_Y, 8]
    f[:, TOP_NOSLIP_Y, 4] = f[:, TOP_NOSLIP_Y, 2]
    f[:, TOP_NOSLIP_Y, 7] = f[:, TOP_NOSLIP_Y, 5]
    f[:, TOP_NOSLIP_Y, 8] = f[:, TOP_NOSLIP_Y, 6]

    # left wall no-slip  (skips inlet rows)
    m = masks["left_noslip_ymask"]
    f[LEFT_NOSLIP_X, m, 1] = f[LEFT_NOSLIP_X, m, 3]
    f[LEFT_NOSLIP_X, m, 5] = f[LEFT_NOSLIP_X, m, 7]
    f[LEFT_NOSLIP_X, m, 8] = f[LEFT_NOSLIP_X, m, 6]

    # right wall no-slip  (skips outlet rows)
    m = masks["right_noslip_ymask"]
    f[RIGHT_NOSLIP_X, m, 3] = f[RIGHT_NOSLIP_X, m, 1]
    f[RIGHT_NOSLIP_X, m, 6] = f[RIGHT_NOSLIP_X, m, 8]
    f[RIGHT_NOSLIP_X, m, 7] = f[RIGHT_NOSLIP_X, m, 5]

    # inlet slot edge bounce-back  (interior orifice cols x=1,2,3)
    xi = inlet_interior_x
    for r in IR:
        bot, top = r.start, r.stop - 1
        f[xi, bot, 2] = f[xi, bot, 4];  f[xi, bot, 5] = f[xi, bot, 7];  f[xi, bot, 6] = f[xi, bot, 8]
        f[xi, top, 4] = f[xi, top, 2];  f[xi, top, 7] = f[xi, top, 5];  f[xi, top, 8] = f[xi, top, 6]

    # outlet slot edge bounce-back  (interior orifice cols x=60,61,62)
    xo = outlet_interior_x
    for r in OR:
        bot, top = r.start, r.stop - 1
        f[xo, bot, 2] = f[xo, bot, 4];  f[xo, bot, 5] = f[xo, bot, 7];  f[xo, bot, 6] = f[xo, bot, 8]
        f[xo, top, 4] = f[xo, top, 2];  f[xo, top, 7] = f[xo, top, 5];  f[xo, top, 8] = f[xo, top, 6]

    # Zou-He velocity BC  (x=0, all inlets)
    for r in IR:
        y = r
        rho_in = (
            f[0, y, 0] + f[0, y, 2] + f[0, y, 4]
            + 2.0 * (f[0, y, 3] + f[0, y, 6] + f[0, y, 7])
        ) / (1.0 - u_in)
        f[0, y, 1] = f[0, y, 3] + (2.0/3.0)*rho_in*u_in
        f[0, y, 5] = f[0, y, 7] - 0.5*(f[0,y,2]-f[0,y,4]) + (1.0/6.0)*rho_in*u_in
        f[0, y, 8] = f[0, y, 6] + 0.5*(f[0,y,2]-f[0,y,4]) + (1.0/6.0)*rho_in*u_in

    # Zou-He pressure BC  (x=Nx-1, all outlets)
    for r in OR:
        y = r
        rho_out = 1.0
        u_out   = -1.0 + (
            f[-1, y, 0] + f[-1, y, 2] + f[-1, y, 4]
            + 2.0 * (f[-1, y, 1] + f[-1, y, 5] + f[-1, y, 8])
        ) / rho_out
        f[-1, y, 3] = f[-1, y, 1] - (2.0/3.0)*rho_out*u_out
        f[-1, y, 7] = f[-1, y, 5] + 0.5*(f[-1,y,2]-f[-1,y,4]) - (1.0/6.0)*rho_out*u_out
        f[-1, y, 6] = f[-1, y, 8] - 0.5*(f[-1,y,2]-f[-1,y,4]) - (1.0/6.0)*rho_out*u_out

    return f


# =========================================================
# LBM STEP
# =========================================================
def lbm_step(f, density_physics, alpha_max, t, masks):
    rho = f.sum(-1).clamp(min=0.5, max=10.0)
    u   = torch.einsum("xyi,ia->xya", f, c_float) / rho.unsqueeze(-1)
    u   = torch.clamp(u, min=-0.3, max=0.3)

    alpha     = alpha_max * (1.0 - density_physics)
    u_eq      = u / (1.0 + alpha.unsqueeze(-1))
    feq       = equilibrium(rho, u_eq)
    omega_eff = omega * density_physics + 1.0 * (1.0 - density_physics)
    f = f - omega_eff.unsqueeze(-1) * (f - feq)
    f = streaming(f)

    current_u = 0.05 * min(t / 500.0, 1.0)
    f = apply_bcs(f, u_in=current_u, masks=masks)
    return f


# =========================================================
# SIMULATION
# =========================================================
def filter_density(d):
    return F.avg_pool2d(
        d.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
    )[0, 0]


def simulate(alpha_max, masks):
    """Run warm-up + gradient phase. Returns (pressure_drop, filtered_density)."""
    global topology

    f = torch.ones(Nx, Ny, 9, device=device) * w

    density = topology.clamp(0.0, 1.0)
    density = torch.where(masks["solid_mask"], torch.zeros_like(density), density)
    density = torch.where(masks["fluid_mask"], torch.ones_like(density),  density)

    with torch.no_grad():
        for t in range(timesteps_no_grad):
            f = lbm_step(f, density, alpha_max, t, masks)

    p_in_accum = p_out_accum = 0.0
    for t in range(timesteps_no_grad, timesteps_no_grad + timesteps_grad):
        f = lbm_step(f, density, alpha_max, t, masks)
        rho_t = f.sum(-1)
        p_in  = sum(rho_t[P_IN_X,  r].mean() for r in masks["inlet_ranges"])  / len(masks["inlet_ranges"])
        p_out = sum(rho_t[P_OUT_X, r].mean() for r in masks["outlet_ranges"]) / len(masks["outlet_ranges"])
        p_in_accum  = p_in_accum  + p_in
        p_out_accum = p_out_accum + p_out

    cs2    = 1.0 / 3.0
    mean_dp = (p_in_accum - p_out_accum) / timesteps_grad * cs2

    d_filt = filter_density(topology.clamp(0.0, 1.0))
    d_filt = torch.where(masks["solid_mask"], torch.zeros_like(d_filt), d_filt)
    d_filt = torch.where(masks["fluid_mask"], torch.ones_like(d_filt),  d_filt)

    return mean_dp, d_filt


# =========================================================
# FEASIBILITY
# =========================================================
def check_connectivity(density_np, masks):
    """BFS from ALL inlet cells to ANY outlet cell through fluid (>0.5)."""
    binary = (density_np > 0.5).astype(np.uint8)

    starts = []
    for r in masks["inlet_ranges"]:
        for y in range(r.start, r.stop):
            if binary[0, y]:
                starts.append((0, y))
    if not starts:
        return False

    outlet_set = set()
    for r in masks["outlet_ranges"]:
        for y in range(r.start, r.stop):
            outlet_set.add((Nx - 1, y))

    visited = set(starts)
    queue   = deque(starts)
    while queue:
        x, y = queue.popleft()
        if (x, y) in outlet_set:
            return True
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx_, ny_ = x+dx, y+dy
            if 0 <= nx_ < Nx and 0 <= ny_ < Ny:
                if binary[nx_, ny_] and (nx_, ny_) not in visited:
                    visited.add((nx_, ny_))
                    queue.append((nx_, ny_))
    return False


def is_feasible(density_tensor, dp_val, vol, masks,
                vol_min=0.10, vol_max=0.40,
                dp_min=0.005, dp_max=5.0):
    if vol < vol_min:  return False, f"vol_low:{vol:.3f}"
    if vol > vol_max:  return False, f"vol_high:{vol:.3f}"
    if dp_val < dp_min: return False, f"dp_low:{dp_val:.4f}"
    if dp_val > dp_max: return False, f"dp_high:{dp_val:.4f}"
    if not check_connectivity(density_tensor.detach().cpu().numpy(), masks):
        return False, "disconnected"
    return True, "OK"


# =========================================================
# CHECKPOINT
# =========================================================
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as fh:
            ckpt = json.load(fh)
        print(f"Loaded checkpoint: {ckpt['total_designs']} designs collected so far.")
        return ckpt
    print("No checkpoint found. Starting fresh.")
    return {"total_designs": 0, "used_bc_configs": [], "run_number": 0}


def save_checkpoint(ckpt):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    ckpt["last_saved"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w") as fh:
        json.dump(ckpt, fh, indent=2)


# =========================================================
# VISUALIZATION
# =========================================================
def save_density_plot(density_np, masks, filepath, title=None):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(density_np.T, origin="lower", cmap="gray_r",
              vmin=0, vmax=1, interpolation="none")

    inlet_colors  = ["#2ecc71", "#1a8a4a"]
    outlet_colors = ["#e74c3c", "#922b21"]

    for i, r in enumerate(masks["inlet_ranges"]):
        ax.plot([0, 0], [r.start, r.stop - 1],
                color=inlet_colors[i % 2], linewidth=3)
    for i, r in enumerate(masks["outlet_ranges"]):
        ax.plot([Nx-1, Nx-1], [r.start, r.stop - 1],
                color=outlet_colors[i % 2], linewidth=3)

    if title:
        ax.set_title(title, fontsize=7)
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)


# =========================================================
# HDF5 SAVE
# =========================================================
def _pack_bc(bc_info_list):
    """Pack variable-length port lists into fixed (N,2) arrays; pad unused slots with -1."""
    n = len(bc_info_list)
    in_arr  = np.full((n, 2), -1, dtype=np.int32)
    out_arr = np.full((n, 2), -1, dtype=np.int32)
    for i, b in enumerate(bc_info_list):
        for j, v in enumerate(b["inlet_centers"]):
            in_arr[i, j] = v
        for j, v in enumerate(b["outlet_centers"]):
            out_arr[i, j] = v
    return in_arr, out_arr


def save_dataset_file(filepath, densities, pressure_drops, volumes, bc_info, dataset_type):
    if not densities:
        print(f"  Skipping {os.path.basename(filepath)} — no data.")
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    in_arr, out_arr = _pack_bc(bc_info)

    with h5py.File(filepath, "w") as hf:
        hf.create_dataset("density",         data=torch.stack(densities).numpy(), compression="gzip")
        hf.create_dataset("pressure_drop",   data=np.array(pressure_drops, dtype=np.float32))
        hf.create_dataset("volume_fraction", data=np.array(volumes,        dtype=np.float32))
        # bc_inlet_y / bc_outlet_y: shape (N,2). Second column is -1 if only 1 port.
        hf.create_dataset("bc_inlet_y",      data=in_arr)
        hf.create_dataset("bc_outlet_y",     data=out_arr)
        hf.create_dataset("num_inlets",      data=np.array([b["num_inlets"]      for b in bc_info], dtype=np.int32))
        hf.create_dataset("num_outlets",     data=np.array([b["num_outlets"]     for b in bc_info], dtype=np.int32))
        hf.create_dataset("eso_iteration",   data=np.array([b["iteration"]       for b in bc_info], dtype=np.int32))
        hf.create_dataset("optimization_id", data=np.array([b["opt_id"]          for b in bc_info], dtype=np.int32))
        hf.create_dataset("is_intermediate", data=np.array([b["is_intermediate"] for b in bc_info], dtype=bool))
        hf.create_dataset("metadata",
            data=[json.dumps(b).encode("utf-8") for b in bc_info])

        hf.attrs["num_designs"]  = len(densities)
        hf.attrs["dataset_type"] = dataset_type
        hf.attrs["nx"]           = Nx
        hf.attrs["ny"]           = Ny
        hf.attrs["timestamp"]    = datetime.now().isoformat()

    dps = pressure_drops
    vols = volumes
    print(f"  Saved {os.path.basename(filepath):45s} "
          f"| {len(densities):4d} designs "
          f"| dp=[{min(dps):.3f},{max(dps):.3f}] "
          f"| vol=[{min(vols):.3f},{max(vols):.3f}]")


# =========================================================
# RANDOM BC SAMPLER  (1 or 2 inlets, 1 or 2 outlets)
# =========================================================
_PORT_MARGIN = WALL_THICKNESS + PORT_HEIGHT // 2 + 2   # keep ports away from corners


def _sample_centers(n, lo, hi, min_gap):
    """Sample n non-overlapping center positions in [lo, hi)."""
    centers = []
    for _ in range(200):
        c = int(np.random.randint(lo, hi))
        if all(abs(c - e) >= min_gap for e in centers):
            centers.append(c)
        if len(centers) == n:
            return centers
    return centers   # may be fewer than n if space is tight


def sample_bc_config():
    """Return (inlet_centers, outlet_centers) — each a sorted list of 1 or 2 ints."""
    lo, hi = _PORT_MARGIN, Ny - _PORT_MARGIN
    min_gap = PORT_HEIGHT + 2

    n_in  = int(np.random.choice([1, 2]))
    n_out = int(np.random.choice([1, 2]))

    in_c  = _sample_centers(n_in,  lo, hi, min_gap)
    out_c = _sample_centers(n_out, lo, hi, min_gap)

    # fall back to single port if overlap rejection failed
    if len(in_c)  == 0: in_c  = [int(np.random.randint(lo, hi))]
    if len(out_c) == 0: out_c = [int(np.random.randint(lo, hi))]

    return sorted(in_c), sorted(out_c)


# =========================================================
# MAIN
# =========================================================
def main():
    global topology

    # load checkpoint
    ckpt            = load_checkpoint()
    used_bc_configs = set(
        (tuple(pair[0]), tuple(pair[1])) for pair in ckpt["used_bc_configs"]
    )
    start_count  = ckpt["total_designs"]
    run_number   = ckpt["run_number"] + 1
    global_opt_id = start_count

    print(f"Run #{run_number} | batch_size={BATCH_SIZE} | total_target={NUM_DESIGNS_TARGET}")
    print(f"  Already collected : {start_count}")
    print(f"  BC configs used   : {len(used_bc_configs)}")
    print("=" * 70)

    # batch accumulators
    final_d,  final_dp,  final_vol,  final_bc  = [], [], [], []
    inter_d,  inter_dp,  inter_vol,  inter_bc  = [], [], [], []
    all_d,    all_dp,    all_vol,    all_bc     = [], [], [], []

    completed = 0
    attempts  = 0

    while (completed < BATCH_SIZE
           and start_count + completed < NUM_DESIGNS_TARGET
           and attempts < BATCH_SIZE * 10):

        attempts += 1

        # sample port config
        in_c, out_c = sample_bc_config()
        bc_key = (tuple(in_c), tuple(out_c))
        if bc_key in used_bc_configs:
            continue

        masks         = create_masks(in_c, out_c)
        is_designable = (~masks["solid_mask"]
                         & ~masks["fluid_mask"]
                         & ~masks["orifice_mask"])

        # fresh topology + randomised volume target
        topology   = torch.nn.Parameter(torch.ones(Nx, Ny, device=device))
        target_vol = float(np.random.uniform(0.20, VOLUME_THRESHOLD))

        best_density = None
        best_dp_val  = float("inf")
        best_vol_val = 1.0
        best_iter    = 0

        # "intermediate" condition: any inlet center above any outlet center
        save_inter = any(iy > oy for iy in in_c for oy in out_c)

        print(f"\n  attempt {attempts:3d} | inlets={in_c} outlets={out_c} "
              f"| target_vol={target_vol:.2f} | save_inter={save_inter}")

        # --- ESO loop ---
        for it in range(MAX_ESO_ITERS):
            if topology.grad is not None:
                topology.grad.zero_()

            dp, density = simulate(ALPHA_MAX, masks)
            vol_val = topology[is_designable].mean().item()
            dp_val  = dp.item()
            dp.backward()

            feasible, reason = is_feasible(
                density, dp_val, vol_val, masks,
                vol_min=0.10, vol_max=0.40,
                dp_min=0.005, dp_max=5.0
            )

            # track best feasible snapshot
            if feasible and dp_val < best_dp_val:
                best_density = density.detach().cpu().clone()
                best_dp_val  = dp_val
                best_vol_val = vol_val
                best_iter    = it

            bc_meta = {
                "inlet_centers":   in_c,
                "outlet_centers":  out_c,
                "num_inlets":      len(in_c),
                "num_outlets":     len(out_c),
                "iteration":       it,
                "opt_id":          global_opt_id,
                "feasibility":     reason,
                "is_intermediate": True,
            }

            # file 2: intermediate trajectory
            #if SAVE_INTERMEDIATE and save_inter and it >= 10 and feasible:
            #    inter_d.append(density.detach().cpu().clone())
            #    inter_dp.append(dp_val)
            #    inter_vol.append(vol_val)
            #    inter_bc.append(bc_meta.copy())

            # file 3: all feasible
            if SAVE_ALL_FEASIBLE and it >= 10 and feasible:
                all_d.append(density.detach().cpu().clone())
                all_dp.append(dp_val)
                all_vol.append(vol_val)
                all_bc.append(bc_meta.copy())

            # --- ESO removal ---
            if vol_val > target_vol:
                dL = topology.grad
                if dL is not None:
                    is_fluid = topology > 0.5
                    eligible = is_fluid & is_designable & ~masks["fluid_dilated"]
                    sens     = dL[eligible]
                    if sens.numel() > 0:
                        n_rm = max(int(eligible.sum().item() * REMOVAL_FRACTION), 1)
                        n_rm = min(n_rm, sens.numel())
                        _, idx = torch.topk(sens, n_rm, largest=True)
                        flat_elig = torch.where(eligible.flatten())[0]
                        rm = torch.zeros_like(topology, dtype=torch.bool)
                        rm.flatten()[flat_elig[idx]] = True

                        # outlet seal guard — check every outlet slot
                        sealed = False
                        for r in masks["outlet_ranges"]:
                            guard = rm[max(0, RIGHT_NOSLIP_X - 2):RIGHT_NOSLIP_X + 1, r]
                            if guard.all():
                                sealed = True
                                break
                        if not sealed:
                            topology.data[rm] = 0.0
            else:
                # volume target reached; no more removals possible
                if best_density is not None and it >= 10:
                    break

        # --- record final best design from this optimization run ---
        if best_density is not None:
            bc_final = {
                "inlet_centers":   in_c,
                "outlet_centers":  out_c,
                "num_inlets":      len(in_c),
                "num_outlets":     len(out_c),
                "iteration":       best_iter,
                "opt_id":          global_opt_id,
                "feasibility":     "best_from_run",
                "is_intermediate": False,
            }

            if SAVE_FINAL_ONLY:
                final_d.append(best_density)
                final_dp.append(best_dp_val)
                final_vol.append(best_vol_val)
                final_bc.append(bc_final)

            # PNG
            os.makedirs("./output/plots", exist_ok=True)
            plot_path = (
                f"./output/plots/"
                f"opt{global_opt_id:04d}"
                f"_in{'_'.join(map(str, in_c))}"
                f"_out{'_'.join(map(str, out_c))}"
                f"_dp{best_dp_val:.3f}.png"
            )
            save_density_plot(
                best_density.numpy(), masks, plot_path,
                title=(f"opt#{global_opt_id} | in={in_c} out={out_c} | "
                       f"Δp={best_dp_val:.3f} vol={best_vol_val:.2f}")
            )

            # checkpoint update
            used_bc_configs.add(bc_key)
            completed     += 1
            global_opt_id += 1

            ckpt["total_designs"]   = start_count + completed
            ckpt["used_bc_configs"] = [
                [list(k[0]), list(k[1])] for k in used_bc_configs
            ]
            ckpt["run_number"] = run_number
            save_checkpoint(ckpt)

            print(f"  ✓ design {start_count+completed:4d}/{NUM_DESIGNS_TARGET} | "
                  f"Δp={best_dp_val:.4f} | vol={best_vol_val:.3f} | "
                  f"inlets={in_c} outlets={out_c}")
        else:
            print(f"  ✗ no feasible design — inlets={in_c} outlets={out_c}")

    # =========================================================
    # SAVE HDF5 FILES
    # =========================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fp_final = f"{OUTPUT_DIR}/dataset_final_run{run_number}.h5"
    fp_inter = f"{OUTPUT_DIR}/dataset_intermediate_run{run_number}.h5"
    fp_all   = f"{OUTPUT_DIR}/dataset_all_run{run_number}.h5"

    print("\nSaving HDF5 files...")
    save_dataset_file(fp_final, final_d, final_dp, final_vol, final_bc, "final_designs_only")
    #save_dataset_file(fp_inter, inter_d, inter_dp, inter_vol, inter_bc, "intermediate_inlet_above_outlet")
    save_dataset_file(fp_all,   all_d,   all_dp,   all_vol,   all_bc,   "all_feasible_designs")

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"  File 1 (final only):    {len(final_d):5d} designs  →  {fp_final}")
    print(f"  File 2 (intermediate):  {len(inter_d):5d} designs  →  {fp_inter}")
    print(f"  File 3 (all feasible):  {len(all_d):5d} designs  →  {fp_all}")
    print(f"  Attempts this run:      {attempts}")
    print(f"  Completed this run:     {completed}")
    if attempts:
        print(f"  Success rate:           {completed / attempts * 100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()