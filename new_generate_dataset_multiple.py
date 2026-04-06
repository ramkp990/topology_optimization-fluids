import torch
import torch.nn.functional as F
import numpy as np
import os
from pyevtk.hl import gridToVTK
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import h5py
import json
from datetime import datetime
from collections import deque


device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# PARAMETERS
# =========================================================
Nx = 64
Ny = 64
WALL_THICKNESS = 4
timesteps_no_grad = 2000
timesteps_grad    = 300
tau   = 0.6
omega = 1.0 / tau
target_volume = 0.2
REMOVAL_FRACTION = 0.1

topology = None
# =========================================================
# GENERIC PORT SYSTEM  (ports can be on ANY wall)
# =========================================================

def make_slot(center, height):
    lo = int(center - height // 2)
    hi = int(center + height // 2)
    return slice(lo, hi)

PORT_HEIGHT = int(0.10 * Ny)

# You can place ports on ANY wall now:
# wall ∈ {"left","right","top","bottom"}

# =========================================================
# RANDOM PORT GENERATOR (max 2 inlets / max 2 outlets)
# =========================================================

WALLS = ["left","right","top","bottom"]

def random_center(wall):
    margin = WALL_THICKNESS + PORT_HEIGHT//2 + 2
    if wall in ["left","right"]:
        return np.random.randint(margin, Ny-margin)
    else:
        return np.random.randint(margin, Nx-margin)

def sample_ports(n, port_type):
    ports = []
    attempts = 0

    while len(ports) < n and attempts < 50:
        wall   = np.random.choice(WALLS)
        center = random_center(wall)

        # avoid overlap on same wall
        ok = True
        for p in ports:
            if p["wall"] == wall:
                c_old = (p["range"].start + p["range"].stop)//2
                if abs(center - c_old) < PORT_HEIGHT + 5:
                    ok = False
        if ok:
            ports.append({
                "type": port_type,
                "wall": wall,
                "range": make_slot(center, PORT_HEIGHT),
                "center": center
            })
        attempts += 1

    return ports

def ports_overlap(port_a, port_b, gap=5):
    """
    Returns True if two ports conflict — same wall AND their
    slot ranges are closer than `gap` pixels from each other.
    """
    if port_a["wall"] != port_b["wall"]:
        return False   # different walls can never spatially overlap

    ra = port_a["range"]
    rb = port_b["range"]

    # slots overlap if one starts before the other ends
    # with gap: treat each slot as expanded by gap//2 on each side
    lo_a, hi_a = ra.start - gap, ra.stop + gap
    lo_b, hi_b = rb.start, rb.stop

    return lo_a < hi_b and lo_b < hi_a


'''
# sample random configuration
n_in  = np.random.choice([1,2])
n_out = np.random.choice([1,2])

ports = sample_ports(n_in,"inlet") + sample_ports(n_out,"outlet")

inlet_ports  = [p for p in ports if p["type"]=="inlet"]
outlet_ports = [p for p in ports if p["type"]=="outlet"]

print("\nRandom ports:")
for p in ports:
    print(p)
'''
def bc_to_filename(inlets, outlets):
    def join_ports(ports):
        return "_".join([f"{w}_{c}" for w,c in ports])

    return (
        f"{len(inlets)}_inlets_{join_ports(inlets)}_"
        f"{len(outlets)}_outlets_{join_ports(outlets)}"
    )


# =========================================================
# BUILD GENERIC WALL + PORT MASKS
# =========================================================


def build_bc_masks(Nx, Ny, WALL_THICKNESS, ports):
    """
    Build all boundary condition masks and port metadata.
    Returns a dictionary with all required tensors and ranges.
    """
    solid = torch.zeros(Nx, Ny, device=device, dtype=torch.bool)
    orifice = torch.zeros_like(solid)
    fluid = torch.zeros_like(solid)

    # Build outer walls
    solid[0:WALL_THICKNESS, :]  = True
    solid[-WALL_THICKNESS:, :]  = True
    solid[:, 0:WALL_THICKNESS]  = True
    solid[:, -WALL_THICKNESS:]  = True

    def carve_port(port, solid, orifice, fluid):
        r = port["range"]
        wall = port["wall"]

        if wall == "left":
            solid[0:WALL_THICKNESS, r] = False
            orifice[0:WALL_THICKNESS, r] = True
            fluid[0, r] = True
        elif wall == "right":
            solid[-WALL_THICKNESS:, r] = False
            orifice[-WALL_THICKNESS:, r] = True
            fluid[Nx-1, r] = True
        elif wall == "bottom":
            solid[r, 0:WALL_THICKNESS] = False
            orifice[r, 0:WALL_THICKNESS] = True
            fluid[r, 0] = True
        elif wall == "top":
            solid[r, -WALL_THICKNESS:] = False
            orifice[r, -WALL_THICKNESS:] = True
            fluid[r, Ny-1] = True

    for p in ports:
        carve_port(p, solid, orifice, fluid)

    # Dilated fluid mask for sensitivity filtering near ports
    dilated = F.max_pool2d(
        orifice.float().unsqueeze(0).unsqueeze(0),
        3, stride=1, padding=1
    )[0,0].bool()

    # Extract inlet/outlet metadata
    inlet_ports  = [p for p in ports if p["type"] == "inlet"]
    outlet_ports = [p for p in ports if p["type"] == "outlet"]

    inlet_centers = []
    outlet_centers = []
    inlet_ranges = []
    outlet_ranges = []

    for p in inlet_ports:
        r = p["range"]
        center = (r.start + r.stop) // 2
        inlet_centers.append(center)
        inlet_ranges.append(r)

    for p in outlet_ports:
        r = p["range"]
        center = (r.start + r.stop) // 2
        outlet_centers.append(center)
        outlet_ranges.append(r)

    # No-slip y-masks for left/right walls (y-dimension boolean masks)
    # These indicate which y-positions have solid adjacent to fluid on the vertical walls
    xL = WALL_THICKNESS
    xR = Nx - WALL_THICKNESS - 1
    left_noslip_ymask = solid[xL-1, :]   # solid cell just left of fluid at x=xL
    right_noslip_ymask = solid[xR+1, :]  # solid cell just right of fluid at x=xR

    return {
        "inlet_centers":      inlet_centers,
        "outlet_centers":     outlet_centers,
        "inlet_ranges":       inlet_ranges,
        "outlet_ranges":      outlet_ranges,
        "solid_mask":         solid,
        "orifice_mask":       orifice,
        "fluid_mask":         fluid,
        "left_noslip_ymask":  left_noslip_ymask,
        "right_noslip_ymask": right_noslip_ymask,
        "fluid_dilated":      dilated,
        "inlet_ports": inlet_ports,
        "outlet_ports": outlet_ports,
    }

'''
# =========================================================
# BUILD MASKS VIA NEW FUNCTION
# =========================================================

n_in  = np.random.choice([1,2])
n_out = np.random.choice([1,2])
ports = sample_ports(n_in,"inlet") + sample_ports(n_out,"outlet")

bc = build_bc_masks(Nx, Ny, WALL_THICKNESS, ports)

inlet_centers      = bc["inlet_centers"]
outlet_centers     = bc["outlet_centers"]
inlet_ranges       = bc["inlet_ranges"]
outlet_ranges      = bc["outlet_ranges"]
solid_mask         = bc["solid_mask"]
orifice_mask       = bc["orifice_mask"]
fluid_mask         = bc["fluid_mask"]
left_noslip_ymask  = bc["left_noslip_ymask"]
right_noslip_ymask = bc["right_noslip_ymask"]
fluid_dilated      = bc["fluid_dilated"]

inlet_ports  = [p for p in ports if p["type"] == "inlet"]
outlet_ports = [p for p in ports if p["type"] == "outlet"]
'''


# =========================================================
# LBM D2Q9 SETUP
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

topology = torch.nn.Parameter(torch.ones(Nx, Ny, device=device))

# =========================================================
# CORE LBM FUNCTIONS
# =========================================================
def equilibrium(rho, u):
    cu   = torch.einsum("ia,xya->xyi", c_float, u)
    usqr = (u ** 2).sum(-1, keepdim=True)
    return rho.unsqueeze(-1) * w * (1 + 3*cu + 4.5*cu**2 - 1.5*usqr)


def streaming(f):
    parts = []
    for i, ci in enumerate(c):
        parts.append(
            torch.roll(f[:, :, i:i+1],
                       shifts=(ci[0].item(), ci[1].item()),
                       dims=(0, 1))
        )
    return torch.cat(parts, dim=-1)

def bounce_back_port_tunnels(f):
    """
    Bounce-back inside wall thickness around the port tunnels.
    Prevents diagonal leakage into solid at slot corners.
    """
    f = f.clone()

    for p in ports:
        r = p["range"]
        wall = p["wall"]

        if wall in ["left","right"]:
            x_slice = slice(1, WALL_THICKNESS) if wall=="left" else slice(Nx-WALL_THICKNESS, Nx-1)
            y0, y1 = r.start, r.stop-1

            # bottom edge of tunnel
            f[x_slice, y0, 2] = f[x_slice, y0, 4]
            f[x_slice, y0, 5] = f[x_slice, y0, 7]
            f[x_slice, y0, 6] = f[x_slice, y0, 8]

            # top edge of tunnel
            f[x_slice, y1, 4] = f[x_slice, y1, 2]
            f[x_slice, y1, 7] = f[x_slice, y1, 5]
            f[x_slice, y1, 8] = f[x_slice, y1, 6]

        else:  # bottom/top ports
            y_slice = slice(1, WALL_THICKNESS) if wall=="bottom" else slice(Ny-WALL_THICKNESS, Ny-1)
            x0, x1 = r.start, r.stop-1

            # left edge of tunnel
            f[x0, y_slice, 1] = f[x0, y_slice, 3]
            f[x0, y_slice, 5] = f[x0, y_slice, 7]
            f[x0, y_slice, 8] = f[x0, y_slice, 6]

            # right edge of tunnel
            f[x1, y_slice, 3] = f[x1, y_slice, 1]
            f[x1, y_slice, 6] = f[x1, y_slice, 8]
            f[x1, y_slice, 7] = f[x1, y_slice, 5]

    return f


# =========================================================
# GENERIC BOUNDARY CONDITIONS (all walls)
# =========================================================
def bounce_back_walls(f):
    """
    Bounce-back at the SOLID–FLUID INTERFACE (not outer boundary).
    Skips port openings automatically.
    """
    f = f.clone()

    # first fluid cell next to each wall
    xL = WALL_THICKNESS
    xR = Nx - WALL_THICKNESS - 1
    yB = WALL_THICKNESS
    yT = Ny - WALL_THICKNESS - 1

    # --- bottom wall ---
    mask = solid_mask[:, yB-1]   # solid cell just below fluid
    f[mask, yB, 2] = f[mask, yB, 4]
    f[mask, yB, 5] = f[mask, yB, 7]
    f[mask, yB, 6] = f[mask, yB, 8]

    # --- top wall ---
    mask = solid_mask[:, yT+1]
    f[mask, yT, 4] = f[mask, yT, 2]
    f[mask, yT, 7] = f[mask, yT, 5]
    f[mask, yT, 8] = f[mask, yT, 6]

    # --- left wall ---
    mask = solid_mask[xL-1, :]
    f[xL, mask, 1] = f[xL, mask, 3]
    f[xL, mask, 5] = f[xL, mask, 7]
    f[xL, mask, 8] = f[xL, mask, 6]

    # --- right wall ---
    mask = solid_mask[xR+1, :]
    f[xR, mask, 3] = f[xR, mask, 1]
    f[xR, mask, 6] = f[xR, mask, 8]
    f[xR, mask, 7] = f[xR, mask, 5]

    return f


# =========================================================
# LBM STEP
# =========================================================
def lbm_step(f, density, alpha_max, t):
    rho = f.sum(-1).clamp(min=0.5, max=10.0)
    u   = torch.einsum("xyi,ia->xya", f, c_float) / rho.unsqueeze(-1)
    u   = torch.clamp(u, min=-0.3, max=0.3)

    alpha  = alpha_max * (1.0 - density)
    u_eq   = u / (1.0 + alpha.unsqueeze(-1))
    feq    = equilibrium(rho, u_eq)
    omega_eff = omega * density + 1.0 * (1.0 - density)
    f = f - omega_eff.unsqueeze(-1) * (f - feq)
    f = streaming(f)

    current_u = 0.05 * min(t / 500.0, 1.0)
    f = apply_bcs(f, u_in=current_u)
    return f

def sample_port_pressure(rho, port):
    r = port["range"]
    wall = port["wall"]

    if wall == "left":
        return rho[WALL_THICKNESS, r].mean()
    if wall == "right":
        return rho[Nx - WALL_THICKNESS - 1, r].mean()
    if wall == "bottom":
        return rho[r, WALL_THICKNESS].mean()
    if wall == "top":
        return rho[r, Ny - WALL_THICKNESS - 1].mean()
    return torch.tensor(0.0, device=device)
    
# =========================================================
# SIMULATION
# =========================================================
def simulate(alpha_max):
    f = torch.ones(Nx, Ny, 9, device=device) * w

    density = topology.clamp(0.0, 1.0)
    density = torch.where(solid_mask, torch.zeros_like(density), density)
    density = torch.where(fluid_mask, torch.ones_like(density),  density)

    with torch.no_grad():
        for t in tqdm(range(timesteps_no_grad), desc="warm-up", leave=False):
            f = lbm_step(f, density, alpha_max, t)

    p_in_accum  = 0.0
    p_out_accum = 0.0

    for t in tqdm(range(timesteps_no_grad, timesteps_no_grad + timesteps_grad),
                  desc="grad", leave=False):
        f = lbm_step(f, density, alpha_max, t)
        rho_t = f.sum(-1)
        p_in  = torch.stack([sample_port_pressure(rho_t, p) for p in inlet_ports]).mean()
        p_out = torch.stack([sample_port_pressure(rho_t, p) for p in outlet_ports]).mean()
        p_in_accum  = p_in_accum  + p_in
        p_out_accum = p_out_accum + p_out

    cs2 = 1.0 / 3.0
    return (p_in_accum - p_out_accum) / timesteps_grad * cs2, density

def apply_inlet(f, port, u_in):
    r = port["range"]
    wall = port["wall"]

    if wall == "left":
        rho = (f[0,r,0]+f[0,r,2]+f[0,r,4]+2*(f[0,r,3]+f[0,r,6]+f[0,r,7]))/(1-u_in)
        f[0,r,1] = f[0,r,3] + (2/3)*rho*u_in
        f[0,r,5] = f[0,r,7] - 0.5*(f[0,r,2]-f[0,r,4]) + (1/6)*rho*u_in
        f[0,r,8] = f[0,r,6] + 0.5*(f[0,r,2]-f[0,r,4]) + (1/6)*rho*u_in

    elif wall == "right":
        rho = (f[-1,r,0]+f[-1,r,2]+f[-1,r,4]+2*(f[-1,r,1]+f[-1,r,5]+f[-1,r,8])) / (1 - u_in)
        f[-1,r,3] = f[-1,r,1] + (2/3)*rho*u_in
        f[-1,r,7] = f[-1,r,5] + 0.5*(f[-1,r,2]-f[-1,r,4]) + (1/6)*rho*u_in
        f[-1,r,6] = f[-1,r,8] - 0.5*(f[-1,r,2]-f[-1,r,4]) + (1/6)*rho*u_in

    elif wall == "bottom":
        rho = (f[r,0,0]+f[r,0,1]+f[r,0,3]+2*(f[r,0,4]+f[r,0,7]+f[r,0,8]))/(1-u_in)
        f[r,0,2] = f[r,0,4] + (2/3)*rho*u_in
        f[r,0,5] = f[r,0,7] - 0.5*(f[r,0,1]-f[r,0,3]) + (1/6)*rho*u_in
        f[r,0,6] = f[r,0,8] + 0.5*(f[r,0,1]-f[r,0,3]) + (1/6)*rho*u_in

    elif wall == "top":
        rho = (f[r,-1,0]+f[r,-1,1]+f[r,-1,3]+2*(f[r,-1,2]+f[r,-1,5]+f[r,-1,6])) / (1 - u_in)
        f[r,-1,4] = f[r,-1,2] + (2/3)*rho*u_in
        f[r,-1,7] = f[r,-1,5] + 0.5*(f[r,-1,1]-f[r,-1,3]) + (1/6)*rho*u_in
        f[r,-1,8] = f[r,-1,6] - 0.5*(f[r,-1,1]-f[r,-1,3]) + (1/6)*rho*u_in

def apply_outlet(f, port):
    r = port["range"]
    wall = port["wall"]
    rho = 1.0

    if wall == "left":
        u = 1 - (f[0,r,0]+f[0,r,2]+f[0,r,4]+2*(f[0,r,3]+f[0,r,6]+f[0,r,7]))/rho
        f[0,r,1] = f[0,r,3] + (2/3)*rho*u
        f[0,r,5] = f[0,r,7] - 0.5*(f[0,r,2]-f[0,r,4]) + (1/6)*rho*u
        f[0,r,8] = f[0,r,6] + 0.5*(f[0,r,2]-f[0,r,4]) + (1/6)*rho*u

    elif wall == "right":
        u = -1 + (f[-1,r,0]+f[-1,r,2]+f[-1,r,4]+2*(f[-1,r,1]+f[-1,r,5]+f[-1,r,8]))/rho
        f[-1,r,3] = f[-1,r,1] - (2/3)*rho*u
        f[-1,r,7] = f[-1,r,5] + 0.5*(f[-1,r,2]-f[-1,r,4]) - (1/6)*rho*u
        f[-1,r,6] = f[-1,r,8] - 0.5*(f[-1,r,2]-f[-1,r,4]) - (1/6)*rho*u

    elif wall == "bottom":
        u = 1 - (f[r,0,0]+f[r,0,1]+f[r,0,3]+2*(f[r,0,4]+f[r,0,7]+f[r,0,8]))/rho
        f[r,0,2] = f[r,0,4] + (2/3)*rho*u
        f[r,0,5] = f[r,0,7] - 0.5*(f[r,0,1]-f[r,0,3]) + (1/6)*rho*u
        f[r,0,6] = f[r,0,8] + 0.5*(f[r,0,1]-f[r,0,3]) + (1/6)*rho*u

    elif wall == "top":
        u = -1 + (f[r,-1,0]+f[r,-1,1]+f[r,-1,3]+2*(f[r,-1,2]+f[r,-1,5]+f[r,-1,6]))/rho
        f[r,-1,4] = f[r,-1,2] - (2/3)*rho*u
        f[r,-1,7] = f[r,-1,5] + 0.5*(f[r,-1,1]-f[r,-1,3]) - (1/6)*rho*u
        f[r,-1,8] = f[r,-1,6] - 0.5*(f[r,-1,1]-f[r,-1,3]) - (1/6)*rho*u

def apply_bcs(f, u_in):
    f = bounce_back_walls(f)
    f = bounce_back_port_tunnels(f)

    for p in inlet_ports:
        apply_inlet(f, p, u_in)

    for p in outlet_ports:
        apply_outlet(f, p)

    return f

# =========================================================
# FEASIBILITY
# =========================================================

def port_cells(port):
    r = port["range"]
    wall = port["wall"]
    if wall=="left":   return [(WALL_THICKNESS, y) for y in range(r.start,r.stop)]
    if wall=="right":  return [(Nx-WALL_THICKNESS-1, y) for y in range(r.start,r.stop)]
    if wall=="bottom": return [(x, WALL_THICKNESS) for x in range(r.start,r.stop)]
    if wall=="top":    return [(x, Ny-WALL_THICKNESS-1) for x in range(r.start,r.stop)]


def bfs(binary, starts):
    visited = set(starts)
    queue = deque(starts)
    while queue:
        x,y = queue.popleft()
        for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx,ny = x+dx, y+dy
            if 0<=nx<Nx and 0<=ny<Ny and binary[nx,ny] and (nx,ny) not in visited:
                visited.add((nx,ny))
                queue.append((nx,ny))
    return visited


def check_connectivity(density_np, masks):
    binary = (density_np > 0.5).astype(np.uint8)

    outlet_cells = set()
    for p in masks["outlet_ports"]:
        outlet_cells |= set(port_cells(p))

    # 🔴 Every inlet must reach an outlet
    for inlet in masks["inlet_ports"]:
        starts = port_cells(inlet)
        visited = bfs(binary, starts)

        if not any(cell in outlet_cells for cell in visited):
            return False  # this inlet is dead

    return True


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

def remove_floating_fluid(density):
    binary = (density.detach().cpu().numpy() > 0.5).astype(np.uint8)

    inlet_cells = []
    for p in inlet_ports:
        inlet_cells += port_cells(p)

    connected = bfs(binary, inlet_cells)

    mask = torch.zeros_like(density, dtype=torch.bool)
    for x,y in connected:
        mask[x,y] = True

    # remove floating blobs
    density.data[~mask & (density>0.5)] = 0.0

def remove_dead_branches(density):
    binary = (density.detach().cpu().numpy() > 0.5).astype(np.uint8)

    outlet_cells = []
    for p in outlet_ports:
        outlet_cells += port_cells(p)

    connected = bfs(binary, outlet_cells)

    mask = torch.zeros_like(density, dtype=torch.bool)
    for x,y in connected:
        mask[x,y] = True

    density.data[~mask & (density>0.5)] = 0.0

# =========================================================
# VISUALIZATION
# =========================================================
def save_density_plot(density_np, masks, filepath, title=None):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(density_np.T, origin="lower", cmap="gray_r",
              vmin=0, vmax=1, interpolation="none")

    inlet_colors  = ["#2ecc71", "#1a8a4a"]
    outlet_colors = ["#e74c3c", "#922b21"]

    for p in masks["inlet_ports"]:
        r = p["range"]
        if p["wall"]=="left":   ax.plot([0,0],[r.start,r.stop],lw=3,color="green")
        if p["wall"]=="right":  ax.plot([Nx-1,Nx-1],[r.start,r.stop],lw=3,color="green")
        if p["wall"]=="bottom": ax.plot([r.start,r.stop],[0,0],lw=3,color="green")
        if p["wall"]=="top":    ax.plot([r.start,r.stop],[Ny-1,Ny-1],lw=3,color="green")

    for p in masks["outlet_ports"]:
        r = p["range"]
        if p["wall"]=="left":   ax.plot([0,0],[r.start,r.stop],lw=3,color="red")
        if p["wall"]=="right":  ax.plot([Nx-1,Nx-1],[r.start,r.stop],lw=3,color="red")
        if p["wall"]=="bottom": ax.plot([r.start,r.stop],[0,0],lw=3,color="red")
        if p["wall"]=="top":    ax.plot([r.start,r.stop],[Ny-1,Ny-1],lw=3,color="red")

    if title:
        ax.set_title(title, fontsize=7)
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_dataset_file(filepath, densities, pressure_drops, volumes, bc_info, dataset_type):
    if not densities:
        print(f"  Skipping {os.path.basename(filepath)} — no data.")
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with h5py.File(filepath, "w") as hf:
        hf.create_dataset("density",         data=torch.stack(densities).numpy(), compression="gzip")
        hf.create_dataset("pressure_drop",   data=np.array(pressure_drops, dtype=np.float32))
        hf.create_dataset("volume_fraction", data=np.array(volumes,        dtype=np.float32))
        hf.create_dataset("metadata",
            data=[json.dumps(b).encode("utf-8") for b in bc_info])
        hf.create_dataset("eso_iteration",   data=np.array([b["iteration"]       for b in bc_info], dtype=np.int32))
        hf.create_dataset("optimization_id", data=np.array([b["opt_id"]          for b in bc_info], dtype=np.int32))
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
# DATASET / RUN PARAMETERS  ← edit these
# =========================================================
NUM_DESIGNS_TARGET = 10000   # overall goal across all runs
BATCH_SIZE         = 50     # designs to collect per script execution
VOLUME_THRESHOLD   = 0.2    # upper bound for random target volume
CHECKPOINT_FILE    = "./data/new/generation_checkpoint.json"
OUTPUT_DIR         = "./data/new"

SAVE_FINAL_ONLY   = True
SAVE_ALL_FEASIBLE = True

ALPHA_MAX        = 100.0
MAX_ESO_ITERS    = 30

def main():
    """
    Main dataset generation loop: samples BC configs, runs ESO optimization,
    collects feasible designs, and saves to HDF5 files.
    """
    global topology
    global solid_mask, fluid_mask, orifice_mask, fluid_dilated
    global inlet_ports, outlet_ports
    global ports

    # =====================================================================
    # LOAD CHECKPOINT & INITIALIZATION
    # =====================================================================
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

    # Batch accumulators for HDF5 export
    final_d,  final_dp,  final_vol,  final_bc  = [], [], [], []
    all_d,    all_dp,    all_vol,    all_bc     = [], [], [], []

    completed = 0
    attempts  = 0

    # =====================================================================
    # MAIN SAMPLING & OPTIMIZATION LOOP
    # =====================================================================
    while (completed < BATCH_SIZE
           and start_count + completed < NUM_DESIGNS_TARGET
           and attempts < BATCH_SIZE * 10):

        attempts += 1

        # Sample a new BC configuration (inlet/outlet center lists)
        n_in  = np.random.choice([1,2])
        n_out = np.random.choice([1,2])
        ports = sample_ports(n_in,"inlet") + sample_ports(n_out,"outlet")
        
        overlap_found = False
        for i in range(len(ports)):
            for j in range(i + 1, len(ports)):
                if ports_overlap(ports[i], ports[j], gap=5):
                    overlap_found = True
                    break
            if overlap_found:
                break

        if overlap_found:
            print(f"  attempt {attempts:3d} | port overlap detected — resampling")
            continue


        bc    = build_bc_masks(Nx,Ny,WALL_THICKNESS,ports)
        masks = bc


        solid_mask    = masks["solid_mask"]
        fluid_mask    = masks["fluid_mask"]
        orifice_mask  = masks["orifice_mask"]
        fluid_dilated = masks["fluid_dilated"]
        inlet_ports   = masks["inlet_ports"]
        outlet_ports  = masks["outlet_ports"]

        is_designable = (~masks["solid_mask"]
                         & ~masks["fluid_mask"]
                         & ~masks["orifice_mask"])

        # Fresh topology parameter + randomized volume target
        topology   = torch.nn.Parameter(torch.ones(Nx, Ny, device=device))
        target_vol = float(np.random.uniform(0.20, VOLUME_THRESHOLD))
        def ports_to_strings(port_list):
            return [f"{p['wall']}_{p['center']}" for p in port_list]

        in_desc  = ports_to_strings([p for p in ports if p["type"]=="inlet"])
        out_desc = ports_to_strings([p for p in ports if p["type"]=="outlet"])
        bc_key = (tuple(sorted(in_desc)), tuple(sorted(out_desc)))
        if bc_key in used_bc_configs:
            print(f"  Skipping already used BC config: in={in_desc} out={out_desc}")
            continue

        # Track best feasible snapshot within this optimization run
        best_density = None
        best_dp_val  = float("inf")
        best_vol_val = 1.0
        best_iter    = 0


        print(f"\n  attempt {attempts:3d} | inlets={in_desc} outlets={out_desc} "
              f"| target_vol={target_vol:.2f}")

        # -------------------------------------------------------------
        # ESO OPTIMIZATION LOOP
        # -------------------------------------------------------------
        for it in range(MAX_ESO_ITERS):
            #print(it)
            if topology.grad is not None:
                topology.grad.zero_()

            # Forward simulation + sensitivity
            dp, density = simulate(ALPHA_MAX)
            vol_val = topology[is_designable].mean().item()
            dp_val  = dp.item()
            dp.backward()

            # Feasibility check
            feasible, reason = is_feasible(
                density, dp_val, vol_val, masks,
                vol_min=0.10, vol_max=0.30,
                dp_min=0.005, dp_max=5.0
            )

            # Update best feasible snapshot
            if feasible and dp_val < best_dp_val:
                best_density = density.detach().cpu().clone()
                best_dp_val  = dp_val
                best_vol_val = vol_val
                best_iter    = it

            # Metadata for logging/export
            bc_meta = {
                "inlets":   in_desc,
                "outlets":  out_desc,
                "iteration":       it,
                "opt_id":          global_opt_id,
                "feasibility":     reason,
            }

            # Save to "all feasible" dataset (file 3)
            if SAVE_ALL_FEASIBLE and it >= 15 and feasible:
                all_d.append(density.detach().cpu().clone())
                all_dp.append(dp_val)
                all_vol.append(vol_val)
                all_bc.append(bc_meta.copy())

            # ---------------------------------------------------------
            # ESO REMOVAL STEP
            # ---------------------------------------------------------
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
                        # check every outlet before committing removal
                        sealed = False
                        for p in outlet_ports:
                            r = p["range"]
                            wall = p["wall"]
                            if wall == "right":
                                guard = rm[Nx-WALL_THICKNESS-1:Nx, r]
                            elif wall == "left":
                                guard = rm[0:WALL_THICKNESS+1, r]
                            elif wall == "bottom":
                                guard = rm[r, 0:WALL_THICKNESS+1]
                            elif wall == "top":
                                guard = rm[r, Ny-WALL_THICKNESS-1:Ny]
                            if guard.all():
                                sealed = True
                                break

                        if not sealed:
                            topology.data[rm] = 0.0
                            remove_floating_fluid(topology)
                            remove_dead_branches(topology)
            else:
                # Volume target reached; early exit if we have a best candidate
                if best_density is not None and it >= 25:
                    break

        # =================================================================
        # RECORD FINAL BEST DESIGN FROM THIS RUN
        # =================================================================
        if best_density is not None:
            bc_final = {
                "inlet_centers":   in_desc,
                "outlet_centers":  out_desc,
                "num_inlets":      len(in_desc),
                "num_outlets":     len(out_desc),
                "iteration":       best_iter,
                "opt_id":          global_opt_id,
                "feasibility":     "best_from_run",
            }

            # Save to "final only" dataset (file 1)
            if SAVE_FINAL_ONLY:
                final_d.append(best_density)
                final_dp.append(best_dp_val)
                final_vol.append(best_vol_val)
                final_bc.append(bc_final)

            # Save PNG visualization
            os.makedirs("./output/plots", exist_ok=True)
            plot_path = (
                f"./output/plots/"
                f"opt{global_opt_id:04d}"
                f"_in{'_'.join(map(str, in_desc))}"
                f"_out{'_'.join(map(str, out_desc))}"
                f"_dp{best_dp_val:.3f}.png"
            )
            save_density_plot(
                best_density.numpy(), masks, plot_path,
                title=(f"opt#{global_opt_id} | in={in_desc} out={out_desc} | "
                       f"Δp={best_dp_val:.3f} vol={best_vol_val:.2f}")
            )

            # Update checkpoint tracking
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
                  f"inlets={in_desc} outlets={out_desc}")
        else:
            print(f"  ✗ no feasible design — inlets={in_desc} outlets={out_desc}")

    # =====================================================================
    # SAVE HDF5 DATASETS
    # =====================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fp_final = f"{OUTPUT_DIR}/dataset_final_run{run_number}.h5"
    fp_all   = f"{OUTPUT_DIR}/dataset_all_run{run_number}.h5"

    print("\nSaving HDF5 files...")
    save_dataset_file(fp_final, final_d, final_dp, final_vol, final_bc, "final_designs_only")
    save_dataset_file(fp_all,   all_d,   all_dp,   all_vol,   all_bc,   "all_feasible_designs")

    # =====================================================================
    # SUMMARY PRINTOUT
    # =====================================================================
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"  File 1 (final only):    {len(final_d):5d} designs  →  {fp_final}")
    print(f"  File 3 (all feasible):  {len(all_d):5d} designs  →  {fp_all}")
    print(f"  Attempts this run:      {attempts}")
    print(f"  Completed this run:     {completed}")
    if attempts:
        print(f"  Success rate:           {completed / attempts * 100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()