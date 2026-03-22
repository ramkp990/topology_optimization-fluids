import torch
import torch.nn.functional as F
import numpy as np
import os
from pyevtk.hl import gridToVTK
from tqdm import tqdm
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# Customization Parameters
# ---------------------------------------------------------
WALL_THICKNESS = 4
REMOVAL_FRACTION = 0.4

# ---------------------------------------------------------
# Domain
# ---------------------------------------------------------
Nx = 64
Ny = 64
timesteps_no_grad = 2000
timesteps_grad    = 300

tau   = 0.6
omega = 1.0 / tau

'''
# Inlet: left wall, y=48..53
inlet_center = Ny // 2 + int(0.3 * Nx)
inlet_height = int(0.1 * Nx)
inlet_lo     = inlet_center - inlet_height // 2   # = 48
inlet_hi     = inlet_center + inlet_height // 2   # = 54 (exclusive)
inlet_range  = slice(inlet_lo, inlet_hi)

# Outlet: right wall, y=10..15
outlet_center = Ny // 2 - int(0.3 * Ny)
outlet_height = int(0.1 * Ny)
outlet_lo     = outlet_center - outlet_height // 2  # = 10
outlet_hi     = outlet_center + outlet_height // 2  # = 16 (exclusive)
outlet_range  = slice(outlet_lo, outlet_hi)

# ---------------------------------------------------------
# BC interface indices — all at first/last FLUID cell
# ---------------------------------------------------------
LEFT_NOSLIP_X  = WALL_THICKNESS            # = 4
RIGHT_NOSLIP_X = Nx - WALL_THICKNESS - 1   # = 59
BOT_NOSLIP_Y   = WALL_THICKNESS            # = 4
TOP_NOSLIP_Y   = Ny - WALL_THICKNESS - 1   # = 59

# Orifice slot edge rows
INLET_SLOT_BOT_Y  = inlet_lo       # = 48
INLET_SLOT_TOP_Y  = inlet_hi - 1   # = 53
OUTLET_SLOT_BOT_Y = outlet_lo      # = 10
OUTLET_SLOT_TOP_Y = outlet_hi - 1  # = 15

# Interior orifice x columns (not the Zou-He cell)
inlet_interior_x  = slice(1, WALL_THICKNESS)            # x=1,2,3
outlet_interior_x = slice(Nx - WALL_THICKNESS, Nx - 1)  # x=60,61,62

# Pressure probes at first/last fluid cell
P_IN_X  = LEFT_NOSLIP_X   # = 4
P_OUT_X = RIGHT_NOSLIP_X  # = 59

# ---------------------------------------------------------
# Solid mask — wall bands with orifices cleared
# ---------------------------------------------------------
solid_mask = torch.zeros(Nx, Ny, device=device, dtype=torch.bool)
solid_mask[0:WALL_THICKNESS, :]  = True
solid_mask[-WALL_THICKNESS:, :]  = True
solid_mask[:, 0:WALL_THICKNESS]  = True
solid_mask[:, -WALL_THICKNESS:]  = True
solid_mask[0:WALL_THICKNESS,  inlet_range]  = False
solid_mask[-WALL_THICKNESS:,  outlet_range] = False

# Orifice mask: all cells in the orifice slots (solid_mask=False there)
orifice_mask = torch.zeros(Nx, Ny, device=device, dtype=torch.bool)
orifice_mask[0:WALL_THICKNESS,  inlet_range]  = True
orifice_mask[-WALL_THICKNESS:,  outlet_range] = True

# ---------------------------------------------------------
# Forced-fluid mask — only the Zou-He BC cells
# ---------------------------------------------------------
fluid_mask = torch.zeros(Nx, Ny, device=device, dtype=torch.bool)
fluid_mask[0,    inlet_range]  = True
fluid_mask[Nx-1, outlet_range] = True

# ---------------------------------------------------------
# y-masks for left/right no-slip (skip orifice rows)
# ---------------------------------------------------------
left_noslip_ymask = torch.ones(Ny, dtype=torch.bool, device=device)
left_noslip_ymask[inlet_range] = False

right_noslip_ymask = torch.ones(Ny, dtype=torch.bool, device=device)
right_noslip_ymask[outlet_range] = False

# ---------------------------------------------------------
# Adjacency-protection mask for ESO
# Dilate the full orifice mask (not just the Zou-He cell) so
# the optimizer never removes a cell immediately adjacent to
# any part of the orifice slot.
# ---------------------------------------------------------
fluid_dilated = F.max_pool2d(
    orifice_mask.float().unsqueeze(0).unsqueeze(0),
    kernel_size=3, stride=1, padding=1
)[0, 0].bool()
'''
# ---------------------------------------------------------
# LBM D2Q9 setup
# ---------------------------------------------------------
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

topology      = torch.nn.Parameter(torch.ones(Nx, Ny, device=device))
target_volume = 0.2

# ---------------------------------------------------------
# Core LBM functions
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# [ADD] Function to create masks with variable inlet/outlet Y positions
# ---------------------------------------------------------
def create_masks_for_bc(inlet_y_center, outlet_y_center, 
                        inlet_height=None, outlet_height=None):
    """
    Create all masks for given inlet/outlet center positions.
    Height defaults to original if not specified.
    """
    if inlet_height is None:
        inlet_height = int(0.1 * Nx)
    if outlet_height is None:
        outlet_height = int(0.1 * Ny)
    
    # Compute ranges
    inlet_lo = max(WALL_THICKNESS, inlet_y_center - inlet_height // 2)
    inlet_hi = min(Ny - WALL_THICKNESS, inlet_y_center + inlet_height // 2)
    inlet_range = slice(inlet_lo, inlet_hi)
    
    outlet_lo = max(WALL_THICKNESS, outlet_y_center - outlet_height // 2)
    outlet_hi = min(Ny - WALL_THICKNESS, outlet_y_center + outlet_height // 2)
    outlet_range = slice(outlet_lo, outlet_hi)
    
    # BC indices
    LEFT_NOSLIP_X = WALL_THICKNESS
    RIGHT_NOSLIP_X = Nx - WALL_THICKNESS - 1
    BOT_NOSLIP_Y = WALL_THICKNESS
    TOP_NOSLIP_Y = Ny - WALL_THICKNESS - 1
    INLET_SLOT_BOT_Y = inlet_lo
    INLET_SLOT_TOP_Y = inlet_hi - 1
    OUTLET_SLOT_BOT_Y = outlet_lo
    OUTLET_SLOT_TOP_Y = outlet_hi - 1
    inlet_interior_x = slice(1, WALL_THICKNESS)
    outlet_interior_x = slice(Nx - WALL_THICKNESS, Nx - 1)
    P_IN_X = LEFT_NOSLIP_X
    P_OUT_X = RIGHT_NOSLIP_X
    
    # Masks
    solid_mask = torch.zeros(Nx, Ny, device=device, dtype=torch.bool)
    solid_mask[0:WALL_THICKNESS, :] = True
    solid_mask[-WALL_THICKNESS:, :] = True
    solid_mask[:, 0:WALL_THICKNESS] = True
    solid_mask[:, -WALL_THICKNESS:] = True
    solid_mask[0:WALL_THICKNESS, inlet_range] = False
    solid_mask[-WALL_THICKNESS:, outlet_range] = False
    
    orifice_mask = torch.zeros(Nx, Ny, device=device, dtype=torch.bool)
    orifice_mask[0:WALL_THICKNESS, inlet_range] = True
    orifice_mask[-WALL_THICKNESS:, outlet_range] = True
    
    fluid_mask = torch.zeros(Nx, Ny, device=device, dtype=torch.bool)
    fluid_mask[0, inlet_range] = True
    fluid_mask[Nx-1, outlet_range] = True
    
    left_noslip_ymask = torch.ones(Ny, dtype=torch.bool, device=device)
    left_noslip_ymask[inlet_range] = False
    right_noslip_ymask = torch.ones(Ny, dtype=torch.bool, device=device)
    right_noslip_ymask[outlet_range] = False
    
    fluid_dilated = F.max_pool2d(
        orifice_mask.float().unsqueeze(0).unsqueeze(0),
        kernel_size=3, stride=1, padding=1
    )[0, 0].bool()
    
    return {
        'inlet_range': inlet_range, 'outlet_range': outlet_range,
        'LEFT_NOSLIP_X': LEFT_NOSLIP_X, 'RIGHT_NOSLIP_X': RIGHT_NOSLIP_X,
        'BOT_NOSLIP_Y': BOT_NOSLIP_Y, 'TOP_NOSLIP_Y': TOP_NOSLIP_Y,
        'INLET_SLOT_BOT_Y': INLET_SLOT_BOT_Y, 'INLET_SLOT_TOP_Y': INLET_SLOT_TOP_Y,
        'OUTLET_SLOT_BOT_Y': OUTLET_SLOT_BOT_Y, 'OUTLET_SLOT_TOP_Y': OUTLET_SLOT_TOP_Y,
        'inlet_interior_x': inlet_interior_x, 'outlet_interior_x': outlet_interior_x,
        'P_IN_X': P_IN_X, 'P_OUT_X': P_OUT_X,
        'solid_mask': solid_mask, 'orifice_mask': orifice_mask,
        'fluid_mask': fluid_mask, 'left_noslip_ymask': left_noslip_ymask,
        'right_noslip_ymask': right_noslip_ymask, 'fluid_dilated': fluid_dilated,
        'inlet_y_center': inlet_y_center, 'outlet_y_center': outlet_y_center,
    }

# ---------------------------------------------------------
# [MODIFY] apply_bcs to accept masks dict
# ---------------------------------------------------------
def apply_bcs(f, u_in, masks):
    f = f.clone()
    
    # Bottom/top walls (unchanged)
    f[:, masks['BOT_NOSLIP_Y'], 2] = f[:, masks['BOT_NOSLIP_Y'], 4]
    f[:, masks['BOT_NOSLIP_Y'], 5] = f[:, masks['BOT_NOSLIP_Y'], 7]
    f[:, masks['BOT_NOSLIP_Y'], 6] = f[:, masks['BOT_NOSLIP_Y'], 8]
    f[:, masks['TOP_NOSLIP_Y'], 4] = f[:, masks['TOP_NOSLIP_Y'], 2]
    f[:, masks['TOP_NOSLIP_Y'], 7] = f[:, masks['TOP_NOSLIP_Y'], 5]
    f[:, masks['TOP_NOSLIP_Y'], 8] = f[:, masks['TOP_NOSLIP_Y'], 6]
    
    # Left/right no-slip (using masks)
    m = masks['left_noslip_ymask']
    f[masks['LEFT_NOSLIP_X'], m, 1] = f[masks['LEFT_NOSLIP_X'], m, 3]
    f[masks['LEFT_NOSLIP_X'], m, 5] = f[masks['LEFT_NOSLIP_X'], m, 7]
    f[masks['LEFT_NOSLIP_X'], m, 8] = f[masks['LEFT_NOSLIP_X'], m, 6]
    
    m = masks['right_noslip_ymask']
    f[masks['RIGHT_NOSLIP_X'], m, 3] = f[masks['RIGHT_NOSLIP_X'], m, 1]
    f[masks['RIGHT_NOSLIP_X'], m, 6] = f[masks['RIGHT_NOSLIP_X'], m, 8]
    f[masks['RIGHT_NOSLIP_X'], m, 7] = f[masks['RIGHT_NOSLIP_X'], m, 5]
    
    # Inlet slot edges
    xi = masks['inlet_interior_x']
    f[xi, masks['INLET_SLOT_BOT_Y'], 2] = f[xi, masks['INLET_SLOT_BOT_Y'], 4]
    f[xi, masks['INLET_SLOT_BOT_Y'], 5] = f[xi, masks['INLET_SLOT_BOT_Y'], 7]
    f[xi, masks['INLET_SLOT_BOT_Y'], 6] = f[xi, masks['INLET_SLOT_BOT_Y'], 8]
    f[xi, masks['INLET_SLOT_TOP_Y'], 4] = f[xi, masks['INLET_SLOT_TOP_Y'], 2]
    f[xi, masks['INLET_SLOT_TOP_Y'], 7] = f[xi, masks['INLET_SLOT_TOP_Y'], 5]
    f[xi, masks['INLET_SLOT_TOP_Y'], 8] = f[xi, masks['INLET_SLOT_TOP_Y'], 6]
    
    # Outlet slot edges
    xo = masks['outlet_interior_x']
    f[xo, masks['OUTLET_SLOT_BOT_Y'], 2] = f[xo, masks['OUTLET_SLOT_BOT_Y'], 4]
    f[xo, masks['OUTLET_SLOT_BOT_Y'], 5] = f[xo, masks['OUTLET_SLOT_BOT_Y'], 7]
    f[xo, masks['OUTLET_SLOT_BOT_Y'], 6] = f[xo, masks['OUTLET_SLOT_BOT_Y'], 8]
    f[xo, masks['OUTLET_SLOT_TOP_Y'], 4] = f[xo, masks['OUTLET_SLOT_TOP_Y'], 2]
    f[xo, masks['OUTLET_SLOT_TOP_Y'], 7] = f[xo, masks['OUTLET_SLOT_TOP_Y'], 5]
    f[xo, masks['OUTLET_SLOT_TOP_Y'], 8] = f[xo, masks['OUTLET_SLOT_TOP_Y'], 6]
    
    # Inlet Zou-He
    y = masks['inlet_range']
    rho_in = (f[0, y, 0] + f[0, y, 2] + f[0, y, 4] + 2.0*(f[0, y, 3]+f[0, y, 6]+f[0, y, 7])) / (1.0 - u_in)
    f[0, y, 1] = f[0, y, 3] + (2.0/3.0)*rho_in*u_in
    f[0, y, 5] = f[0, y, 7] - 0.5*(f[0, y, 2]-f[0, y, 4]) + (1.0/6.0)*rho_in*u_in
    f[0, y, 8] = f[0, y, 6] + 0.5*(f[0, y, 2]-f[0, y, 4]) + (1.0/6.0)*rho_in*u_in
    
    # Outlet Zou-He
    y = masks['outlet_range']
    rho_out = 1.0
    u_out = -1.0 + (f[-1, y, 0]+f[-1, y, 2]+f[-1, y, 4] + 2.0*(f[-1, y, 1]+f[-1, y, 5]+f[-1, y, 8])) / rho_out
    f[-1, y, 3] = f[-1, y, 1] - (2.0/3.0)*rho_out*u_out
    f[-1, y, 7] = f[-1, y, 5] + 0.5*(f[-1, y, 2]-f[-1, y, 4]) - (1.0/6.0)*rho_out*u_out
    f[-1, y, 6] = f[-1, y, 8] - 0.5*(f[-1, y, 2]-f[-1, y, 4]) - (1.0/6.0)*rho_out*u_out
    
    return f

def filter_density(density):
    return F.avg_pool2d(
        density.unsqueeze(0).unsqueeze(0),
        kernel_size=3, stride=1, padding=1
    )[0, 0]


def write_vtk(iteration, rho, u, density):
    """
    Write VTK with physically correct values in every zone:
      solid wall cells  : density=0, ux=0, uy=0
      orifice slot cells: density=1, ux/uy from LBM
      fluid interior    : density=topology (filtered), ux/uy from LBM
    """
    rho_np     = rho.cpu().detach().numpy().copy()
    ux_np      = u[:, :, 0].cpu().detach().numpy().copy()
    uy_np      = u[:, :, 1].cpu().detach().numpy().copy()
    density_np = density.cpu().detach().numpy().copy()

    solid_np   = solid_mask.cpu().numpy()
    orifice_np = orifice_mask.cpu().numpy()

    # Solid wall cells: zero everything
    rho_np[solid_np]     = 0.0
    ux_np[solid_np]      = 0.0
    uy_np[solid_np]      = 0.0
    density_np[solid_np] = 0.0

    # Orifice slot cells: density=1 (they are always full fluid)
    density_np[orifice_np] = 1.0

    os.makedirs("./output", exist_ok=True)
    gridToVTK(
        f"./output/design_{iteration}",
        np.arange(Nx + 1, dtype=np.float64),
        np.arange(Ny + 1, dtype=np.float64),
        np.arange(2,       dtype=np.float64),
        cellData={
            "rho":     np.ascontiguousarray(rho_np[:, :, None]),
            "density": np.ascontiguousarray(density_np[:, :, None]),
            "ux":      np.ascontiguousarray(ux_np[:, :, None]),
            "uy":      np.ascontiguousarray(uy_np[:, :, None]),
        }
    )


# ---------------------------------------------------------
# [MODIFY] lbm_step to pass masks to apply_bcs
# ---------------------------------------------------------
def lbm_step(f, density_physics, alpha_max, t, masks):
    rho = f.sum(-1).clamp(min=0.5, max=10.0)
    u = torch.einsum("xyi,ia->xya", f, c_float) / rho.unsqueeze(-1)
    u = torch.clamp(u, min=-0.3, max=0.3)
    
    alpha = alpha_max * (1.0 - density_physics)
    u_eq = u / (1.0 + alpha.unsqueeze(-1))
    feq = equilibrium(rho, u_eq)
    omega_eff = omega * density_physics + 1.0 * (1.0 - density_physics)
    f = f - omega_eff.unsqueeze(-1) * (f - feq)
    f = streaming(f)
    
    current_u = 0.05 * min(t / 500.0, 1.0)
    f = apply_bcs(f, u_in=current_u, masks=masks)  # <-- pass masks
    return f


# ---------------------------------------------------------
# [MODIFY] simulate to accept masks + external density + return intermediates
# ---------------------------------------------------------
def simulate(alpha_max, masks, density_input=None):
    f = torch.ones(Nx, Ny, 9, device=device) * w
    
    # Resolve density
    if density_input is not None:
        density_physics = density_input.clamp(0.0, 1.0)
    else:
        density_physics = topology.clamp(0.0, 1.0)
    
    # Apply fixed masks
    density_physics = torch.where(masks['solid_mask'], torch.zeros_like(density_physics), density_physics)
    density_physics = torch.where(masks['fluid_mask'], torch.ones_like(density_physics), density_physics)
    
    # Warm-up
    with torch.no_grad():
        for t in range(timesteps_no_grad):
            f = lbm_step(f, density_physics, alpha_max, t, masks)
    
    # Gradient phase
    p_in_accum = p_out_accum = 0.0
    for t in range(timesteps_no_grad, timesteps_no_grad + timesteps_grad):
        f = lbm_step(f, density_physics, alpha_max, t, masks)
        rho_t = f.sum(-1)
        p_in_accum += rho_t[masks['P_IN_X'], masks['inlet_range']].mean()
        p_out_accum += rho_t[masks['P_OUT_X'], masks['outlet_range']].mean()
    
    cs2 = 1.0/3.0
    mean_pressure_drop = (p_in_accum - p_out_accum) / timesteps_grad * cs2
    
    # Final fields
    rho = f.sum(-1).clamp(min=0.5, max=10.0)
    u = torch.einsum("xyi,ia->xya", f, c_float) / rho.unsqueeze(-1)
    density_raw = density_input.clamp(0,1) if density_input is not None else topology.clamp(0,1)
    density_filtered = filter_density(density_raw)
    density_filtered = torch.where(masks['solid_mask'], torch.zeros_like(density_filtered), density_filtered)
    density_filtered = torch.where(masks['fluid_mask'], torch.ones_like(density_filtered), density_filtered)
    
    return mean_pressure_drop, rho, u, density_filtered

# ---------------------------------------------------------
# Feasibility Check Function (Phase 1 Criteria)
# ---------------------------------------------------------
def is_feasible(density, pressure_drop, volume_fraction, masks,
                vol_min=0.10, vol_max=0.40,
                dp_min=0.01, dp_max=5.0):
    """
    Check if a design is physically feasible before saving to dataset.
    
    Args:
        density: Density field [Nx, Ny] (can be on GPU)
        pressure_drop: Scalar pressure drop value
        volume_fraction: Fluid volume fraction (0-1)
        masks: BC masks dictionary from create_masks_for_bc()
        vol_min, vol_max: Volume fraction bounds
        dp_min, dp_max: Pressure drop bounds
    
    Returns:
        (is_feasible, reason) tuple
    """
    
    # Check 1: Volume Fraction Bounds

    if volume_fraction < vol_min:
        return False, f"Volume too low: {volume_fraction:.3f} < {vol_min}"
    if volume_fraction > vol_max:
        return False, f"Volume too high: {volume_fraction:.3f} > {vol_max}"
    
    # Check 2: Pressure Drop Range

    if pressure_drop < dp_min:
        return False, f"Pressure drop too low: {pressure_drop:.4f} < {dp_min}"
    if pressure_drop > dp_max:
        return False, f"Pressure drop too high: {pressure_drop:.4f} > {dp_max}"
    
    # Check 3: Inlet→Outlet Connectivity (BFS)

    if not check_connectivity(density, masks):
        return False, "No fluid path from inlet to outlet"
    
    return True, "OK"


def check_connectivity(density, masks):
    """
    BFS connectivity check from inlet to outlet.
    
    Args:
        density: Density field [Nx, Ny] (can be on GPU)
        masks: BC masks dictionary
    
    Returns:
        True if fluid path exists, False otherwise
    """
    # Convert to binary (fluid = 1, solid = 0)
    binary = (density > 0.5).cpu().numpy().astype(np.uint8)
    
    # Find inlet fluid cells (left boundary, inlet_range)
    inlet_cells = []
    for y in range(masks['inlet_range'].start, masks['inlet_range'].stop):
        if binary[0, y] == 1:
            inlet_cells.append((0, y))
    
    if len(inlet_cells) == 0:
        return False  # No fluid at inlet
    
    # BFS from inlet cells
    from collections import deque
    visited = set(inlet_cells)
    queue = deque(inlet_cells)
    
    while queue:
        x, y = queue.popleft()
        
        # Check if reached outlet (right boundary, outlet_range)
        if x == Nx - 1 and masks['outlet_range'].start <= y < masks['outlet_range'].stop:
            return True
        
        # 4-connectivity neighbors (up, down, left, right)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < Nx and 0 <= ny < Ny:
                if binary[nx, ny] == 1 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    
    return False  # No path found

# ---------------------------------------------------------
# Checkpoint Load/Save Functions
# ---------------------------------------------------------
import json

def load_checkpoint():
    """Load progress from checkpoint file"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        print(f"📂 Loaded checkpoint: {checkpoint['total_designs']} designs already collected")
        return checkpoint
    else:
        print("📂 No checkpoint found. Starting fresh.")
        return {
            'total_designs': 0,
            'used_bc_configs': [],  # List of (inlet_y, outlet_y) tuples already tried
            'run_number': 0,
            'all_optimization_ids': []
        }

def save_checkpoint(checkpoint):
    """Save progress to checkpoint file"""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"💾 Checkpoint saved: {checkpoint['total_designs']} designs")


# ---------------------------------------------------------
# Plotting Utility: Save Density as PNG
# ---------------------------------------------------------
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt

def save_density_plot(density, masks, filepath, title=None):
    """
    Save density field as PNG image with BC markers.
    
    Args:
        density: [Nx, Ny] tensor or numpy array
        masks: BC masks dictionary from create_masks_for_bc()
        filepath: Output PNG path
        title: Optional title string
    """
    # Convert to numpy if needed
    if torch.is_tensor(density):
        density = density.detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
    
    # Plot density (fluid = white, solid = black)
    im = ax.imshow(density.T, origin='lower', cmap='gray_r', 
                   vmin=0, vmax=1, interpolation='none')
    
    # Mark inlet (green) and outlet (red)
    inlet_range = masks['inlet_range']
    outlet_range = masks['outlet_range']
    
    # Inlet: left boundary
    ax.axvspan(-0.5, 0.5, inlet_range.start - 0.5, inlet_range.stop - 0.5, 
               color='green', alpha=0.3, label='Inlet')
    ax.plot([0, 0], [inlet_range.start, inlet_range.stop], 
            color='green', linewidth=2)
    
    # Outlet: right boundary
    ax.axvspan(Nx - 1.5, Nx - 0.5, outlet_range.start - 0.5, outlet_range.stop - 0.5, 
               color='red', alpha=0.3, label='Outlet')
    ax.plot([Nx - 1, Nx - 1], [outlet_range.start, outlet_range.stop], 
            color='red', linewidth=2)
    
    # Labels and title
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)
    ax.tick_params(labelsize=7)
    
    # Minimal legend
    ax.legend(fontsize=6, loc='upper right', frameon=True)
    
    # Save and close
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)  # Free memory

# ---------------------------------------------------------
# ESO optimization loop
# ---------------------------------------------------------
alpha_max_constant = 100.0

# ---------------------------------------------------------
# [REPLACE] Final ESO loop with dataset generation
# ---------------------------------------------------------
import h5py
from collections import deque


# Dataset config
NUM_DESIGNS = 3            # How many valid designs to collect
VOLUME_THRESHOLD = 0.25        # Save if volume < this
PRESSURE_THRESHOLD = 2.0       # Save if pressure drop < this
# ---------------------------------------------------------
# Dataset Output Files (3 separate files)
# ---------------------------------------------------------
OUTPUT_FILE_FINAL = "./data/dataset_final.h5"              # Final designs only
OUTPUT_FILE_INTERMEDIATE = "./data/dataset_intermediate_above.h5"  # Trajectories (inlet > outlet)
OUTPUT_FILE_ALL = "./data/dataset_all_feasible.h5"         # All feasible designs


# ---------------------------------------------------------
# Checkpoint Configuration
# ---------------------------------------------------------
CHECKPOINT_FILE = "./data/generation_checkpoint.json"
BATCH_SIZE = 50  # Designs per run
TOTAL_DESIGNS_TARGET = 10000  # Overall goal (10 runs = 1000 designs)


# What to save in each
SAVE_FINAL_ONLY = True           # File 1: Just end result
SAVE_INTERMEDIATE_ABOVE = True   # File 2: Trajectories when inlet above outlet
SAVE_ALL_FEASIBLE = True         # File 3: Everything passing feasibility

ALPHA_MAX = 100.0
MAX_ESO_ITERS = 30             # ESO iterations per BC config

# Storage
# Storage for 3 files
final_densities, final_pressure_drops, final_volumes, final_bc_info = [], [], [], []
intermediate_densities, intermediate_pressure_drops, intermediate_volumes, intermediate_bc_info = [], [], [], []
all_densities, all_pressure_drops, all_volumes, all_bc_info = [], [], [], []

# ---------------------------------------------------------
# Dataset Generation Loop (3-file output)
# ---------------------------------------------------------
# ---------------------------------------------------------
# Dataset Generation Loop (with checkpointing)
# ---------------------------------------------------------
import h5py
from collections import deque
from datetime import datetime

# Load checkpoint
checkpoint = load_checkpoint()
used_bc_configs = set(tuple(bc) for bc in checkpoint['used_bc_configs'])
start_design_count = checkpoint['total_designs']
run_number = checkpoint['run_number'] + 1

print(f"🎯 Run #{run_number} | Target: {BATCH_SIZE} designs | Total goal: {TOTAL_DESIGNS_TARGET}")
print(f"   Already collected: {start_design_count} designs")
print(f"   BC configs used: {len(used_bc_configs)}")
print("=" * 70)

# Storage for this batch only
final_densities, final_pressure_drops, final_volumes, final_bc_info = [], [], [], []
intermediate_densities, intermediate_pressure_drops, intermediate_volumes, intermediate_bc_info = [], [], [], []
all_densities, all_pressure_drops, all_volumes, all_bc_info = [], [], [], []

attempts = 0
completed_optimizations = 0
global_optimization_id = checkpoint['total_designs']  # Continue ID from previous runs

while completed_optimizations < BATCH_SIZE and start_design_count + completed_optimizations < TOTAL_DESIGNS_TARGET and attempts < BATCH_SIZE * 10:
    attempts += 1
    
    # Sample random inlet/outlet Y centers
    inlet_y = np.random.randint(WALL_THICKNESS + 5, Ny - WALL_THICKNESS - 5)
    outlet_y = np.random.randint(WALL_THICKNESS + 5, Ny - WALL_THICKNESS - 5)
    
    # ─────────────────────────────────────────────────────
    # Skip if this BC config was already used
    # ─────────────────────────────────────────────────────
    bc_key = (int(inlet_y), int(outlet_y))
    if bc_key in used_bc_configs:
        continue  # Try next random config
    
    # Create masks for this BC config
    masks = create_masks_for_bc(inlet_y, outlet_y)
    
    # Reset topology for fresh optimization
    topology = torch.nn.Parameter(torch.ones(Nx, Ny, device=device))
    is_designable = ~masks['solid_mask'] & ~masks['fluid_mask'] & ~masks['orifice_mask']
    target_volume = np.random.uniform(0.15, VOLUME_THRESHOLD)
    
    # Track best design from this optimization run
    best_density = None
    best_dp = float('inf')
    best_volume = 1.0
    best_iteration = 0
    
    # Run ESO optimization
    for it in range(MAX_ESO_ITERS):
        if topology.grad is not None:
            topology.grad.zero_()
        
        # Forward simulation
        dp, rho, u, density = simulate(ALPHA_MAX, masks)
        volume = topology[is_designable].mean().item()
        dp_value = dp.item()
        
        # Backward pass
        dp.backward()
        
        # Check feasibility
        feasible, reason = is_feasible(
            density, dp_value, volume, masks,
            vol_min=0.10, vol_max=0.40,
            dp_min=0.01, dp_max=5.0
        )
        
        # Track best design from this run
        if feasible and dp_value < best_dp:
            best_density = density.clone().detach().cpu()
            best_dp = dp_value
            best_volume = volume
            best_iteration = it
        
        # FILE 2: Save intermediate steps (inlet > outlet only)
        if SAVE_INTERMEDIATE_ABOVE and inlet_y > outlet_y:
            if it >= 10 and feasible:  # Removed: it % 5 == 0
                intermediate_densities.append(density.clone().detach().cpu())
                intermediate_pressure_drops.append(dp_value)
                intermediate_volumes.append(volume)
                intermediate_bc_info.append({
                    'inlet_y': int(inlet_y),
                    'outlet_y': int(outlet_y),
                    'height_diff': int(inlet_y - outlet_y),
                    'iteration': int(it),
                    'optimization_id': global_optimization_id,
                    'feasibility_reason': reason,
                    'is_intermediate': True
                })
        
        # FILE 3: Save all feasible designs
        if SAVE_ALL_FEASIBLE and feasible and it >= 10:
            all_densities.append(density.clone().detach().cpu())
            all_pressure_drops.append(dp_value)
            all_volumes.append(volume)
            all_bc_info.append({
                'inlet_y': int(inlet_y),
                'outlet_y': int(outlet_y),
                'height_diff': int(inlet_y - outlet_y),
                'iteration': int(it),
                'optimization_id': global_optimization_id,
                'feasibility_reason': reason,
                'is_intermediate': (it < MAX_ESO_ITERS - 1)
            })
        
        # ESO removal step
        if volume > target_volume:
            dL_dgamma = topology.grad
            if dL_dgamma is not None:
                is_fluid = (topology > 0.5)
                eligible = is_fluid & is_designable & ~masks['fluid_dilated']
                sens = dL_dgamma[eligible]
                
                if sens.numel() > 0:
                    n_remove = max(int(eligible.sum().item() * REMOVAL_FRACTION), 1)
                    _, idx = torch.topk(sens, min(n_remove, sens.numel()), largest=True)
                    
                    flat_eligible = torch.where(eligible.flatten())[0]
                    removal = torch.zeros_like(topology, dtype=torch.bool)
                    removal.flatten()[flat_eligible[idx]] = True
                    
                    guard = removal[
                        max(0, masks['RIGHT_NOSLIP_X']-2):masks['RIGHT_NOSLIP_X']+1, 
                        masks['outlet_range']
                    ]
                    if not guard.all():
                        topology.data[removal] = 0.0
    
    # FILE 1: Save final design from this optimization run
    if best_density is not None:
        final_densities.append(best_density)
        final_pressure_drops.append(best_dp)
        final_volumes.append(best_volume)
        final_bc_info.append({
            'inlet_y': int(inlet_y),
            'outlet_y': int(outlet_y),
            'height_diff': int(inlet_y - outlet_y),
            'iteration': int(best_iteration),
            'optimization_id': global_optimization_id,
            'feasibility_reason': 'Best from ESO run',
            'is_intermediate': False
        })

        os.makedirs("./output/plots", exist_ok=True)
        plot_filename = f"./output/plots/final_opt{global_optimization_id:04d}_in{inlet_y}_out{outlet_y}_dp{best_dp:.3f}.png"
        
        save_density_plot(best_density, masks, plot_filename, 
                        title=f"Opt #{global_optimization_id} | Δp={best_dp:.3f} | Vol={best_volume:.2f}")
        

        # Mark this BC config as used
        used_bc_configs.add(bc_key)
        completed_optimizations += 1
        global_optimization_id += 1
        
        # ─────────────────────────────────────────────────
        # Save checkpoint after each successful design
        # ─────────────────────────────────────────────────
        checkpoint['total_designs'] = start_design_count + completed_optimizations
        checkpoint['used_bc_configs'] = [list(bc) for bc in used_bc_configs]
        checkpoint['run_number'] = run_number
        checkpoint['last_saved'] = datetime.now().isoformat()
        save_checkpoint(checkpoint)
        
        print(f"✅ Design {start_design_count + completed_optimizations}/{TOTAL_DESIGNS_TARGET} | "
              f"dp={best_dp:.4f} | vol={best_volume:.3f} | "
              f"inlet={inlet_y}, outlet={outlet_y}")

# Save batch files (with run number in filename)
OUTPUT_FILE_FINAL = f"./data/dataset_final_run{run_number}.h5"
OUTPUT_FILE_INTERMEDIATE = f"./data/dataset_intermediate_run{run_number}.h5"
OUTPUT_FILE_ALL = f"./data/dataset_all_run{run_number}.h5"


from datetime import datetime
# ---------------------------------------------------------
# Save All 3 Dataset Files
# ---------------------------------------------------------
def save_dataset_file(filepath, densities, pressure_drops, volumes, bc_info, dataset_type):
    """Generic function to save dataset to HDF5"""
    if len(densities) == 0:
        print(f"⚠️  Skipping {filepath} (no data)")
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        # Main data
        f.create_dataset('density', data=torch.stack(densities).numpy(), compression='gzip')
        f.create_dataset('pressure_drop', data=np.array(pressure_drops))
        f.create_dataset('volume_fraction', data=np.array(volumes))
        
        # BC parameters
        f.create_dataset('bc_inlet_y', data=np.array([b['inlet_y'] for b in bc_info]))
        f.create_dataset('bc_outlet_y', data=np.array([b['outlet_y'] for b in bc_info]))
        f.create_dataset('bc_height_diff', data=np.array([b['height_diff'] for b in bc_info]))
        
        # Optimization metadata
        f.create_dataset('eso_iteration', data=np.array([b['iteration'] for b in bc_info]))
        f.create_dataset('optimization_id', data=np.array([b['optimization_id'] for b in bc_info]))
        f.create_dataset('is_intermediate', data=np.array([b['is_intermediate'] for b in bc_info]))
        
        # Attributes
        f.attrs['num_designs'] = len(densities)
        f.attrs['dataset_type'] = dataset_type
        f.attrs['nx'] = Nx
        f.attrs['ny'] = Ny
        f.attrs['timestamp'] = datetime.now().isoformat()
        
        # Detailed metadata (JSON strings)
        import json
        metadata_json = [json.dumps(b) for b in bc_info]
        f.create_dataset('metadata', data=[m.encode('utf-8') for m in metadata_json])
    
    print(f"✅ Saved: {filepath}")
    print(f"   Designs: {len(densities)}")
    print(f"   Pressure range: [{min(pressure_drops):.3f}, {max(pressure_drops):.3f}]")
    print(f"   Volume range: [{min(volumes):.3f}, {max(volumes):.3f}]")


# Call save for all 3 files
save_dataset_file(OUTPUT_FILE_FINAL, final_densities, final_pressure_drops, 
                  final_volumes, final_bc_info, "final_designs_only")

save_dataset_file(OUTPUT_FILE_INTERMEDIATE, intermediate_densities, intermediate_pressure_drops, 
                  intermediate_volumes, intermediate_bc_info, "intermediate_inlet_above_outlet")

save_dataset_file(OUTPUT_FILE_ALL, all_densities, all_pressure_drops, 
                  all_volumes, all_bc_info, "all_feasible_designs")

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("📊 DATASET GENERATION COMPLETE")
print("=" * 70)
print(f"   File 1 (Final only):     {len(final_densities):4d} designs")
print(f"   File 2 (Intermediates):  {len(intermediate_densities):4d} designs")
print(f"   File 3 (All feasible):   {len(all_densities):4d} designs")
print(f"   Total attempts:          {attempts:4d}")
print(f"   Success rate:            {completed_optimizations / attempts * 100:.1f}%")
print("=" * 70)




'''
attempts = 0
while len(densities) < NUM_DESIGNS and attempts < NUM_DESIGNS * 10:
    attempts += 1
    
    # Sample random inlet/outlet Y centers (keep original height)
    inlet_y = np.random.randint(WALL_THICKNESS + 5, Ny - WALL_THICKNESS - 5)
    outlet_y = np.random.randint(WALL_THICKNESS + 5, Ny - WALL_THICKNESS - 5)
    
    # Create masks for this BC config
    masks = create_masks_for_bc(inlet_y, outlet_y)
    
    # Reset topology for fresh optimization
    topology = torch.nn.Parameter(torch.ones(Nx, Ny, device=device))
    is_designable = ~masks['solid_mask'] & ~masks['fluid_mask'] & ~masks['orifice_mask']
    target_volume = np.random.uniform(0.15, VOLUME_THRESHOLD)
    
    # Run ESO, collect intermediates
    for it in range(MAX_ESO_ITERS):
        if topology.grad is not None:
            topology.grad.zero_()
        
        dp, rho, u, density = simulate(ALPHA_MAX, masks)
        dp.backward()
        volume = topology[is_designable].mean().item()
        

        feasible, reason = is_feasible(
                density, dp_value, volume, masks,
                vol_min=0.10, vol_max=0.40,
                dp_min=0.01, dp_max=5.0
            )
        
        if it >= 10 and feasible:  # Only save if feasible + past warm-up
            densities.append(density.clone().detach().cpu())
            pressure_drops.append(dp_value)
            volumes.append(volume)
            bc_info.append({
                'inlet_y': inlet_y,
                'outlet_y': outlet_y,
                'height_diff': inlet_y - outlet_y,
                'iteration': it,
                'feasibility_reason': reason
            })
            print(f"✅ Saved design {len(densities)}/{NUM_DESIGNS} | vol={volume:.3f} | dp={dp_value:.4f} | {reason}")
 
            if len(densities) >= NUM_DESIGNS:
                break
        elif it >= 10 and not feasible:
            print(f"   ⚠️  Rejected: {reason}")

        # Simple ESO removal (same logic as original, adapted for masks)
        if volume > target_volume:
            dL_dgamma = topology.grad
            if dL_dgamma is not None:
                is_fluid = (topology > 0.5)
                eligible = is_fluid & is_designable & ~masks['fluid_dilated']
                sens = dL_dgamma[eligible]
                if sens.numel() > 0:
                    n_remove = max(int(eligible.sum().item() * REMOVAL_FRACTION), 1)
                    _, idx = torch.topk(sens, min(n_remove, sens.numel()), largest=True)
                    flat_eligible = torch.where(eligible.flatten())[0]
                    removal = torch.zeros_like(topology, dtype=torch.bool)
                    removal.flatten()[flat_eligible[idx]] = True
                    # Prevent outlet sealing
                    guard = removal[max(0, masks['RIGHT_NOSLIP_X']-2):masks['RIGHT_NOSLIP_X']+1, masks['outlet_range']]
                    if not guard.all():
                        topology.data[removal] = 0.0

# Save to HDF5
if len(densities) > 0:
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with h5py.File(OUTPUT_FILE, 'w') as f:
        # Detach + stack + convert to numpy
        dens_stack = torch.stack([d.detach() for d in densities]).numpy()
        
        f.create_dataset('density', data=dens_stack, compression='gzip')
        f.create_dataset('pressure_drop', data=np.array(pressure_drops))
        f.create_dataset('volume_fraction', data=np.array(volumes))
        f.create_dataset('bc_inlet_y', data=np.array([b['inlet_y'] for b in bc_info]))
        f.create_dataset('bc_outlet_y', data=np.array([b['outlet_y'] for b in bc_info]))
        f.attrs['num_designs'] = len(densities)
        f.attrs['volume_threshold'] = VOLUME_THRESHOLD
        f.attrs['pressure_threshold'] = PRESSURE_THRESHOLD
    print(f"\n💾 Dataset saved: {OUTPUT_FILE}")
    print(f"   Designs: {len(densities)} | Pressure range: [{min(pressure_drops):.3f}, {max(pressure_drops):.3f}]")
else:
    print("❌ No valid designs collected. Try relaxing thresholds.")
'''