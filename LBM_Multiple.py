import torch
import torch.nn.functional as F
import numpy as np
import os
from pyevtk.hl import gridToVTK
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

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


# =========================================================
# GENERIC PORT SYSTEM  (ports can be on ANY wall)
# =========================================================

def make_slot(center, height):
    lo = int(center - height // 2)
    hi = int(center + height // 2)
    return slice(lo, hi)

PORT_HEIGHT = int(0.10 * Ny)

# usage wxample : python lbm_multiple.py --port inlet  top    16  --port inlet  bottom 25 --port outlet left   32  --target_volume 0.2

# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LBM Multiple Ports")

parser.add_argument(
    "--target_volume",
    type=float,
    default=0.195,
    help="Target volume fraction"
)

parser.add_argument(
    "--port",
    dest="ports",
    action="append",
    nargs=4,
    metavar=("TYPE","WALL","CENTER","HEIGHT"),
    help="--port inlet|outlet left|right|top|bottom CENTER HEIGHT",
)

args = parser.parse_args()

# ─────────────────────────────────────────────────────────────
# Build ports from CLI
# ─────────────────────────────────────────────────────────────


ports = []
for ptype, wall, center, height in args.ports:
    center = int(center)
    height = int(height)

    ports.append({
        "type":  ptype,
        "wall":  wall,
        "range": make_slot(center, height)
    })

print("\n[LBM] Ports from CLI:")
for p in ports:
    r = p["range"]
    print(f"   {p['type']:6s} @ {p['wall']:6s}  center={(r.start+r.stop)//2}  height={r.stop-r.start}")



inlet_ports  = [p for p in ports if p["type"] == "inlet"]
outlet_ports = [p for p in ports if p["type"] == "outlet"]


# =========================================================
# BUILD GENERIC WALL + PORT MASKS
# =========================================================

solid_mask   = torch.zeros(Nx, Ny, device=device, dtype=torch.bool)
orifice_mask = torch.zeros_like(solid_mask)
fluid_mask   = torch.zeros_like(solid_mask)

# Build outer walls
solid_mask[0:WALL_THICKNESS, :]  = True
solid_mask[-WALL_THICKNESS:, :]  = True
solid_mask[:, 0:WALL_THICKNESS]  = True
solid_mask[:, -WALL_THICKNESS:]  = True

def carve_port(port):
    r = port["range"]
    wall = port["wall"]

    if wall == "left":
        solid_mask[0:WALL_THICKNESS, r] = False
        orifice_mask[0:WALL_THICKNESS, r] = True
        fluid_mask[0, r] = True

    elif wall == "right":
        solid_mask[-WALL_THICKNESS:, r] = False
        orifice_mask[-WALL_THICKNESS:, r] = True
        fluid_mask[Nx-1, r] = True

    elif wall == "bottom":
        solid_mask[r, 0:WALL_THICKNESS] = False
        orifice_mask[r, 0:WALL_THICKNESS] = True
        fluid_mask[r, 0] = True

    elif wall == "top":
        solid_mask[r, -WALL_THICKNESS:] = False
        orifice_mask[r, -WALL_THICKNESS:] = True
        fluid_mask[r, Ny-1] = True

for p in ports:
    carve_port(p)

fluid_dilated = F.max_pool2d(
    orifice_mask.float().unsqueeze(0).unsqueeze(0),
    3, stride=1, padding=1
)[0,0].bool()

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
    return (p_in_accum - p_out_accum) / timesteps_grad * cs2

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
# VISUALIZATION
# =========================================================
def save_design_png(topo_tensor, dp_val, filename="results/best_design.png",
                    current_vol=None):
    topo_np  = topo_tensor.detach().cpu().numpy()
    solid_np = solid_mask.cpu().numpy()

    if current_vol is not None:
        vol = float(current_vol)
    else:
        # Correct fallback — designable cells only
        is_designable_np = (~solid_mask & ~fluid_mask & ~orifice_mask).cpu().numpy()
        vol = float(topo_np[is_designable_np].mean())  

    # Build RGBA canvas — default white (void)
    img = np.ones((Ny, Nx, 4))

    # Fluid material (topology > 0.5, not wall, not orifice)
    designable_np = ~solid_np & ~orifice_mask.cpu().numpy()
    fluid_cells   = (topo_np > 0.5) & designable_np
    img[fluid_cells.T] = [0.22, 0.22, 0.22, 1.0]   # dark gray

    for p in inlet_ports:
        r = p["range"]
        if p["wall"] == "left":
            img[r, 0:WALL_THICKNESS] = [0.18,0.45,0.80,1]
        elif p["wall"] == "right":
            img[r, -WALL_THICKNESS:] = [0.18,0.45,0.80,1]
        elif p["wall"] == "bottom":
            img[0:WALL_THICKNESS, r] = [0.18,0.45,0.80,1]
        elif p["wall"] == "top":
            img[-WALL_THICKNESS:, r] = [0.18,0.45,0.80,1]

    for p in outlet_ports:
        r = p["range"]
        if p["wall"] == "left":
            img[r, 0:WALL_THICKNESS] = [0.82,0.20,0.20,1]
        elif p["wall"] == "right":
            img[r, -WALL_THICKNESS:] = [0.82,0.20,0.20,1]
        elif p["wall"] == "bottom":
            img[0:WALL_THICKNESS, r] = [0.82,0.20,0.20,1]
        elif p["wall"] == "top":
            img[-WALL_THICKNESS:, r] = [0.82,0.20,0.20,1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, origin="lower", interpolation="nearest")
    ax.set_title(
        f"Optimized topology  |  Δp = {dp_val:.5f}  |  vol = {vol:.3f}",
        fontsize=10
    )
    ax.axis("off")

    patches = [
        mpatches.Patch(color=[0.22]*3,             label="fluid material"),
        mpatches.Patch(color=[0.06]*3,             label="solid wall"),
        mpatches.Patch(color=[0.18, 0.45, 0.80],  label="inlet"),
        mpatches.Patch(color=[0.82, 0.20, 0.20],  label="outlet"),
        mpatches.Patch(facecolor="white", edgecolor="gray", lw=0.8, label="void / channel"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7,
              framealpha=0.88, edgecolor="gray")

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Image saved → {filename}")


# =========================================================
# MAIN ESO LOOP
# =========================================================
def main():
    alpha_max_constant = 100.0
    n_iterations = 50
    dp_history = []

    for it in range(n_iterations):
        start_time = time.time()

        if topology.grad is not None:
            topology.grad.zero_()

        pressure_drop = simulate(alpha_max_constant)
        pressure_drop.backward()

        dp_val = pressure_drop.item()
        dp_history.append(dp_val)

        with torch.no_grad():
            is_designable = ~solid_mask & ~fluid_mask & ~orifice_mask
            current_vol   = topology[is_designable].mean()

            if current_vol <= target_volume:
                print(f"iter {it:03d} | Δp={dp_val:.6f} | vol={topology.mean().item():.3f} | Target volume reached. Stopping.")
                print(f"Final Δp = {dp_val:.6f}")
                print(f"volume fraction = {current_vol:.3f} (target was {target_volume:.3f})")
                break

            dL_dgamma     = topology.grad

            is_fluid      = (topology > 0.5)
            eligible_mask = is_fluid & is_designable & ~fluid_dilated
            eligible_sens = dL_dgamma[eligible_mask] 


            n_fluid_cells  = is_fluid[is_designable].sum().item()
            n_target_cells = int(target_volume * is_designable.sum().item())
            n_remove = max(int((n_fluid_cells - n_target_cells) * REMOVAL_FRACTION), 1)
            n_remove = min(n_remove, eligible_sens.numel())
            _, idx = torch.topk(eligible_sens, n_remove, largest=True)

            removal_mask  = torch.zeros_like(topology, dtype=torch.bool)
            eligible_flat = torch.where(eligible_mask.flatten())[0]
            removal_mask.flatten()[eligible_flat[idx]] = True


            # -------------------------------------------------
            # GENERIC outlet sealing protection (all walls)
            # -------------------------------------------------
            sealed = False
            for p in outlet_ports:
                r = p["range"]
                wall = p["wall"]
                if wall == "left":
                    guard = removal_mask[0:WALL_THICKNESS+1, r]
                elif wall == "right":
                    guard = removal_mask[Nx-WALL_THICKNESS-1:Nx, r]
                elif wall == "bottom":
                    guard = removal_mask[r, 0:WALL_THICKNESS+1]
                elif wall == "top":
                    guard = removal_mask[r, Ny-WALL_THICKNESS-1:Ny]

                if guard.all():
                    print(f"iter {it:03d} | WARNING: removal would seal an outlet — skipping.")
                    sealed = True
                    break

            if not sealed:
                topology[removal_mask] = 0.0

        elapsed = time.time() - start_time
        print(
            f"iter {it:03d} | Δp={dp_val:.6f} | "
            f"vol={topology.mean().item():.3f} | "
            f"removed={n_remove} | time={elapsed:.1f}s"
        )

    # =========================================================
    # SAVE — always use the FINAL topology (converged channels)
    # Iter 0 has lowest Δp only because the domain is full of
    # material (least resistance). That is NOT the optimized design.
    # =========================================================
    final_topology = topology.detach().clone()
    final_dp       = dp_history[-1]

    print(f"\nFinal Δp = {final_dp:.6f}")
    print(f"(Iter 0 Δp was {dp_history[0]:.6f} — full domain, not the design)")

    os.makedirs("results", exist_ok=True)
    torch.save(final_topology,  "results/best_topology.pt")
    np.save("results/best_topology.npy", final_topology.cpu().numpy())
    #save_design_png(final_topology, final_dp, "results/best_design.png", current_vol=final_topology.mean().item())
    is_designable = ~solid_mask & ~fluid_mask & ~orifice_mask
    final_vol     = final_topology[is_designable].mean().item()

    save_design_png(
        final_topology, final_dp,
        "results/best_design.png",
        current_vol=final_vol          # ← correct
    )

    # Convergence plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(dp_history, marker="o", markersize=3, linewidth=1.2, color="#1a6fb5")
    ax.set_xlabel("ESO iteration")
    ax.set_ylabel("Mean pressure drop")
    ax.set_title("Convergence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/convergence.png", dpi=150)
    plt.close()
    print("  Convergence plot → results/convergence.png")

    print("\nSaved:")
    print("   results/best_topology.pt")
    print("   results/best_topology.npy")
    print("   results/best_design.png")
    print("   results/convergence.png")

    return final_topology, final_vol, final_dp


if __name__ == "__main__":
    main()