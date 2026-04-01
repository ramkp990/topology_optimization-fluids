import torch
import torch.nn.functional as F
import numpy as np
import os
from pyevtk.hl import gridToVTK
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
REMOVAL_FRACTION = 0.4


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

ports = [
    {"type":"inlet",  "wall":"left",   "range": make_slot(25, PORT_HEIGHT)},
    {"type":"inlet",  "wall":"bottom", "range": make_slot(30, PORT_HEIGHT)},
    {"type":"outlet", "wall":"right",  "range": make_slot(10, PORT_HEIGHT)},
    {"type":"outlet", "wall":"top",    "range": make_slot(50, PORT_HEIGHT)},
]

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


# =========================================================
# GENERIC BOUNDARY CONDITIONS (all walls)
# =========================================================
def bounce_back_walls(f):
    """Bounce-back on all outer walls except ports."""
    f = f.clone()

    # bottom wall (y = 0)
    f[:,0,2] = f[:,0,4]
    f[:,0,5] = f[:,0,7]
    f[:,0,6] = f[:,0,8]

    # top wall (y = Ny-1)
    f[:,-1,4] = f[:,-1,2]
    f[:,-1,7] = f[:,-1,5]
    f[:,-1,8] = f[:,-1,6]

    # left wall (x = 0)
    f[0,:,1] = f[0,:,3]
    f[0,:,5] = f[0,:,7]
    f[0,:,8] = f[0,:,6]

    # right wall (x = Nx-1)
    f[-1,:,3] = f[-1,:,1]
    f[-1,:,6] = f[-1,:,8]
    f[-1,:,7] = f[-1,:,5]

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
        f[0,r,5] = f[0,r,7] + (1/6)*rho*u_in
        f[0,r,8] = f[0,r,6] + (1/6)*rho*u_in

    elif wall == "right":
        rho = (f[-1,r,0]+f[-1,r,2]+f[-1,r,4]+2*(f[-1,r,1]+f[-1,r,5]+f[-1,r,8]))/(1+u_in)
        f[-1,r,3] = f[-1,r,1] - (2/3)*rho*u_in
        f[-1,r,7] = f[-1,r,5] - (1/6)*rho*u_in
        f[-1,r,6] = f[-1,r,8] - (1/6)*rho*u_in

    elif wall == "bottom":
        rho = (f[r,0,0]+f[r,0,1]+f[r,0,3]+2*(f[r,0,4]+f[r,0,7]+f[r,0,8]))/(1-u_in)
        f[r,0,2] = f[r,0,4] + (2/3)*rho*u_in
        f[r,0,5] = f[r,0,7] + (1/6)*rho*u_in
        f[r,0,6] = f[r,0,8] + (1/6)*rho*u_in

    elif wall == "top":
        rho = (f[r,-1,0]+f[r,-1,1]+f[r,-1,3]+2*(f[r,-1,2]+f[r,-1,5]+f[r,-1,6]))/(1+u_in)
        f[r,-1,4] = f[r,-1,2] - (2/3)*rho*u_in
        f[r,-1,7] = f[r,-1,5] - (1/6)*rho*u_in
        f[r,-1,8] = f[r,-1,6] - (1/6)*rho*u_in

def apply_outlet(f, port):
    r = port["range"]
    wall = port["wall"]
    rho_out = 1.0

    if wall == "left":
        u = 1 - (f[0,r,0]+f[0,r,2]+f[0,r,4]+2*(f[0,r,3]+f[0,r,6]+f[0,r,7]))/rho_out
        f[0,r,1] = f[0,r,3] + (2/3)*rho_out*u

    elif wall == "right":
        u = -1 + (f[-1,r,0]+f[-1,r,2]+f[-1,r,4]+2*(f[-1,r,1]+f[-1,r,5]+f[-1,r,8]))/rho_out
        f[-1,r,3] = f[-1,r,1] - (2/3)*rho_out*u

    elif wall == "bottom":
        u = 1 - (f[r,0,0]+f[r,0,1]+f[r,0,3]+2*(f[r,0,4]+f[r,0,7]+f[r,0,8]))/rho_out
        f[r,0,2] = f[r,0,4] + (2/3)*rho_out*u

    elif wall == "top":
        u = -1 + (f[r,-1,0]+f[r,-1,1]+f[r,-1,3]+2*(f[r,-1,2]+f[r,-1,5]+f[r,-1,6]))/rho_out
        f[r,-1,4] = f[r,-1,2] - (2/3)*rho_out*u

def apply_bcs(f, u_in):
    f = bounce_back_walls(f)

    for p in inlet_ports:
        apply_inlet(f, p, u_in)

    for p in outlet_ports:
        apply_outlet(f, p)

    return f


# =========================================================
# VISUALIZATION
# =========================================================
def save_design_png(topo_tensor, dp_val, filename="results/best_design.png"):
    """
    Color map:
      white      = void  (topology = 0, open channel)
      dark gray  = solid material (topology = 1)
      near-black = wall
      blue       = inlet ports
      red        = outlet ports
    """
    topo_np    = topo_tensor.detach().cpu().numpy()   # shape (Nx, Ny)
    solid_np   = solid_mask.cpu().numpy()
    vol        = float((topo_np > 0.5).mean())

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
    save_design_png(final_topology, final_dp, "results/best_design.png")

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


if __name__ == "__main__":
    main()