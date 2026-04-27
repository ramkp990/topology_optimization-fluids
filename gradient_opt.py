"""
Latent Gradient Optimizer
=========================
Standalone optimizer that finds low-∆p fluid topologies by
gradient descent directly in the VAE latent space.

Gradient path:
    z  →  VAE decoder  →  sigmoid(logits/T)  →  LBM  →  ∆p
    ∂∆p/∂z = ∂∆p/∂ρ · ∂ρ/∂logits · ∂logits/∂z

Usage:
    python latent_grad_optimizer.py \
        --port inlet  bottom 32 \
        --port inlet  right  40 \
        --port outlet left   32 \
        --vae_path vae_best_new.pth \
        --n_restarts 5 \
        --n_steps 80

Outputs (parallel to cmaes outputs for direct comparison):
    latgrad_result_<tag>.npz
    latgrad_<tag>.png
    latgrad_convergence_<tag>.png
    latgrad_intermediates/<tag>/step_*.png
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vae_fluid_multiple import FluidVAE, make_bc_mask, port_cells
import new_generate_dataset_multiple as fds
from new_generate_dataset_multiple import build_bc_masks, sample_ports

# =========================================================
# CONSTANTS  (mirror cmaes_mit_data_multiple.py)
# =========================================================
LATENT_DIM = 32
ALPHA_MAX  = 100.0
THRESHOLD  = 0.5
Nx, Ny     = 64, 64
WALL       = 4
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'


# =========================================================
# PORT HELPERS  (identical to CMA-ES script)
# =========================================================
def ports_to_desc(ports):
    return " | ".join(f"{p['type']}@{p['wall']}:{p['center']}" for p in ports)

def ports_to_tag(ports):
    return "_".join(f"{p['type'][0]}{p['wall'][0]}{p['center']}" for p in ports)


# =========================================================
# CONNECTIVITY CHECK  (bidirectional, identical to CMA-ES)
# =========================================================
def bfs(binary, starts):
    visited = set(map(tuple, starts))
    queue   = deque(starts)
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx_, ny_ = x+dx, y+dy
            if 0 <= nx_ < Nx and 0 <= ny_ < Ny:
                if binary[nx_, ny_] and (nx_, ny_) not in visited:
                    visited.add((nx_, ny_))
                    queue.append((nx_, ny_))
    return visited



def check_connectivity(density_np, ports):
    binary = (density_np > THRESHOLD).astype(np.uint8)

    inlet_ports  = [p for p in ports if p["type"] == "inlet"]
    outlet_ports = [p for p in ports if p["type"] == "outlet"]

    outlet_cell_sets = [set(map(tuple, port_cells(p))) for p in outlet_ports]
    inlet_cell_sets  = [set(map(tuple, port_cells(p))) for p in inlet_ports]

    for i, inlet in enumerate(inlet_ports):
        starts  = [tuple(c) for c in port_cells(inlet)]
        visited = bfs(binary, starts)
        for j, oc in enumerate(outlet_cell_sets):
            if not any(cell in oc for cell in visited):
                return False, f"inlet_{i}_cannot_reach_outlet_{j}"

    for j, outlet in enumerate(outlet_ports):
        starts  = [tuple(c) for c in port_cells(outlet)]
        visited = bfs(binary, starts)
        for i, ic in enumerate(inlet_cell_sets):
            if not any(cell in ic for cell in visited):
                return False, f"outlet_{j}_cannot_reach_inlet_{i}"

    return True, "OK"

# funtion for checking onnectivity and giving penalty

def connectivity_penalty(density_np, ports):
    connected, reason = check_connectivity(density_np.detach().cpu().numpy(), ports)
    if connected:
        return 0.0
    else:
        print(f"  [diag] DISCONNECTED — reason: {reason} — returning penalty=1.0")
        return 1.0

# =========================================================
# DIFFERENTIABLE LBM FORWARD PASS
# =========================================================
def simulate_soft(soft_density, masks, alpha_max,
                  timesteps_warmup=2000, timesteps_grad=300):
    device = soft_density.device

    # Set fds module globals — required by lbm_step → apply_bcs → bounce_back_walls
    fds.solid_mask   = masks["solid_mask"]
    fds.fluid_mask   = masks["fluid_mask"]
    fds.orifice_mask = masks["orifice_mask"]
    fds.inlet_ports  = masks["inlet_ports"]
    fds.outlet_ports = masks["outlet_ports"]
    fds.ports        = masks["inlet_ports"] + masks["outlet_ports"]
    fds.topology     = torch.nn.Parameter(soft_density.detach().clone())

    solid_mask   = masks["solid_mask"]
    fluid_mask   = masks["fluid_mask"]
    inlet_ports  = masks["inlet_ports"]
    outlet_ports = masks["outlet_ports"]

    # Apply BC masks — keep differentiable
    density_clamped = soft_density.clamp(0.0, 1.0)
    density_clean   = torch.where(solid_mask,
                                   torch.zeros_like(density_clamped),
                                   density_clamped)
    density_clean   = torch.where(fluid_mask,
                                   torch.ones_like(density_clean),
                                   density_clean)

    f = torch.ones(Nx, Ny, 9, device=device) * fds.w

    # Warmup: detach density → no gradient → maximum speed
    density_warmup = density_clean.detach()
    with torch.no_grad():
        for t in range(timesteps_warmup):
            f = fds.lbm_step(f, density_warmup, alpha_max, t)

    # Grad phase: keep gradient connection through density_clean
    p_in_accum  = torch.tensor(0.0, device=device)
    p_out_accum = torch.tensor(0.0, device=device)

    for t in range(timesteps_warmup, timesteps_warmup + timesteps_grad):
        f = fds.lbm_step(f, density_clean, alpha_max, t)
        rho_t = f.sum(-1)
        p_in  = torch.stack([fds.sample_port_pressure(rho_t, p)
                              for p in inlet_ports]).mean()
        p_out = torch.stack([fds.sample_port_pressure(rho_t, p)
                              for p in outlet_ports]).mean()
        p_in_accum  = p_in_accum  + p_in
        p_out_accum = p_out_accum + p_out

    cs2 = 1.0 / 3.0
    dp  = (p_in_accum - p_out_accum) / timesteps_grad * cs2
    return dp, density_clean


# =========================================================
# PLOT HELPER  (mirrors CMA-ES plot_design)
# =========================================================
def plot_design(design_np, ports, dp, vol, step,
                title, save_path, extra_info=""):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(design_np.T, cmap='gray_r', origin='lower', vmin=0, vmax=1)

    for p in ports:
        r     = p["range"]
        wall  = p["wall"]
        color = "green" if p["type"] == "inlet" else "red"
        if wall == "left":
            ax.plot([0, 0], [r.start, r.stop], color=color, linewidth=3)
        elif wall == "right":
            ax.plot([Nx-1, Nx-1], [r.start, r.stop], color=color, linewidth=3)
        elif wall == "bottom":
            ax.plot([r.start, r.stop], [0, 0], color=color, linewidth=3)
        elif wall == "top":
            ax.plot([r.start, r.stop], [Ny-1, Ny-1], color=color, linewidth=3)

    dp_str  = f"{dp:.4f}"  if dp  is not None else "N/A"
    vol_str = f"{vol:.3f}" if vol is not None else "N/A"
    ax.set_title(f"{title}\nDp={dp_str} | Vol={vol_str}{extra_info}", fontsize=8)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================
# SINGLE-TRAJECTORY GRADIENT OPTIMIZER
# =========================================================
def optimize_single(vae, bc_mask_tensor, masks, ports,
                    z_init, lambda_volume,
                    n_steps, lr,
                    temp_start, temp_end,
                    lambda_binary,
                    lambda_conn,
                    intermediate_dir, run_id,
                    vol_lo=0.15, vol_hi=0.3,
                    save_every=10):

    device = DEVICE

    is_designable = (
        ~masks["solid_mask"] &
        ~masks["fluid_mask"] &
        ~masks["orifice_mask"]
    )

    z = torch.nn.Parameter(z_init.clone().to(device))
    optimizer = torch.optim.Adam([z], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.05
    )

    best_dp  = float('inf')
    best_z   = z.detach().clone()
    best_vol = None
    history  = []

    # Latch — never rolls back once True
    phase_connected = False

    for step in range(n_steps):

        optimizer.zero_grad()

        progress    = step / max(n_steps - 1, 1)
        temperature = temp_start * (temp_end / temp_start) ** progress

        # ── Forward pass ───────────────────────────────────────────────
        logits       = vae.decode(z, bc_mask_tensor)
        soft_density = torch.sigmoid(logits[0, 0] / temperature)

        dp, density_clean = simulate_soft(soft_density, masks, ALPHA_MAX)
        dp_val  = dp.item()
        vol     = density_clean[is_designable].mean()
        vol_val = vol.item()

        # ── Binary penalty ─────────────────────────────────────────────
        bin_penalty = (density_clean * (1.0 - density_clean)).mean()

        if step < n_steps // 3:

            with torch.no_grad():
                design_np = (density_clean > THRESHOLD).cpu().numpy()
            
            conn_penalty = connectivity_penalty(design_np, ports)


            vol_in_range      = (vol_val >= vol_lo) and (vol_val <= vol_hi)
            # Push volume UP toward vol_hi so fluid cells exist to
            # bridge the gap between inlet and outlet.
            vol_too_low   = torch.clamp(vol_hi - vol, min=0.0)
            vol_floor_pen = vol_too_low ** 2

            loss = (
                20.0      * vol_floor_pen
                + dp
            )

            phase_str = "PHASE1:reduce_dp_and_connect"

    
        else:
            with torch.no_grad():
                design_np = (density_clean > THRESHOLD).cpu().numpy()
            
            conn_penalty = connectivity_penalty(design_np, ports)
            w_bin = lambda_binary * progress

            # ── Detect if phase 2 has broken connectivity ──────────────────
            # outlet — a gradient step in phase 2 removed a critical cell.
            # Switch to reconnection mode temporarily without resetting latch.

            if conn_penalty == 1:
                _vol_hi = 0.3

                # Reconnection mode — same as phase 1 logic but within phase 2.
                # Strong connectivity drive, no dp, no volume upper bound.
                # Volume floor: don't let it drop too low while reconnecting.
                vol_too_low   = torch.clamp(_vol_hi - vol, min=0.0)
                vol_floor_pen = vol_too_low ** 2

                loss = (
                    lambda_conn * conn_penalty      # strong reconnection
                    + 20.0      * vol_floor_pen     # add fluid back if needed
                    + w_bin     * bin_penalty       # keep binary pressure
                    + dp
                )
                phase_str = "PHASE2:reconnect"

            else:
                _vol_hi = 0.2
                # ── Volume range penalty (two-sided) ──────────────────────────
                vol_below         = torch.clamp(vol_lo - vol, min=0.0)
                vol_above         = torch.clamp(vol - _vol_hi, min=0.0)
                vol_range_penalty = vol_below**2 + vol_above**2
                vol_in_range      = (vol_val >= vol_lo) and (vol_val <= vol_hi)

                # Normal phase 2 — connected, optimize dp + volume.
                loss = (
                    dp
                    + lambda_volume * vol_range_penalty
                    + lambda_conn * conn_penalty   # slightly stronger maintenance
                    + w_bin * bin_penalty
                )
                phase_str = "PHASE2:dp+vol"

        # ── Gradient step ──────────────────────────────────────────────
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        z.data.clamp_(-3.0, 3.0)

        # save best deisng if connected in phase 2
        if conn_penalty == 0.0 and phase_str.startswith("PHASE2") and dp_val < best_dp:
            best_dp  = dp_val
            best_z   = z.detach().clone()
            best_vol = vol_val

            with torch.no_grad():
                design_np = (torch.sigmoid(
                    vae.decode(best_z, bc_mask_tensor)[0, 0]
                ) > THRESHOLD).float().cpu().numpy()

            plot_design(
                design_np, ports, best_dp, best_vol, step,
                title=f"[{run_id}] Step {step} | Phase2 best",
                save_path=os.path.join(
                    intermediate_dir,
                    f"{run_id}_step{step:04d}_dp{best_dp:.4f}.png"
                )
            )

        # ── Logging ────────────────────────────────────────────────────
        vol_str = f"{'OK' if vol_in_range else 'OOB'}"
        print(f"  [{run_id}] step {step:03d} | {phase_str} | "
              f"T={temperature:.3f} | dp={dp_val:.5f} | "
              f"vol={vol_val:.3f}({vol_str}) | "
              f"conn={conn_penalty:.3f} | loss={loss.item():.5f}")

        history.append({
            'step':         step,
            'phase':        1 if not phase_connected else 2,
            'dp':           dp_val,
            'vol':          vol_val,
            'conn':         conn_penalty,
            'vol_in_range': vol_in_range,
            'loss':         loss.item(),
            'temperature':  temperature,
        })

    # If phase 2 was never reached, return final z with a warning
    if best_dp == float('inf'):
        best_z   = z.detach().clone()
        best_dp  = dp_val
        best_vol = vol_val
        print(f"  [{run_id}] WARNING: never connected — returning final z")

    return {
        'best_z':          best_z.squeeze(0).cpu().numpy(),
        'best_dp':         best_dp,
        'best_vol':        best_vol,
        'history':         history,
        'run_id':          run_id,
        'phase_reached':   2 if phase_connected else 1,
    }


# =========================================================
# FINAL LBM EVALUATION ON HARD BINARY
# =========================================================
def evaluate_binary(vae, bc_mask_tensor, masks, ports, z_np):
    """
    Decode z, threshold to binary, verify connectivity,
    run full LBM at binary topology.
    Returns dp_binary, vol_binary, connected, design_np.
    """
    device = DEVICE

    is_designable = (
        ~masks["solid_mask"] &
        ~masks["fluid_mask"] &
        ~masks["orifice_mask"]
    ).cpu().numpy()

    with torch.no_grad():
        z_t     = torch.FloatTensor(z_np).unsqueeze(0).to(device)
        logits  = vae.decode(z_t, bc_mask_tensor)
        prob    = torch.sigmoid(logits[0, 0])
        binary  = (prob > THRESHOLD).float()

    design_np = binary.cpu().numpy()

    # Connectivity check
    connected, reason = check_connectivity(design_np, ports)
    if not connected:
        return None, None, False, design_np, reason

    # Volume
    vol = float(design_np[is_designable].mean())

    # Full LBM
    fds.topology     = torch.nn.Parameter(binary.to(device))
    fds.solid_mask   = masks["solid_mask"]
    fds.fluid_mask   = masks["fluid_mask"]
    fds.orifice_mask = masks["orifice_mask"]
    fds.inlet_ports  = masks["inlet_ports"]
    fds.outlet_ports = masks["outlet_ports"]
    fds.ports        = masks["inlet_ports"] + masks["outlet_ports"]

    dp, _ = fds.simulate(ALPHA_MAX)
    return dp.item(), vol, True, design_np, "OK"


# =========================================================
# MAIN OPTIMIZER WITH MULTIPLE RESTARTS
# =========================================================
def run_latent_grad(ports,
                    vae_path      = "vae_best_new.pth",
                    n_restarts    = 5,
                    n_steps       = 80,
                    lr            = 0.05,
                    lambda_volume = 0.5,
                    lambda_binary = 2.0,
                    temp_start    = 1.0,
                    temp_end      = 0.1):
    """
    Main entry point. Runs n_restarts independent gradient trajectories
    from different random z initializations, returns best result.

    Multiple restarts are necessary because:
    - Gradient descent is local — different starts find different basins
    - The latent space has multiple topology families
    - Temperature annealing can get stuck in local minima
    """
    print(f"\nLatent Gradient Optimizer")
    print(f"  Device:        {DEVICE}")
    print(f"  Ports:         {ports_to_desc(ports)}")
    print(f"  VAE:           {vae_path}")
    print(f"  Restarts:      {n_restarts}")
    print(f"  Steps/restart: {n_steps}")
    print(f"  lr:            {lr}")
    print(f"  lambda_volume: {lambda_volume}")
    print(f"  lambda_binary: {lambda_binary}")
    print(f"  temperature:   {temp_start} → {temp_end}")
    print("=" * 70)

    run_tag          = ports_to_tag(ports) + f"_lv{lambda_volume}"
    intermediate_dir = f"./latgrad_intermediates/{run_tag}"
    os.makedirs(intermediate_dir, exist_ok=True)

    # ── Load VAE ──────────────────────────────────────────────────────
    vae = FluidVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(
        torch.load(vae_path, map_location=DEVICE, weights_only=True))
    vae.eval()

    # Freeze decoder weights — only z is optimized
    for param in vae.parameters():
        param.requires_grad_(False)

    # ── BC setup ──────────────────────────────────────────────────────
    bc_mask_np     = make_bc_mask(ports)
    bc_mask_tensor = torch.FloatTensor(bc_mask_np).unsqueeze(0).to(DEVICE)
    masks          = build_bc_masks(Nx, Ny, WALL, ports)

    # ── Run restarts ──────────────────────────────────────────────────
    restart_results = []

    for restart in range(n_restarts):
        print(f"\n── Restart {restart+1}/{n_restarts} ──────────────────────────")

        # Each restart uses a different random z initialization
        # Initialization strategy:
        #   restart 0: z = 0  (prior mean — safe, near-average topology)
        #   restart 1: z ~ N(0,1)  (random sample from prior)
        #   restart 2+: z ~ N(0, 0.5)  (tighter sampling, explore near-prior)
        if restart == 0:
            z_init = torch.zeros(1, LATENT_DIM)
        elif restart == 1:
            z_init = torch.randn(1, LATENT_DIM).clamp(-3, 3)
        else:
            z_init = (torch.randn(1, LATENT_DIM) * 0.5).clamp(-3, 3)

        run_id = f"r{restart+1:02d}"

        result = optimize_single(
            vae            = vae,
            bc_mask_tensor = bc_mask_tensor,
            masks          = masks,
            ports          = ports,
            z_init         = z_init,
            lambda_volume  = 50,
            n_steps        = n_steps,
            lr             = lr,
            temp_start     = 1,
            temp_end       = temp_end,
            lambda_binary  = lambda_binary,
            lambda_conn    = 10.0,  # fixed weight on connectivity penalty
            intermediate_dir = intermediate_dir,
            run_id         = run_id,
        )

        # Evaluate on hard binary
        print(f"\n  Evaluating binary topology for restart {restart+1}...")
        dp_bin, vol_bin, connected, design_np, reason = evaluate_binary(
            vae, bc_mask_tensor, masks, ports, result['best_z']
        )

        result['dp_binary']  = dp_bin
        result['vol_binary'] = vol_bin
        result['connected']  = connected
        result['reason']     = reason
        result['design_np']  = design_np

        if connected:
            print(f"  ✓ Restart {restart+1}: dp_soft={result['best_dp']:.5f} | "
                  f"dp_binary={dp_bin:.5f} | vol={vol_bin:.3f}")
        else:
            print(f"  ✗ Restart {restart+1}: binary topology disconnected ({reason})")

        restart_results.append(result)

    # ── Select best across restarts ───────────────────────────────────
    feasible = [r for r in restart_results if r['connected']]

    if not feasible:
        print("\nNo feasible binary topology found across all restarts.")
        print("Consider: more restarts, higher temp_end, lower lambda_binary")
        return None

    # Best = lowest dp on binary evaluation
    best = min(feasible, key=lambda r: r['dp_binary'])

    print(f"\n{'='*70}")
    print(f"Best result: restart={best['run_id']} | "
          f"dp={best['dp_binary']:.5f} | vol={best['vol_binary']:.3f}")
    print(f"Soft dp was: {best['best_dp']:.5f} | "
          f"Gap: {best['dp_binary'] - best['best_dp']:+.5f}")

    # ── Save outputs ──────────────────────────────────────────────────

    # Final topology plot
    plot_path = f"latgrad_{run_tag}.png"
    plot_design(
        best['design_np'], ports,
        best['dp_binary'], best['vol_binary'], step=None,
        title=f"Latent Grad Best | {ports_to_desc(ports)}",
        save_path=plot_path,
        extra_info=f" | restart={best['run_id']}"
    )
    print(f"Saved: {plot_path}")

    # NPZ result (same format as CMA-ES for direct comparison)
    npz_path = f"latgrad_result_{run_tag}.npz"
    np.savez(
        npz_path,
        best_z        = best['best_z'],
        best_design   = best['design_np'],
        best_dp       = best['dp_binary'],
        best_dp_soft  = best['best_dp'],
        best_vol      = best['vol_binary'],
        lambda_volume = lambda_volume,
        n_restarts    = n_restarts,
        n_steps       = n_steps,
        threshold     = THRESHOLD,
        ports_json    = json.dumps([
            {"type": p["type"], "wall": p["wall"], "center": p["center"]}
            for p in ports
        ]),
    )
    print(f"Saved: {npz_path}")

    # Convergence plot — all restarts on same axes
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, n_restarts))
    for i, r in enumerate(restart_results):
        hist    = r['history']
        steps   = [h['step'] for h in hist]
        dp_vals = [h['dp']   for h in hist]
        label   = f"{r['run_id']} ({'✓' if r['connected'] else '✗'})"
        lw = 2.5 if r == best else 1.0
        axes[0].plot(steps, dp_vals, color=colors[i], linewidth=lw, label=label)

    axes[0].set_xlabel("Gradient step")
    axes[0].set_ylabel("Pressure drop (soft)")
    axes[0].set_title(f"∆p per step — all restarts")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    # Temperature schedule
    prog   = np.linspace(0, 1, n_steps)
    temps  = temp_start * (temp_end / temp_start) ** prog
    axes[1].plot(temps, color='steelblue', linewidth=2)
    axes[1].set_xlabel("Gradient step")
    axes[1].set_ylabel("Temperature")
    axes[1].set_title("Sigmoid temperature annealing")
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Latent Gradient Optimizer | {ports_to_desc(ports)}", fontsize=9)
    plt.tight_layout()
    conv_path = f"latgrad_convergence_{run_tag}.png"
    plt.savefig(conv_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {conv_path}")

    # Summary table across all restarts
    print(f"\n{'─'*70}")
    print(f"{'Restart':<10} {'Best soft dp':<15} {'Binary dp':<12} "
          f"{'Vol':<8} {'Connected'}")
    print(f"{'─'*70}")
    for r in restart_results:
        dp_b = f"{r['dp_binary']:.5f}" if r['connected'] else "N/A"
        vol  = f"{r['vol_binary']:.3f}" if r['connected'] else "N/A"
        mark = "✓" if r['connected'] else "✗"
        best_mark = " ← best" if r == best else ""
        print(f"  {r['run_id']:<8} {r['best_dp']:<15.5f} {dp_b:<12} "
              f"{vol:<8} {mark}{best_mark}")
    print(f"{'─'*70}")

    return best['best_z'], best['design_np'], best['dp_binary'], best['vol_binary']


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Latent gradient optimizer for fluid topology"
    )
    parser.add_argument(
        '--port', action='append', nargs=3,
        metavar=('TYPE', 'WALL', 'CENTER'),
        help='Add a port. Example: --port inlet left 30'
    )
    parser.add_argument('--vae_path',      type=str,   default='vae_best_new.pth')
    parser.add_argument('--n_restarts',    type=int,   default=5,
                        help='Number of independent gradient trajectories')
    parser.add_argument('--n_steps',       type=int,   default=80,
                        help='Gradient steps per restart')
    parser.add_argument('--lr',            type=float, default=0.05)
    parser.add_argument('--lambda_volume', type=float, default=0.5)
    parser.add_argument('--lambda_binary', type=float, default=2.0,
                        help='Weight on binary penalty ρ(1-ρ)')
    parser.add_argument('--temp_start',    type=float, default=1.0,
                        help='Starting sigmoid temperature (smooth)')
    parser.add_argument('--temp_end',      type=float, default=0.1,
                        help='Final sigmoid temperature (near-binary)')
    parser.add_argument('--random_ports',  action='store_true')
    parser.add_argument('--n_inlets',      type=int,   default=1)

    args = parser.parse_args()

    PORT_HEIGHT = int(0.10 * Ny)

    if args.random_ports:
        ports = sample_ports(args.n_inlets, "inlet") + sample_ports(1, "outlet")
        print(f"Randomly sampled ports: {ports_to_desc(ports)}")
    else:
        if not args.port:
            parser.error("Provide ports via --port TYPE WALL CENTER "
                         "or use --random_ports")
        ports = []
        for port_args in args.port:
            ptype, wall, center = port_args[0], port_args[1], int(port_args[2])
            lo = max(WALL, center - PORT_HEIGHT // 2)
            hi = min(Ny - WALL, center + PORT_HEIGHT // 2)
            ports.append({
                "type":   ptype,
                "wall":   wall,
                "range":  slice(lo, hi),
                "center": center,
            })

    run_latent_grad(
        ports          = ports,
        vae_path       = args.vae_path,
        n_restarts     = args.n_restarts,
        n_steps        = args.n_steps,
        lr             = args.lr,
        lambda_volume  = args.lambda_volume,
        lambda_binary  = args.lambda_binary,
        temp_start     = args.temp_start,
        temp_end       = args.temp_end,
    )