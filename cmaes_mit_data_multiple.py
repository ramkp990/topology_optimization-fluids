import argparse
import cma
import torch
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vae_fluid_multiple import FluidVAE, make_bc_mask, port_cells

# FIX: import the module as a whole so we can patch its globals before simulate()
import new_generate_dataset_multiple as fds
from new_generate_dataset_multiple import build_bc_masks, sample_ports

# =========================================================
# CONSTANTS
# =========================================================
LATENT_DIM      = 32
ALPHA_MAX       = 100.0
MAX_GENERATIONS = 50
POPSIZE         = 24
SIGMA0          = 0.5
BOUNDS          = [-3.0, 3.0]
THRESHOLD       = 0.5
Nx, Ny          = 64, 64
WALL            = 4
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'


# =========================================================
# PORT DESCRIPTION HELPERS
# =========================================================
def ports_to_desc(ports):
    parts = []
    for p in ports:
        parts.append(f"{p['type']}@{p['wall']}:{p['center']}")
    return " | ".join(parts)


def ports_to_tag(ports):
    parts = []
    for p in ports:
        parts.append(f"{p['type'][0]}{p['wall'][0]}{p['center']}")
    return "_".join(parts)


# =========================================================
# FEASIBILITY
# =========================================================
def bfs(binary, starts):
    from collections import deque
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

    outlet_cells = set()
    for p in outlet_ports:
        outlet_cells |= set(map(tuple, port_cells(p)))

    for inlet in inlet_ports:
        starts  = [tuple(c) for c in port_cells(inlet)]
        visited = bfs(binary, starts)
        if not any(cell in outlet_cells for cell in visited):
            return False, "disconnected"
    return True, "OK"


# =========================================================
# PLOT HELPER
# =========================================================
def plot_design(design_np, ports, dp, vol, fitness,
                lambda_volume, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(design_np.T, cmap='gray_r', origin='lower', vmin=0, vmax=1)

    inlet_colors  = ["green", "lime"]
    outlet_colors = ["red",   "tomato"]
    in_idx = out_idx = 0

    for p in ports:
        r    = p["range"]
        wall = p["wall"]
        if p["type"] == "inlet":
            color = inlet_colors[in_idx % 2]; in_idx += 1
        else:
            color = outlet_colors[out_idx % 2]; out_idx += 1

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
    ax.set_title(
        f"{title}\nDp={dp_str} | Vol={vol_str} | "
        f"fitness={fitness:.4f} (lv={lambda_volume})",
        fontsize=8
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================
# SET FDS GLOBALS  — must be called before every simulate()
# =========================================================
def set_fds_globals(masks, density_tensor):
    """
    Patch all globals that fds.simulate() reads so it uses
    the current port config and topology.
    """
    fds.topology      = torch.nn.Parameter(density_tensor)
    fds.solid_mask    = masks["solid_mask"]
    fds.fluid_mask    = masks["fluid_mask"]
    fds.orifice_mask  = masks["orifice_mask"]
    fds.inlet_ports   = masks["inlet_ports"]
    fds.outlet_ports  = masks["outlet_ports"]
    fds.ports         = masks["inlet_ports"] + masks["outlet_ports"]


# =========================================================
# FITNESS FUNCTION
# =========================================================
def decode_and_evaluate(solutions, vae, bc_mask_tensor, masks, ports,
                        lambda_volume, population_log):
    fitnesses  = []
    dp_values  = []
    vol_values = []

    with torch.no_grad():
        for z in solutions:
            z_tensor   = torch.FloatTensor(z).unsqueeze(0).to(DEVICE)
            recon      = vae.decode(z_tensor, bc_mask_tensor)
            density_np = recon[0, 0].cpu().numpy()   # [64,64]

            density_binary = (density_np > THRESHOLD).astype(np.float32)
            volume         = float(density_binary.mean())

            # --- volume sanity check ---
            if volume < 0.05 or volume > 0.55:
                fitnesses.append(1e6)
                dp_values.append(None)
                vol_values.append(volume)
                population_log.append({
                    "ports":         ports_to_tag(ports),
                    "volume":        volume,
                    "pressure_drop": None,
                    "feasible":      False,
                    "reason":        "volume_out_of_range",
                })
                continue

            # --- BFS connectivity ---
            connected, reason = check_connectivity(density_binary, ports)
            if not connected:
                fitnesses.append(1e6)
                dp_values.append(None)
                vol_values.append(volume)
                population_log.append({
                    "ports":         ports_to_tag(ports),
                    "volume":        volume,
                    "pressure_drop": None,
                    "feasible":      False,
                    "reason":        reason,
                })
                continue

            # --- LBM pressure drop ---
            # FIX: patch fds globals, then call fds.simulate(ALPHA_MAX)
            # simulate() reads topology, solid_mask, fluid_mask, inlet_ports,
            # outlet_ports from the fds module namespace — we set them all here
            try:
                density_tensor = torch.FloatTensor(density_binary).to(DEVICE)
                set_fds_globals(masks, density_tensor)

                # FIX: correct call — simulate takes only alpha_max
                dp, _ = fds.simulate(ALPHA_MAX)
                dp_val  = dp.item()
                fitness = dp_val + lambda_volume * volume

                fitnesses.append(fitness)
                dp_values.append(dp_val)
                vol_values.append(volume)
                population_log.append({
                    "ports":         ports_to_tag(ports),
                    "volume":        volume,
                    "pressure_drop": dp_val,
                    "feasible":      True,
                    "reason":        "OK",
                })

            except Exception as e:
                print(f"  LBM error: {e}")
                fitnesses.append(1e6)
                dp_values.append(None)
                vol_values.append(volume)
                population_log.append({
                    "ports":         ports_to_tag(ports),
                    "volume":        volume,
                    "pressure_drop": None,
                    "feasible":      False,
                    "reason":        f"lbm_error: {e}",
                })

    return fitnesses, dp_values, vol_values


# =========================================================
# MAIN CMA-ES LOOP
# =========================================================
def run_cmaes(ports, vae_path="vae_best_new.pth", lambda_volume=0.0):
    print(f"Device: {DEVICE}")
    print(f"Ports:  {ports_to_desc(ports)}")
    print(f"VAE:    {vae_path}")
    print(f"lambda_volume={lambda_volume}")
    print("=" * 70)

    run_tag          = ports_to_tag(ports) + f"_lv{lambda_volume}"
    intermediate_dir = f"./cmaes_intermediates/{run_tag}"
    os.makedirs(intermediate_dir, exist_ok=True)

    # --- Load VAE ---
    vae = FluidVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(
        torch.load(vae_path, map_location=DEVICE, weights_only=True)
    )
    vae.eval()
    print(f"VAE loaded from {vae_path}")

    # --- BC mask tensor [1,2,64,64] ---
    bc_mask_np     = make_bc_mask(ports)
    bc_mask_tensor = torch.FloatTensor(bc_mask_np).unsqueeze(0).to(DEVICE)

    # --- Physics masks ---
    masks = build_bc_masks(Nx, Ny, WALL, ports)
    print("BC masks built")

    # --- CMA-ES ---
    population_log = []

    es = cma.CMAEvolutionStrategy(
        [0.0] * LATENT_DIM,
        SIGMA0,
        {
            'bounds':    BOUNDS,
            'popsize':   POPSIZE,
            'maxiter':   MAX_GENERATIONS,
            'verb_disp': 5,
        }
    )

    print(f"\nStarting CMA-ES | pop={POPSIZE} | gen={MAX_GENERATIONS}")
    print(f"  fitness = dp + {lambda_volume} x volume")
    print("=" * 70)

    best_fitness = float('inf')
    best_z       = None
    best_design  = None
    best_dp      = None
    best_vol     = None
    generation   = 0
    best_count   = 0

    while not es.stop():
        solutions = es.ask()

        fitnesses, dp_values, vol_values = decode_and_evaluate(
            solutions, vae, bc_mask_tensor, masks, ports,
            lambda_volume, population_log
        )

        es.tell(solutions, fitnesses)
        es.disp()
        generation += 1

        current_best_idx = int(np.argmin(fitnesses))
        current_best_fit = fitnesses[current_best_idx]

        if current_best_fit < best_fitness and current_best_fit < 1e5:
            best_fitness = current_best_fit
            best_z       = solutions[current_best_idx].copy()
            best_dp      = dp_values[current_best_idx]
            best_vol     = vol_values[current_best_idx]

            with torch.no_grad():
                z_t         = torch.FloatTensor(best_z).unsqueeze(0).to(DEVICE)
                recon       = vae.decode(z_t, bc_mask_tensor)
                best_design = recon[0, 0].cpu().numpy()

            print(f"  New best | gen={generation} | "
                  f"fitness={best_fitness:.4f} | "
                  f"dp={best_dp:.4f} | vol={best_vol:.3f}")

            best_count += 1
            plot_design(
                best_design, ports,
                best_dp, best_vol, best_fitness, lambda_volume,
                title=f"Best #{best_count} | Gen {generation}",
                save_path=os.path.join(
                    intermediate_dir,
                    f"best{best_count:03d}_gen{generation:03d}"
                    f"_dp{best_dp:.4f}_vol{best_vol:.3f}.png"
                )
            )

    # --- Summary ---
    print("\n" + "=" * 70)
    feasible_records   = [r for r in population_log if r['feasible']]
    infeasible_records = [r for r in population_log if not r['feasible']]

    print(f"Population stats:")
    print(f"  Total evaluations : {len(population_log)}")
    print(f"  Feasible          : {len(feasible_records)}")
    print(f"  Infeasible        : {len(infeasible_records)}")

    if infeasible_records:
        reasons = {}
        for r in infeasible_records:
            reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
        print("  Infeasibility breakdown:")
        for reason, count in reasons.items():
            print(f"    {reason}: {count}")

    # --- Save surrogate data ---
    if feasible_records:
        surrogate_path = f"surrogate_data_{run_tag}.npz"
        np.savez(
            surrogate_path,
            pressure_drop = np.array([r['pressure_drop'] for r in feasible_records]),
            volume        = np.array([r['volume']        for r in feasible_records]),
        )
        dp_vals = [r['pressure_drop'] for r in feasible_records]
        print(f"\nSurrogate data saved: {surrogate_path}")
        print(f"  dp  range: [{min(dp_vals):.4f}, {max(dp_vals):.4f}]")
        print(f"  vol range: "
              f"[{min(r['volume'] for r in feasible_records):.3f}, "
              f"{max(r['volume'] for r in feasible_records):.3f}]")

    if best_design is None:
        print("No feasible design found.")
        return None

    print(f"\nOptimization complete!")
    print(f"  Fitness = {best_fitness:.4f}")
    print(f"  Best dp = {best_dp:.4f}")
    print(f"  Volume  = {best_vol:.3f}")

    # --- Final plot ---
    plot_path = f"cmaes_{run_tag}.png"
    plot_design(
        best_design, ports,
        best_dp, best_vol, best_fitness, lambda_volume,
        title=f"CMA-ES Best | {ports_to_desc(ports)}",
        save_path=plot_path
    )
    print(f"Saved: {plot_path}")

    # --- Save result npz ---
    npz_path = f"cmaes_result_{run_tag}.npz"
    np.savez(
        npz_path,
        best_z        = best_z,
        best_design   = best_design,
        best_dp       = best_dp,
        best_vol      = best_vol,
        best_fitness  = best_fitness,
        lambda_volume = lambda_volume,
        n_generations = generation,
        n_evaluations = len(population_log),
        n_feasible    = len(feasible_records),
        threshold     = THRESHOLD,
        ports_json    = json.dumps([
            {"type": p["type"], "wall": p["wall"], "center": p["center"]}
            for p in ports
        ]),
    )
    print(f"Saved: {npz_path}")

    # --- Convergence plot ---
    if feasible_records:
        dp_vals  = [r['pressure_drop'] for r in feasible_records]
        vol_vals = [r['volume']        for r in feasible_records]
        fit_vals = [dp + lambda_volume * vol
                    for dp, vol in zip(dp_vals, vol_vals)]
        running_best = np.minimum.accumulate(fit_vals)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(fit_vals,     alpha=0.4, color='steelblue',
                     label=f'fitness (dp + {lambda_volume}*vol)')
        axes[0].plot(running_best, color='red', linewidth=2,
                     label='running best')
        axes[0].set_xlabel('Feasible evaluation index')
        axes[0].set_ylabel('Fitness')
        axes[0].set_title(f'Convergence | lambda={lambda_volume}')
        axes[0].legend(); axes[0].grid(alpha=0.3)

        ax2 = axes[1].twinx()
        axes[1].plot(dp_vals,  alpha=0.6, color='steelblue', label='dp')
        ax2.plot(    vol_vals, alpha=0.6, color='orange',    label='volume')
        axes[1].set_xlabel('Feasible evaluation index')
        axes[1].set_ylabel('Pressure drop', color='steelblue')
        ax2.set_ylabel('Volume', color='orange')
        axes[1].set_title('dp and volume over evaluations')
        axes[1].legend(loc='upper left'); ax2.legend(loc='upper right')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        conv_path = f"cmaes_convergence_{run_tag}.png"
        plt.savefig(conv_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {conv_path}")

    return best_z, best_design, best_dp, best_vol


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CMA-ES latent-space search with BC-conditioned VAE"
    )
    parser.add_argument(
        '--port', action='append', nargs=3,
        metavar=('TYPE', 'WALL', 'CENTER'),
        help=(
            'Add a port. TYPE=inlet|outlet, '
            'WALL=left|right|top|bottom, '
            'CENTER=integer position. '
            'Repeat for multiple ports. '
            'Example: --port inlet left 30 --port outlet right 40'
        )
    )
    parser.add_argument('--vae_path',      type=str,   default='vae_best_new.pth')
    parser.add_argument('--max_gen',       type=int,   default=50)
    parser.add_argument('--popsize',       type=int,   default=24)
    parser.add_argument('--sigma0',        type=float, default=0.5)
    parser.add_argument('--lambda_volume', type=float, default=0.0,
                        help='Weight on volume in fitness. 0=dp only.')
    parser.add_argument('--random_ports',  action='store_true',
                        help='Sample a random port config.')
    parser.add_argument('--n_inlets',      type=int,   default=1)
    parser.add_argument('--n_outlets',     type=int,   default=1)

    args = parser.parse_args()

    MAX_GENERATIONS = args.max_gen
    POPSIZE         = args.popsize
    SIGMA0          = args.sigma0

    PORT_HEIGHT = int(0.10 * Ny)

    if args.random_ports:
        ports = (sample_ports(args.n_inlets,  "inlet") +
                 sample_ports(args.n_outlets, "outlet"))
        print(f"Randomly sampled ports: {ports_to_desc(ports)}")
    else:
        if not args.port:
            parser.error(
                "Provide ports via --port TYPE WALL CENTER "
                "or use --random_ports"
            )
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

    run_cmaes(
        ports         = ports,
        vae_path      = args.vae_path,
        lambda_volume = args.lambda_volume,
    )