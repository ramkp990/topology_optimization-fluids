"""
CMA-ES + BC-Conditioned VAE Optimizer
======================================
Minimizes pressure drop and optionally volume for a given BC config.

Usage:
    python cmaes_opt.py --inlet_y 45 --outlet_y 20
    python cmaes_opt.py --inlet_y 45 --outlet_y 20 --lambda_volume 0.5
    python cmaes_opt.py --inlet_y 45 --outlet_y 20 --lambda_volume 0.0  # dp only
"""

import argparse
import cma
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vae_fluid import FluidVAE, is_feasible, make_bc_mask
from fluid import create_masks_for_bc, simulate

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
LATENT_DIM      = 32
ALPHA_MAX       = 100.0
MAX_GENERATIONS = 50
POPSIZE         = 24
SIGMA0          = 0.5
BOUNDS          = [-3.0, 3.0]
THRESHOLD       = 0.5    # single binarization threshold used everywhere
Nx, Ny          = 64, 64
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------
# Plot helper
# ---------------------------------------------------------
def plot_design(design_np, inlet_y, outlet_y, dp, vol, fitness,
                lambda_volume, title, save_path):
    inlet_h  = int(0.1 * Nx)
    outlet_h = int(0.1 * Ny)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(design_np.T, cmap='gray_r', origin='lower', vmin=0, vmax=1)

    ax.plot([0, 0],
            [inlet_y - inlet_h//2, inlet_y + inlet_h//2],
            color='green', linewidth=3, label=f'Inlet y={inlet_y}')
    ax.plot([Nx-1, Nx-1],
            [outlet_y - outlet_h//2, outlet_y + outlet_h//2],
            color='red', linewidth=3, label=f'Outlet y={outlet_y}')

    ax.set_title(
        f"{title}\n"
        f"Δp={dp:.4f} | Vol={vol:.3f} | "
        f"fitness={fitness:.4f} (λ={lambda_volume})",
        fontsize=9
    )
    ax.legend(fontsize=8, loc='upper right')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------
# Fitness Function
# ---------------------------------------------------------
def decode_and_evaluate(solutions, vae, bc_mask, masks,
                        inlet_y, outlet_y,
                        lambda_volume, population_log):
    """
    Decode each latent vector, run LBM, compute fitness.

    fitness = dp + lambda_volume * volume

    Returns:
        fitnesses:  list of scalar fitness values (1e6 if infeasible)
        dp_values:  list of actual pressure drops (None if infeasible)
        vol_values: list of volumes
    """
    fitnesses  = []
    dp_values  = []
    vol_values = []

    with torch.no_grad():
        for z in solutions:
            z_tensor   = torch.FloatTensor(z).unsqueeze(0).to(DEVICE)
            recon      = vae.decode(z_tensor, bc_mask)
            density_np = recon[0, 0].cpu().numpy()

            # Binarize — single threshold used everywhere
            density_binary = (density_np > THRESHOLD).astype(np.float32)
            volume         = float(density_binary.mean())

            # Volume sanity check
            if volume < 0.05 or volume > 0.55:
                fitnesses.append(1e6)
                dp_values.append(None)
                vol_values.append(volume)
                population_log.append({
                    'z': z.copy(), 'inlet_y': inlet_y, 'outlet_y': outlet_y,
                    'volume': volume, 'pressure_drop': None,
                    'feasible': False, 'reason': 'Volume out of range',
                })
                continue

            # BFS connectivity check
            feasible, reason, _ = is_feasible(
                density_binary, inlet_y, outlet_y,
                threshold=THRESHOLD
            )

            if not feasible:
                fitnesses.append(1e6)
                dp_values.append(None)
                vol_values.append(volume)
                population_log.append({
                    'z': z.copy(), 'inlet_y': inlet_y, 'outlet_y': outlet_y,
                    'volume': volume, 'pressure_drop': None,
                    'feasible': False, 'reason': reason,
                })
                continue

            # LBM pressure drop
            try:
                density_tensor = torch.FloatTensor(density_binary).to(DEVICE)
                dp, _, _, _    = simulate(ALPHA_MAX, masks,
                                          density_input=density_tensor)
                dp_val  = dp.item()

                # Fitness: dp + weighted volume
                fitness = dp_val + lambda_volume * volume

                fitnesses.append(fitness)
                dp_values.append(dp_val)
                vol_values.append(volume)
                population_log.append({
                    'z': z.copy(), 'inlet_y': inlet_y, 'outlet_y': outlet_y,
                    'volume': volume, 'pressure_drop': dp_val,
                    'feasible': True, 'reason': 'OK',
                })

            except Exception as e:
                print(f"   ⚠️  LBM error: {e}")
                fitnesses.append(1e6)
                dp_values.append(None)
                vol_values.append(volume)
                population_log.append({
                    'z': z.copy(), 'inlet_y': inlet_y, 'outlet_y': outlet_y,
                    'volume': volume, 'pressure_drop': None,
                    'feasible': False, 'reason': f'LBM error: {e}',
                })

    return fitnesses, dp_values, vol_values


# ---------------------------------------------------------
# Main CMA-ES Loop
# ---------------------------------------------------------
def run_cmaes(inlet_y, outlet_y, vae_path="vae_best_new.pth", lambda_volume=0.0):

    print(f"🖥️  Device: {DEVICE}")
    print(f"📐 BC config: inlet_y={inlet_y}, outlet_y={outlet_y}")
    print(f"📂 VAE model: {vae_path}")
    print(f"⚖️  lambda_volume={lambda_volume} "
          f"({'dp only' if lambda_volume == 0 else 'dp + volume minimization'})")
    print("=" * 70)

    run_tag          = f"in{inlet_y}_out{outlet_y}_lv{lambda_volume}"
    intermediate_dir = f"./cmaes_intermediates/{run_tag}"
    os.makedirs(intermediate_dir, exist_ok=True)

    # ── Load VAE ──────────────────────────────────────────
    vae = FluidVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(
        torch.load(vae_path, map_location=DEVICE, weights_only=True)
    )
    vae.eval()
    print(f"✅ VAE loaded from {vae_path}")

    # ── BC tensor ─────────────────────────────────────────
    bc_norm = torch.FloatTensor([
        inlet_y  / Ny,
        outlet_y / Ny,
    ]).unsqueeze(0).to(DEVICE)
    print(f"   bc_norm = [{inlet_y/Ny:.4f}, {outlet_y/Ny:.4f}]")
    bc_mask_np = make_bc_mask(inlet_y, outlet_y)     # [2,64,64]
    bc_mask = torch.FloatTensor(bc_mask_np).unsqueeze(0).to(DEVICE)
    print(f"   BC mask created with inlet_y={inlet_y} and outlet_y={outlet_y}")

    # ── Physics masks ─────────────────────────────────────
    masks = create_masks_for_bc(inlet_y, outlet_y)
    print(f"   Masks created")

    # ── Population log ────────────────────────────────────
    population_log = []

    # ── CMA-ES ────────────────────────────────────────────
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

    print(f"\n🚀 Starting CMA-ES | pop={POPSIZE} | gen={MAX_GENERATIONS}")
    print(f"   fitness = dp + {lambda_volume} × volume")
    print("=" * 70)

    best_fitness = float('inf')
    best_z       = None
    best_design  = None
    best_dp      = float('inf')   # actual dp, NOT fitness
    best_vol     = float('inf')   # actual volume
    generation   = 0
    best_count   = 0

    while not es.stop():
        solutions = es.ask()

        fitnesses, dp_values, vol_values = decode_and_evaluate(
            solutions, vae, bc_mask, masks,
            inlet_y, outlet_y,
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

            # Decode for plot — same z gives same result
            with torch.no_grad():
                z_t         = torch.FloatTensor(best_z).unsqueeze(0).to(DEVICE)
                recon       = vae.decode(z_t, bc_mask)
                best_design = recon[0, 0].cpu().numpy()

            # Actual dp and volume from this generation's evaluation
            # (not recomputed — avoids LBM call)
            best_dp  = dp_values[current_best_idx]
            best_vol = vol_values[current_best_idx]

            print(f"   ⭐ New best | gen={generation} | "
                  f"fitness={best_fitness:.4f} | "
                  f"dp={best_dp:.4f} | vol={best_vol:.3f}")

            # Intermediate plot
            best_count += 1
            plot_design(
                best_design, inlet_y, outlet_y,
                best_dp, best_vol, best_fitness,
                lambda_volume,
                title=f"Best #{best_count} | Gen {generation}",
                save_path=os.path.join(
                    intermediate_dir,
                    f"best{best_count:03d}_gen{generation:03d}"
                    f"_dp{best_dp:.4f}_vol{best_vol:.3f}.png"
                )
            )

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 70)

    feasible_records   = [r for r in population_log if r['feasible']]
    infeasible_records = [r for r in population_log if not r['feasible']]

    print(f"📊 Population stats:")
    print(f"   Total evaluations: {len(population_log)}")
    print(f"   Feasible:          {len(feasible_records)}")
    print(f"   Infeasible:        {len(infeasible_records)}")

    # ── Save surrogate training data ──────────────────────
    if len(feasible_records) > 0:
        surrogate_path = f"surrogate_data_{run_tag}.npz"
        np.savez(
            surrogate_path,
            z             = np.array([r['z']             for r in feasible_records]),
            inlet_y_arr   = np.array([r['inlet_y']       for r in feasible_records]),
            outlet_y_arr  = np.array([r['outlet_y']      for r in feasible_records]),
            pressure_drop = np.array([r['pressure_drop'] for r in feasible_records]),
            volume        = np.array([r['volume']        for r in feasible_records]),
        )
        dp_vals = [r['pressure_drop'] for r in feasible_records]
        print(f"💾 Surrogate data: {surrogate_path}")
        print(f"   dp range:  [{min(dp_vals):.4f}, {max(dp_vals):.4f}]")
        print(f"   vol range: [{min(r['volume'] for r in feasible_records):.3f}, "
              f"{max(r['volume'] for r in feasible_records):.3f}]")

    if best_design is None:
        print("❌ No feasible design found.")
        return None

    print(f"\n✅ Optimization complete!")
    print(f"   fitness = dp + {lambda_volume} × vol")
    print(f"   Fitness  = {best_fitness:.4f}")
    print(f"   Best Δp  = {best_dp:.4f}")   # always the real dp
    print(f"   Volume   = {best_vol:.3f}")

    # ── Final plot ────────────────────────────────────────
    plot_path = f"cmaes_{run_tag}.png"
    plot_design(
        best_design, inlet_y, outlet_y,
        best_dp, best_vol, best_fitness,
        lambda_volume,
        title=f"CMA-ES Best | inlet={inlet_y} outlet={outlet_y}",
        save_path=plot_path
    )
    print(f"💾 Saved: {plot_path}")

    # ── Final result npz ──────────────────────────────────
    npz_path = f"cmaes_result_{run_tag}.npz"
    np.savez(
        npz_path,
        best_z        = best_z,
        best_design   = best_design,
        best_dp       = best_dp,         # actual pressure drop
        best_vol      = best_vol,        # actual volume fraction
        best_fitness  = best_fitness,    # dp + lambda_volume * vol
        lambda_volume = lambda_volume,
        inlet_y       = inlet_y,
        outlet_y      = outlet_y,
        bc_norm       = np.array([inlet_y/Ny, outlet_y/Ny]),
        n_generations = generation,
        n_evaluations = len(population_log),
        n_feasible    = len(feasible_records),
        threshold     = THRESHOLD,
    )
    print(f"💾 Saved: {npz_path}")

    # ── Convergence plot ──────────────────────────────────
    if len(feasible_records) > 0:
        dp_vals  = [r['pressure_drop'] for r in feasible_records]
        vol_vals = [r['volume']        for r in feasible_records]
        fit_vals = [dp + lambda_volume * vol
                    for dp, vol in zip(dp_vals, vol_vals)]
        running_best = np.minimum.accumulate(fit_vals)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: fitness convergence
        axes[0].plot(fit_vals,     alpha=0.4, color='steelblue',
                     label=f'Fitness (dp + {lambda_volume}·vol)')
        axes[0].plot(running_best, color='red', linewidth=2,
                     label='Running best')
        axes[0].set_xlabel('Feasible evaluation index')
        axes[0].set_ylabel('Fitness')
        axes[0].set_title(f'Fitness Convergence | λ={lambda_volume}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Right: dp and volume on dual axes
        ax2 = axes[1].twinx()
        axes[1].plot(dp_vals,  alpha=0.6, color='steelblue', label='Δp')
        ax2.plot(    vol_vals, alpha=0.6, color='orange',    label='Volume')
        axes[1].set_xlabel('Feasible evaluation index')
        axes[1].set_ylabel('Pressure drop Δp', color='steelblue')
        ax2.set_ylabel('Volume', color='orange')
        axes[1].set_title('Δp and Volume over evaluations')
        axes[1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        conv_path = f"cmaes_convergence_{run_tag}.png"
        plt.savefig(conv_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {conv_path}")

    return best_z, best_design, best_dp, best_vol


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CMA-ES latent space search with BC-conditioned VAE"
    )
    parser.add_argument('--inlet_y',       type=int,   required=True)
    parser.add_argument('--outlet_y',      type=int,   required=True)
    parser.add_argument('--vae_path',      type=str,   default='vae_best_new.pth')
    parser.add_argument('--max_gen',       type=int,   default=50)
    parser.add_argument('--popsize',       type=int,   default=24)
    parser.add_argument('--sigma0',        type=float, default=0.5)
    parser.add_argument('--lambda_volume', type=float, default=0.0,
                        help=(
                            'Weight on volume in fitness. '
                            '0.0 = dp only. '
                            '0.5 = balanced. '
                            '2.0+ = strong volume reduction. '
                            'fitness = dp + lambda_volume * volume'
                        ))

    args = parser.parse_args()

    MAX_GENERATIONS = args.max_gen
    POPSIZE         = args.popsize
    SIGMA0          = args.sigma0

    run_cmaes(
        inlet_y       = args.inlet_y,
        outlet_y      = args.outlet_y,
        vae_path      = args.vae_path,
        lambda_volume = args.lambda_volume,
    )