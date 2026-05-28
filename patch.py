
import argparse
import cma
import torch
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from bo_search import run_bo
from vae_fluid_multiple import FluidVAE, make_bc_mask, port_cells

# botorch / gpytorch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.fit import fit_fully_bayesian_model_nuts 
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
 

import new_generate_dataset_multiple as fds
from new_generate_dataset_multiple import build_bc_masks, sample_ports


LATENT_DIM      = 32
ALPHA_MAX       = 100.0
MAX_GENERATIONS = 30
POPSIZE         = 20
SIGMA0          = 0.5
BOUNDS          = [-3.0, 3.0]
THRESHOLD       = 0.5
Nx, Ny          = 64, 64
WALL            = 4
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

# BO-specific
N_INITIAL_DEFAULT   = 20      # random seed points if no CMA-ES warm-start
N_ITERATIONS_DEFAULT= 80      # BO iterations after seeding
BATCH_Q             = 5       # candidates per BO step (q-batch)
N_RESTARTS          = 10      # acquisition optimisation restarts
RAW_SAMPLES         = 512     # raw samples for acq optimisation
NOISE_PERTURB       = 0.4     # std of Gaussian noise around seed_z for warm start
BOUNDS_LO           = -3.0
BOUNDS_HI           =  3.0


SAAS_WARMUP_STEPS  = 256   # NUTS warmup — increase to 512 for more stable fits
SAAS_NUM_SAMPLES   = 64    # posterior samples kept after warmup
SAAS_THINNING      = 16    # keep every Nth NUTS sample (reduces autocorrelation)
 
# How many feasible points needed before switching from vanilla GP to SAASBO.
# SAASBO needs at least ~10 points; below this fall back to SingleTaskGP.
SAASBO_MIN_POINTS  = 10
 
 

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


def check_connectivity(density_np, ports):
    """
    Full bidirectional connectivity check:
    1. Every inlet must reach EVERY outlet
    2. Every outlet must be reachable from EVERY inlet
    
    This prevents:
    - One inlet being isolated while the other carries all flow
    - One outlet being dead while another is connected
    - Shared-path shortcuts that satisfy the union check but not per-pair
    """
    binary = (density_np > THRESHOLD).astype(np.uint8)

    inlet_ports  = [p for p in ports if p["type"] == "inlet"]
    outlet_ports = [p for p in ports if p["type"] == "outlet"]

    # Build per-outlet cell sets (NOT merged)
    outlet_cell_sets = []
    for p in outlet_ports:
        outlet_cell_sets.append(set(map(tuple, port_cells(p))))

    # Build per-inlet cell sets
    inlet_cell_sets = []
    for p in inlet_ports:
        inlet_cell_sets.append(set(map(tuple, port_cells(p))))

    # ── Check 1: every inlet → every outlet ───────────────────────────
    for i, inlet in enumerate(inlet_ports):
        starts  = [tuple(c) for c in port_cells(inlet)]
        visited = bfs(binary, starts)

        for j, outlet_cells_j in enumerate(outlet_cell_sets):
            if not any(cell in outlet_cells_j for cell in visited):
                return (False,
                        f"inlet_{i}_cannot_reach_outlet_{j}")

    # ── Check 2: every outlet → every inlet (reverse BFS) ─────────────
    for j, outlet in enumerate(outlet_ports):
        starts  = [tuple(c) for c in port_cells(outlet)]
        visited = bfs(binary, starts)   # BFS is undirected so same function

        for i, inlet_cells_i in enumerate(inlet_cell_sets):
            if not any(cell in inlet_cells_i for cell in visited):
                return (False,
                        f"outlet_{j}_cannot_reach_inlet_{i}")

    return True, "OK"


# PLOT HELPER

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

def compute_sav(binary, is_designable):
    """
    Surface-area-to-volume ratio on the DESIGNABLE region only.
    Port/wall cells are forced fluid and always contribute the same
    perimeter regardless of topology — excluding them makes the metric
    sensitive only to the actual channel geometry.

    Returns (sav, perimeter, volume).
    """
    # mask to designable fluid cells only
    fluid = (binary * is_designable).astype(np.uint8)

    volume = int(fluid.sum())
    if volume == 0:
        return float("inf"), 0, 0

    padded = np.pad(fluid, 1, mode="constant", constant_values=0)
    center = padded[1:-1, 1:-1]
    perimeter = int(
        ((center == 1) & (padded[:-2, 1:-1] == 0)).sum() +   # up
        ((center == 1) & (padded[2:,  1:-1] == 0)).sum() +   # down
        ((center == 1) & (padded[1:-1, :-2] == 0)).sum() +   # left
        ((center == 1) & (padded[1:-1,  2:] == 0)).sum()     # right
    )
    return perimeter / volume, perimeter, volume

# ── add these two helpers near fit_surrogate ─────────────────────────────

def normalize_X(X, lo=BOUNDS_LO, hi=BOUNDS_HI):
    """Scale latent vectors from [lo, hi] → [0, 1] for GP."""
    return (X - lo) / (hi - lo)

def unnormalize_X(X, lo=BOUNDS_LO, hi=BOUNDS_HI):
    """Scale GP candidates from [0, 1] → [lo, hi] for VAE decode."""
    return X * (hi - lo) + lo


def decode_and_evaluate_debug(solutions, vae, bc_mask_tensor, masks, ports,
                        lambda_volume, population_log, SAV_THRESHOLD):
    fitnesses  = []
    dp_values  = []
    vol_values = []

    is_designable = (
        ~masks["solid_mask"] &
        ~masks["fluid_mask"] &
        ~masks["orifice_mask"]
    ).cpu().numpy()

    with torch.no_grad():
        z_batch    = torch.FloatTensor(np.array(solutions)).to(DEVICE)
        bc_batch   = bc_mask_tensor.expand(len(solutions), -1, -1, -1)
        prob_batch = torch.sigmoid(vae.decode(z_batch, bc_batch))
        density_all = prob_batch[:, 0].cpu().numpy()

    # ── DEBUG: print length check once ───────────────────────────────────
    print(f"  [debug] solutions={len(solutions)}  decoded={len(density_all)}")

    for sample_idx, density_np in enumerate(density_all):

        density_binary = (density_np > THRESHOLD).astype(np.float32)
        volume         = float(density_binary[is_designable].mean())

        # ── 1. volume gate ────────────────────────────────────────────────
        if volume < 0.10 or volume > 0.20 + 0.05:
            print(f"  [debug] sample {sample_idx}: FAIL volume={volume:.3f}")
            fitnesses.append(1e6)
            dp_values.append(None)
            vol_values.append(volume)
            population_log.append({
                "ports": ports_to_tag(ports), "volume": volume,
                "sav": None, "pressure_drop": None,
                "feasible": False, "reason": "volume_out_of_range",
            })
            continue

        # ── 2. BFS connectivity ───────────────────────────────────────────
        connected, reason = check_connectivity(density_binary, ports)
        if not connected:
            print(f"  [debug] sample {sample_idx}: FAIL connectivity vol={volume:.3f} reason={reason}")
            fitnesses.append(1e6)
            dp_values.append(None)
            vol_values.append(volume)
            population_log.append({
                "ports": ports_to_tag(ports), "volume": volume,
                "sav": None, "pressure_drop": None,
                "feasible": False, "reason": reason,
            })
            continue
        
        # ── 3. SA/V pre-screen ────────────────────────────────────────────
        sav, perimeter, _ = compute_sav(density_binary, is_designable)
        print(f"  [debug] sample {sample_idx}: vol={volume:.3f} sav={sav:.4f} threshold={SAV_THRESHOLD}")
        if sav > SAV_THRESHOLD:
            print(f"  [debug] sample {sample_idx}: FAIL sav={sav:.4f} > {SAV_THRESHOLD}")
            fitnesses.append(1e6)
            dp_values.append(None)
            vol_values.append(volume)
            population_log.append({
                "ports": ports_to_tag(ports), "volume": volume,
                "sav": float(sav), "pressure_drop": None,
                "feasible": False, "reason": f"sav_too_high:{sav:.4f}",
            })
            continue

        # ── 4. erosion check ──────────────────────────────────────────────
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(density_binary, iterations=2)
        if eroded[is_designable].sum() == 0:
            print(f"  [debug] sample {sample_idx}: FAIL erosion vol={volume:.3f} sav={sav:.4f}")
            fitnesses.append(1e6)
            dp_values.append(None)
            vol_values.append(volume)
            population_log.append({
                "ports": ports_to_tag(ports), "volume": volume,
                "sav": float(sav), "pressure_drop": None,
                "feasible": False, "reason": "too_thin_after_erosion",
            })
            continue   # ← THIS WAS MISSING — still missing in your code

        # ── 5. LBM ───────────────────────────────────────────────────────
        print(f"  [debug] sample {sample_idx}: reaching LBM vol={volume:.3f} sav={sav:.4f}")
        try:
            density_tensor = torch.FloatTensor(density_binary).to(DEVICE)
            set_fds_globals(masks, density_tensor)
            dp, _ = fds.simulate(ALPHA_MAX)
            dp_val  = dp.item()
            # relax dp_val by adding a fraction of volume to encourage BO to explore slightly higher volumes if it helps dp, but still penalise heavily above 0.2
            fitness = dp_val + lambda_volume * volume
            print(f"  [debug] sample {sample_idx}: LBM OK dp={dp_val:.4f} fit={fitness:.4f}")
            fitnesses.append(fitness)
            dp_values.append(dp_val)
            vol_values.append(volume)
            population_log.append({
                "ports": ports_to_tag(ports), "volume": volume,
                "sav": float(sav), "pressure_drop": dp_val,
                "feasible": True, "reason": "OK",
            })
        except Exception as e:
            print(f"  [debug] sample {sample_idx}: LBM ERROR {e}")
            fitnesses.append(1e6)
            dp_values.append(None)
            vol_values.append(volume)
            population_log.append({
                "ports": ports_to_tag(ports), "volume": volume,
                "sav": float(sav), "pressure_drop": None,
                "feasible": False, "reason": f"lbm_error:{e}",
            })

    # ── DEBUG: length sanity check ────────────────────────────────────────
    print(f"  [debug] fitnesses appended={len(fitnesses)} expected={len(solutions)}")
    assert len(fitnesses) == len(solutions), \
        f"LENGTH MISMATCH: {len(fitnesses)} fitnesses for {len(solutions)} solutions — missing continue somewhere"

    return fitnesses, dp_values, vol_values

def decode_and_evaluate__(solutions, vae, bc_mask_tensor, masks, ports,
                        lambda_volume, population_log,SAV_THRESHOLD):   # ← fast by default
    fitnesses  = []
    dp_values  = []
    vol_values = []

    is_designable = (
        ~masks["solid_mask"] &
        ~masks["fluid_mask"] &
        ~masks["orifice_mask"]
    ).cpu().numpy()

    with torch.no_grad():
        # batched decode — one GPU call for the whole population
        z_batch  = torch.FloatTensor(np.array(solutions)).to(DEVICE)
        bc_batch = bc_mask_tensor.expand(len(solutions), -1, -1, -1)
        prob_batch = torch.sigmoid(vae.decode(z_batch, bc_batch))
        density_all = prob_batch[:, 0].cpu().numpy()          # [pop, 64, 64]

    for density_np in density_all:

        density_binary = (density_np > THRESHOLD).astype(np.float32)
        volume         = float(density_binary[is_designable].mean())

        # ── 1. volume gate ────────────────────────────────────────────────
        # add a buffer from 0.2 for BO to explore slightly higher volumes if it helps dp, but still penalise heavily above 0.2
        if volume < 0.10 or volume > 0.20 + 0.05:
            fitnesses.append(1e6)
            dp_values.append(None)
            vol_values.append(volume)
            population_log.append({
                "ports":         ports_to_tag(ports),
                "volume":        volume,
                "sav":           None,
                "pressure_drop": None,
                "feasible":      False,
                "reason":        "volume_out_of_range",
            })
            continue

        # ── 2. BFS connectivity ───────────────────────────────────────────
        connected, reason = check_connectivity(density_binary, ports)
        if not connected:
            fitnesses.append(1e6)
            dp_values.append(None)
            vol_values.append(volume)
            population_log.append({
                "ports":         ports_to_tag(ports),
                "volume":        volume,
                "sav":           None,
                "pressure_drop": None,
                "feasible":      False,
                "reason":        reason,
            })
            continue

        # ── 3. SA/V pre-screen (skip LBM for geometrically poor designs) ─
        sav, perimeter, _ = compute_sav(density_binary, is_designable)
        if sav > SAV_THRESHOLD:
            fitnesses.append(1e6)
            dp_values.append(None)
            vol_values.append(volume)
            population_log.append({
                "ports":         ports_to_tag(ports),
                "volume":        volume,
                "sav":           float(sav),
                "pressure_drop": None,
                "feasible":      False,
                "reason":        f"sav_too_high:{sav:.4f}",
            })
            continue

        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(density_binary, iterations=2)
        if eroded[is_designable].sum() == 0:
            # channel is too thin everywhere — 1-2 pixel wide only
            fitnesses.append(1e6)
            dp_values.append(None)
            vol_values.append(volume)
            population_log.append({
                "ports":         ports_to_tag(ports),
                "volume":        volume,
                "sav":           float(sav),
                "pressure_drop": None,
                "feasible":      False,
                "reason":        "too_thin_after_erosion",
            })
        # ── 4. LBM pressure drop (fast warmup for intermediate evals) ────
        try:
            density_tensor = torch.FloatTensor(density_binary).to(DEVICE)
            set_fds_globals(masks, density_tensor)

            dp, _ = fds.simulate(ALPHA_MAX)
            dp_val  = dp.item()
            fitness = dp_val + lambda_volume * volume

            fitnesses.append(fitness)
            dp_values.append(dp_val)
            vol_values.append(volume)
            population_log.append({
                "ports":         ports_to_tag(ports),
                "volume":        volume,
                "sav":           float(sav),
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
                "sav":           float(sav),
                "pressure_drop": None,
                "feasible":      False,
                "reason":        f"lbm_error:{e}",
            })

    return fitnesses, dp_values, vol_values


# ─────────────────────────────────────────────────────────────────────────────
#  SEED GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_seed_points(seed_z, n_initial, latent_dim,
                     feasible_zs, feasible_fits,
                     noise_std=NOISE_PERTURB):
    lo, hi = BOUNDS_LO, BOUNDS_HI
    print("feasible_zs:", "provided" if feasible_zs is not None else "None")
    # ── Best case: use CMA-ES feasible population directly ───────────
    if feasible_zs is not None and len(feasible_zs) >= 3:
        Z    = torch.FloatTensor(np.array(feasible_zs))
        fits = feasible_fits
        print(f"  Warm-start: {len(feasible_zs)} points reused from "
              f"CMA-ES feasible population (no new LBM calls needed)")
        # return Z AND fits so Phase 1 can skip re-evaluation entirely
        return Z, fits   # ← two return values now

    # ── Fallback: perturb single best_z ──────────────────────────────
    if seed_z is not None:
        seed      = torch.FloatTensor(seed_z)
        noise     = torch.randn(n_initial - 1, latent_dim) * noise_std
        perturbed = (seed.unsqueeze(0) + noise).clamp(lo, hi)
        Z         = torch.cat([seed.unsqueeze(0), perturbed], dim=0)
        print(f"  Warm-start: {n_initial} points perturbed around best_z")
        return Z, None   # ← None means Phase 1 must evaluate these

    # ── Cold start ────────────────────────────────────────────────────
    sobol = torch.quasirandom.SobolEngine(latent_dim, scramble=True)
    Z     = sobol.draw(n_initial).float() * (hi - lo) + lo
    print(f"  Cold-start: {n_initial} Sobol points")
    return Z, None


# ─────────────────────────────────────────────────────────────────────────────
#  BOTORCH GP SURROGATE + ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────

def _fit_standard_gp(train_X, train_Y):
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
 
    gp  = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()
    return gp, "standard_gp"
 

def _fit_saasbo(train_X, train_Y):
    """
    Fit a SaasFullyBayesianSingleTaskGP via NUTS.
 
    train_X : [N, D]  float64   (D=32 for your latent space)
    train_Y : [N, 1]  float64   (negated fitness — BO maximises)
 
    Returns the fitted model.
    """
    gp = SaasFullyBayesianSingleTaskGP(
        train_X  = train_X,
        train_Y  = train_Y,
        outcome_transform = Standardize(m=1),
    )
 
    fit_fully_bayesian_model_nuts(
        gp,
        warmup_steps = SAAS_WARMUP_STEPS,
        num_samples  = SAAS_NUM_SAMPLES,
        thinning     = SAAS_THINNING,
        disable_progbar = False,   # set True to suppress NUTS progress bar
    )
    gp.eval()
    return gp, "saasbo"
 
 
def fit_surrogate(train_X, train_Y):
    """
    Fit GP surrogate. Automatically switches to SAASBO once enough
    feasible points have been collected.
 
    train_X : [N, D]  float64
    train_Y : [N, 1]  float64  (negated fitness)
 
    Returns (model, model_type_str).
    """
    n_points = train_X.shape[0]
 
    if n_points < SAASBO_MIN_POINTS:
        print(f"    [surrogate] {n_points} pts < {SAASBO_MIN_POINTS} → StandardGP")
        return _fit_standard_gp(train_X, train_Y)
    else:
        print(f"    [surrogate] {n_points} pts → SAASBO (NUTS "
              f"warmup={SAAS_WARMUP_STEPS}, samples={SAAS_NUM_SAMPLES})")
        return _fit_saasbo(train_X, train_Y)
 
 


def get_next_candidates(gp, train_Y, bounds_t, q=5):
    """
    Optimise acquisition function to get the next q candidates.
 
    Works for BOTH standard GP and SAASBO — the acquisition API is the same.
    Returns float32 tensor [q, D].
    """
    best_f = train_Y.max()
 
    # qLogEI handles batches (q>1) for both model types
    acqf = qLogExpectedImprovement(model=gp, best_f=best_f)
 
    candidates, acq_value = optimize_acqf(
        acq_function = acqf,
        bounds       = bounds_t,
        q            = q,
        num_restarts = 10,
        raw_samples  = 512,
        options      = {"maxiter": 200, "batch_limit": 5},
    )
 
    print(f"    [acq] best EI candidate value: {acq_value.item():.4f}")
    return candidates.float()   # [q, D]
 
def _save_convergence_plot(feasible_records, all_fit, lambda_volume, run_tag):
    if not feasible_records:
        return

    dp_vals  = [r["pressure_drop"] for r in feasible_records]
    vol_vals = [r["volume"]        for r in feasible_records]
    fit_vals = all_fit                        # only feasible fits
    running_best = np.minimum.accumulate(fit_vals)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # -- fitness convergence --
    axes[0].plot(fit_vals,     alpha=0.4, color="steelblue",
                 label=f"fitness (dp + {lambda_volume}×vol)")
    axes[0].plot(running_best, color="red", linewidth=2, label="running best")
    axes[0].set_xlabel("Feasible evaluation index")
    axes[0].set_ylabel("Fitness")
    axes[0].set_title("BO Convergence")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # -- dp and volume separately --
    ax2 = axes[1].twinx()
    axes[1].plot(dp_vals,  alpha=0.6, color="steelblue", label="dp")
    ax2.plot(    vol_vals, alpha=0.6, color="orange",    label="volume")
    axes[1].set_xlabel("Feasible evaluation index")
    axes[1].set_ylabel("Pressure drop",  color="steelblue")
    ax2.set_ylabel("Volume fraction", color="orange")
    axes[1].set_title("dp and volume over evaluations")
    axes[1].legend(loc="upper left"); ax2.legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    # -- dp histogram --
    axes[2].hist(dp_vals, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    axes[2].axvline(min(dp_vals), color="red",   linestyle="--", label=f"min={min(dp_vals):.4f}")
    axes[2].axvline(np.mean(dp_vals), color="orange", linestyle="--",
                    label=f"mean={np.mean(dp_vals):.4f}")
    axes[2].set_xlabel("Pressure drop")
    axes[2].set_ylabel("Count")
    axes[2].set_title("dp Distribution (feasible)")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    conv_path = f"bo_convergence_{run_tag}.png"
    plt.savefig(conv_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {conv_path}")


def run_bo(
    ports,
    vae_path        = "vae_best_new.pth",
    seed_z          = None,           # np.ndarray [latent_dim] from CMA-ES
    feasible_zs     = None,           # list of np.ndarray [latent_dim] from CMA-ES population
    feasible_fits   = None,           # list of fitness values corresponding to feasible_zs
    lambda_volume   = 0.0,
    n_initial       = N_INITIAL_DEFAULT,
    n_iterations    = N_ITERATIONS_DEFAULT,
    batch_q         = BATCH_Q,
    sav_threshold   = 0.4,
):
    print("feasible_zs:", "provided" if feasible_zs is not None else "None")
    """
    Bayesian Optimisation in VAE latent space.

    Parameters
    ----------
    ports           : list of port dicts (same format as CMA-ES)
    vae_path        : path to saved VAE weights
    seed_z          : [latent_dim] array — warm-start from CMA-ES best z
    lambda_volume   : weight on volume term in fitness (same as CMA-ES)
    n_initial       : number of seed evaluations before BO starts
    n_iterations    : number of BO acquisition steps
    batch_q         : candidates per BO step
    sav_threshold   : SA/V cutoff passed to decode_and_evaluate

    Returns
    -------
    best_z, best_design, best_dp, best_vol
    """
    run_tag = ports_to_tag(ports) + f"_lv{lambda_volume}_bo"
    out_dir = f"./bo_intermediates/{run_tag}"
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("BAYESIAN OPTIMISATION  (latent-space, warm-started from CMA-ES)")
    print("=" * 70)
    print(f"  Device       : {DEVICE}")
    print(f"  Ports        : {ports_to_desc(ports)}")
    print(f"  VAE          : {vae_path}")
    print(f"  seed_z       : {'CMA-ES best' if seed_z is not None else 'cold start'}")
    print(f"  n_initial    : {n_initial}")
    print(f"  n_iterations : {n_iterations}  (batch_q={batch_q})")
    print(f"  lambda_vol   : {lambda_volume}")
    print("=" * 70)

    # ── Load VAE ─────────────────────────────────────────────────────────────
    vae = FluidVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(
        torch.load(vae_path, map_location=DEVICE, weights_only=True)
    )
    vae.eval()

    # ── BC mask & physics masks ───────────────────────────────────────────────
    bc_mask_np     = make_bc_mask(ports)
    bc_mask_tensor = torch.FloatTensor(bc_mask_np).unsqueeze(0).to(DEVICE)
    masks          = build_bc_masks(Nx, Ny, WALL, ports)

    bounds_t = torch.tensor(
        [[BOUNDS_LO] * LATENT_DIM,
         [BOUNDS_HI] * LATENT_DIM],
        dtype=torch.float64,
        device="cpu",          # botorch acq optimisation runs on CPU
    )

    # ── Tracking ─────────────────────────────────────────────────────────────
    population_log = []
    all_z          = []    # all evaluated z vectors (feasible)
    all_fit        = []    # corresponding fitness values
    best_fitness   = float("inf")
    best_z         = None
    best_design    = None
    best_dp        = None
    best_vol       = None
    best_count     = 0

    # ── Phase 1: seed evaluations ─────────────────────────────────────────────
    print(f"\n[Phase 1] Evaluating {n_initial} seed points …")

    Z_seed, known_fits = make_seed_points(seed_z, n_initial, LATENT_DIM, feasible_zs=feasible_zs, feasible_fits=feasible_fits)

    if known_fits is not None:
        # ── CMA-ES population reused — zero new LBM calls ────────────────
        print(f"[Phase 1] Reusing {len(known_fits)} CMA-ES feasible points — "
            f"skipping LBM re-evaluation")
        for z, fit in zip(Z_seed.numpy(), known_fits):
            all_z.append(z)
            all_fit.append(fit)
            if fit < best_fitness:
                best_fitness = fit
                best_z       = z.copy()
                #best_dp      = dp    # ← now set
                #best_vol     = vol   # ← now set
                with torch.no_grad():
                    z_t         = torch.FloatTensor(z).unsqueeze(0).to(DEVICE)
                    best_design = vae.decode(z_t, bc_mask_tensor)[0, 0].cpu().numpy()
        print(f"  Seed phase done | feasible={len(all_z)} | "
            f"best_fit={best_fitness:.4f}")

    else:
        fitnesses, dp_vals, vol_vals = decode_and_evaluate_debug(
            Z_seed.numpy(), vae, bc_mask_tensor, masks, ports,
            lambda_volume, population_log, sav_threshold,
        )

        for idx, (z, fit, dp, vol) in enumerate(
                zip(Z_seed.numpy(), fitnesses, dp_vals, vol_vals)):

            if fit < 1e5:                    # feasible
                all_z.append(z)
                all_fit.append(fit)

                if fit < best_fitness:
                    best_fitness = fit
                    best_z       = z.copy()
                    best_dp      = dp
                    best_vol     = vol

                    with torch.no_grad():
                        z_t        = torch.FloatTensor(z).unsqueeze(0).to(DEVICE)
                        recon      = vae.decode(z_t, bc_mask_tensor)
                        best_design = recon[0, 0].cpu().numpy()

                    best_count += 1
                    plot_design(
                        best_design, ports, best_dp, best_vol,
                        best_fitness, lambda_volume,
                        title=f"BO seed best #{best_count}",
                        save_path=os.path.join(
                            out_dir,
                            f"seed_best{best_count:03d}"
                            f"_dp{best_dp:.4f}_vol{best_vol:.3f}.png"
                        ),
                    )

            print(f"  seed {idx+1:3d}/{n_initial} | fit={fit:.4f} | "
                f"dp={f'{dp:.4f}' if dp is not None else 'N/A':>8} | "
                f"vol={vol:.3f}")

        if len(all_z) < 3:
            print("\n  Not enough feasible seed points to fit GP. "
                "Try increasing n_initial or relaxing volume bounds.")
            return best_z, best_design, best_dp, best_vol

        print(f"\n  Seed phase done | feasible={len(all_z)}/{n_initial} | "
            f"best_fit={best_fitness:.4f}")

    # ── Phase 2: BO acquisition loop ─────────────────────────────────────────
    print(f"\n[Phase 2] Running {n_iterations} BO iterations (q={batch_q}) …")

    # Convert to float64 tensors for botorch
    # #train_X = torch.tensor(np.array(all_z),   dtype=torch.float64)  # [N, D]
    # #train_Y = torch.tensor(
    #     [[-f] for f in all_fit],               # negate: GP maximises
    #     dtype=torch.float64
    # )                                                                  # [N, 1]

    train_X = normalize_X(torch.tensor(np.array(all_z), dtype=torch.float64))  # [N, D]
    train_Y = torch.tensor(
        [[-f] for f in all_fit], dtype=torch.float64
    )                                                                  # [N, 1] 
    bounds_t = torch.tensor(
        [[0.0] * LATENT_DIM,
        [1.0] * LATENT_DIM],
        dtype=torch.float64,
        device="cpu",
    )
    for iteration in range(n_iterations):

        # ── Fit GP ────────────────────────────────────────────────────────
        try:
            #gp = fit_surrogate(train_X, train_Y)
            gp, model_type = fit_surrogate(train_X, train_Y)
        except Exception as e:
            print(f"  GP fit failed at iter {iteration+1}: {e} — skipping")
            continue

        # ── Acquire next batch ────────────────────────────────────────────
        try:
            candidates_raw = get_next_candidates(
                gp, train_Y, bounds_t, q=batch_q
            )                                          # [q, D]  float32
        except Exception as e:
            print(f"  Acq optimisation failed at iter {iteration+1}: {e} — skipping")
            continue

        candidates = unnormalize_X(candidates_raw)  # [q, D]  float64

        # ── Evaluate candidates ───────────────────────────────────────────
        fits, dps, vols = decode_and_evaluate_debug(
            candidates.numpy(), vae, bc_mask_tensor, masks, ports,
            lambda_volume, population_log, sav_threshold,
        )

        # ── Update dataset & tracking ─────────────────────────────────────
        new_X_rows = []
        new_Y_rows = []

        for c_idx, (z, fit, dp, vol) in enumerate(
                zip(candidates.numpy(), fits, dps, vols)):

            if fit < 1e5:   # only add feasible to GP training set
                new_X_rows.append(z)
                new_Y_rows.append(-fit)   # negate

                all_z.append(z)
                all_fit.append(fit)

                if fit < best_fitness:
                    best_fitness = fit
                    best_z       = z.copy()
                    best_dp      = dp
                    best_vol     = vol

                    with torch.no_grad():
                        z_t        = torch.FloatTensor(z).unsqueeze(0).to(DEVICE)
                        recon      = vae.decode(z_t, bc_mask_tensor)
                        best_design = recon[0, 0].cpu().numpy()

                    best_count += 1
                    plot_design(
                        best_design, ports, best_dp, best_vol,
                        best_fitness, lambda_volume,
                        title=f"BO best #{best_count} | iter {iteration+1}",
                        save_path=os.path.join(
                            out_dir,
                            f"best{best_count:03d}_iter{iteration+1:04d}"
                            f"_dp{best_dp:.4f}_vol{best_vol:.3f}.png"
                        ),
                    )
                    print(f"  ★ New best | iter={iteration+1} | "
                          f"fit={best_fitness:.4f} | dp={best_dp:.4f} | "
                          f"vol={best_vol:.3f}")

        # Append new rows to training tensors
        if new_X_rows:
            train_X = torch.cat([
                train_X,
                normalize_X(torch.tensor(np.array(new_X_rows), dtype=torch.float64))
            ])
            train_Y = torch.cat([
                train_Y,
                torch.tensor([[y] for y in new_Y_rows], dtype=torch.float64)
            ])

        # Progress print every 5 iterations
        if (iteration + 1) % 5 == 0 or iteration == 0:
            feasible_so_far = len(all_z)
            total_evals     = len(population_log)
            print(f"  iter {iteration+1:4d}/{n_iterations} | "
                  f"GP training pts={len(train_X):4d} | "
                  f"feasible={feasible_so_far}/{total_evals} | "
                  f"best_fit={best_fitness:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    feasible_records   = [r for r in population_log if     r["feasible"]]
    infeasible_records = [r for r in population_log if not r["feasible"]]

    print("\n" + "=" * 70)
    print("BO COMPLETE")
    print("=" * 70)
    print(f"  Total evaluations : {len(population_log)}")
    print(f"  Feasible          : {len(feasible_records)}")
    print(f"  Infeasible        : {len(infeasible_records)}")

    if infeasible_records:
        reasons = {}
        for r in infeasible_records:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        print("  Infeasibility breakdown:")
        for reason, count in reasons.items():
            print(f"    {reason}: {count}")

    if best_design is None:
        print("  No feasible design found during BO.")
        return None, None, None, None

    print(f"\n  Best fitness : {best_fitness:.4f}")
    print(f"  Best dp      : {best_dp:.4f}")
    print(f"  Best volume  : {best_vol:.3f}")

    # ── Final plot ────────────────────────────────────────────────────────────
    final_plot = f"bo_final_{run_tag}.png"
    plot_design(
        best_design, ports, best_dp, best_vol, best_fitness, lambda_volume,
        title=f"BO Final | {ports_to_desc(ports)}",
        save_path=final_plot,
    )
    print(f"  Saved: {final_plot}")

    # ── Save result npz ───────────────────────────────────────────────────────
    npz_path = f"bo_result_{run_tag}.npz"
    np.savez(
        npz_path,
        best_z        = best_z,
        best_design   = best_design,
        best_dp       = best_dp,
        best_vol      = best_vol,
        best_fitness  = best_fitness,
        lambda_volume = lambda_volume,
        n_iterations  = n_iterations,
        n_evaluations = len(population_log),
        n_feasible    = len(feasible_records),
        threshold     = THRESHOLD,
        ports_json    = json.dumps([
            {"type": p["type"], "wall": p["wall"], "center": p["center"]}
            for p in ports
        ]),
    )
    print(f"  Saved: {npz_path}")

    # ── Convergence plot ──────────────────────────────────────────────────────
    _save_convergence_plot(
        feasible_records, all_fit, lambda_volume, run_tag
    )

    return best_z, best_design, best_dp, best_vol