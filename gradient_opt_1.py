
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
from scipy.ndimage import binary_erosion

from vae_fluid_multiple import FluidVAE, make_bc_mask, port_cells, make_port_mask
import new_generate_dataset_multiple as fds
from new_generate_dataset_multiple import build_bc_masks, sample_ports


LATENT_DIM = 32
ALPHA_MAX  = 100.0
THRESHOLD  = 0.5
Nx, Ny     = 64, 64
WALL       = 4
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'


def ports_to_desc(ports):
    return " | ".join(f"{p['type']}@{p['wall']}:{p['center']}" for p in ports)

def ports_to_tag(ports):
    return "_".join(f"{p['type'][0]}{p['wall'][0]}{p['center']}" for p in ports)



# CONNECTIVITY CHECK  

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
    connected, reason = check_connectivity(density_np, ports)
    if connected:
        return 0.0
    else:
        print(f"  [diag] DISCONNECTED — reason: {reason} — returning penalty=1.0")
        return 1.0


# DIFFERENTIABLE LBM FORWARD PASS for soft density to keep gradients

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


# plot 

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



def optimize_single1(vae, bc_mask_tensor, masks, ports,
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
    target_reached = False
    target_vol = 0.2
    history  = []

    # Latch — never rolls back once True
    phase_connected = False
    pahse_2_flag = 0

    phase_3_initialized = False
    phase_3_best_dp     = float('inf')
    phase3_best_dp_vol = float('inf')
    phase_3_no_improve  = 0
    PHASE_3_PATIENCE    = 50    # exit if no improvement for this many steps
    PHASE_3_LR_MULT    = 5.0   # aggressive lr multiplier for large jumps
    PHASE_3_GRAD_CLIP  = 5.0   # relaxed from 1.0 — allow larger steps
    lambda_vol_al  = 0.0   # Lagrange multiplier
    rho_al         = 50.0  # penalty strength

# add make_port_mask to this existing import:
# from vae_fluid_multiple import FluidVAE, make_bc_mask, port_cells, make_port_mask

def apply_physical_masks(soft_density, masks):
    """Solid -> 0, port/fluid -> 1, rest stays soft. Matches the density_clean
    construction inside simulate_soft, so volume measured here is consistent
    with the field the LBM actually sees."""
    d = soft_density.clamp(0.0, 1.0)
    d = torch.where(masks["solid_mask"], torch.zeros_like(d), d)
    d = torch.where(masks["fluid_mask"], torch.ones_like(d),  d)
    return d


def soft_connectivity_loss_single(prob, ports, device, n_iters=64, gain=3.0):
    """Differentiable inlet->outlet connectivity via soft flood fill.
    ~0 when connected, ->1 when disconnected, with a real gradient w.r.t. prob
    everywhere (tanh gating, no hard clamp) -- so it tells z HOW to reconnect,
    unlike a 0/1 BFS penalty."""
    inlets  = [p for p in ports if p["type"] == "inlet"]
    outlets = [p for p in ports if p["type"] == "outlet"]
    if not inlets or not outlets:
        return torch.zeros((), device=device)

    topo   = prob.unsqueeze(0).unsqueeze(0)                       # [1,1,Nx,Ny]
    kernel = torch.tensor([[0., 1., 0.],
                           [1., 0., 1.],
                           [0., 1., 0.]], device=device).view(1, 1, 3, 3) / 4.0
    outlet_mask = make_port_mask(ports, "outlet", device=device)
    n_out       = outlet_mask.sum().clamp(min=1.0)

    inlet_losses = []
    for inlet in inlets:
        seed = make_port_mask([inlet], "inlet", device=device)
        act  = seed * topo
        for _ in range(n_iters):
            spread = F.conv2d(act, kernel, padding=1)
            act    = torch.tanh(spread * topo * gain)
            act    = torch.maximum(act, seed * topo)
        reach = (act * outlet_mask).sum() / n_out
        inlet_losses.append(1.0 - reach)
    return torch.stack(inlet_losses).mean()

import torch.nn.functional as F   # ensure available at module level


def optimize_single(vae, bc_mask_tensor, masks, ports,
                    z_init, lambda_volume,
                    n_steps, lr,
                    temp_start, temp_end,
                    lambda_binary,
                    lambda_conn,
                    intermediate_dir, run_id,
                    vol_lo=0.20, vol_hi=0.21, save_every=10,
                    # ---- phase control ----
                    phase1_steps=40,
                    phase2_iters=30,
                    # ---- dp scaling ----
                    dp_ref=0.03,
                    w_dp_p1=1.0,
                    # ---- volume penalties ----
                    w_vol_floor=40.0,
                    vol_growth_target=0.30,
                    # ---- CONSTRICTION PENALTY (Fix 2) ----
                    w_constrict=5.0,        # weight on width penalty
                    width_target=0.6,       # desired local fluid fraction (0..1)
                    constrict_kernel=5,     # neighborhood size for local-width measure
                    # ---- walk parameters ----
                    alpha_init=1.0,
                    alpha_max=3.0,
                    alpha_min=0.05,
                    alpha_grow=1.2,
                    alpha_shrink=0.5,
                    max_backtrack=4,
                    perturb_scale=0.5,
                    n_perturb_tries=2,
                    # ---- connectivity / binary ----
                    grad_clip=5.0,
                    conn_iters=64,
                    plot_every=1):
    """
    Three-stage latent optimizer with constriction (width) penalty.

    Constriction penalty (Fix 2): penalizes fluid cells sitting in thin
    local neighborhoods. At fixed volume this drives material out of
    over-wide channels and into narrow throats, pushing toward uniform
    channel width and eliminating the junction-throat failure mode.

    Phase 1   — GROW:    reach connected, build channels (dp + width shaping).
    Phase 2A  — SQUEEZE: bring vol into band via gradient projection; the
                          protected gradient is now ∇(dp + width), so the
                          squeeze removes material without creating throats.
    Phase 2B  — WALK:    large-step latent walk minimizing (dp + width),
                          accepted only when feasible AND objective improves.
    """
    device = DEVICE

    is_designable = (~masks["solid_mask"] &
                     ~masks["fluid_mask"] &
                     ~masks["orifice_mask"])

    # availability mask for constriction normalization (1 = not solid wall)
    avail = (~masks["solid_mask"]).float()

    def constriction_penalty(soft_dens, topk=60):
            """
            Penalize ONLY the worst (thinnest) cells, not the whole-grid average.
            A localized throat dominates this metric; healthy channels don't
            dilute it. topk passes gradients to exactly the selected cells.
            """
            d = soft_dens.unsqueeze(0).unsqueeze(0)
            a = avail.unsqueeze(0).unsqueeze(0)
            k, p = constrict_kernel, constrict_kernel // 2
            fluid_sum = F.avg_pool2d(d * a, k, stride=1, padding=p)
            avail_sum = F.avg_pool2d(a,     k, stride=1, padding=p)
            local = (fluid_sum / avail_sum.clamp(min=1e-6))[0, 0]
            thinness = soft_dens * torch.clamp(width_target - local, min=0.0)
            # mean of the WORST k cells — concentrated on the throat
            vals, _ = torch.topk(thinness.flatten(), topk)
            return vals.mean()

    z         = torch.nn.Parameter(z_init.clone().to(device))
    optimizer = torch.optim.Adam([z], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase1_steps, eta_min=lr * 0.05)

    target_vol = (vol_lo + vol_hi) / 2.0
    vol_tol    = (vol_hi - vol_lo) / 2.0 + 0.005

    # Trackers store the lowest-OBJECTIVE design, but keep its raw dp for output.
    best_obj, best_dp, best_z, best_vol = float('inf'), float('inf'), z.detach().clone(), None
    any_obj,  any_dp,  any_z,  any_vol  = float('inf'), float('inf'), z.detach().clone(), None
    history = []

    # ============================================================
    # PHASE 1 — GROW (dp + width shaping)
    # ============================================================
    for step in range(phase1_steps):
        optimizer.zero_grad()

        progress    = step / max(phase1_steps - 1, 1)
        temperature = temp_start * (temp_end / temp_start) ** progress
        w_bin       = lambda_binary * progress

        logits       = vae.decode(z, bc_mask_tensor)
        soft_density = torch.sigmoid(logits[0, 0] / temperature)

        density_masked = apply_physical_masks(soft_density, masks)
        vol     = density_masked[is_designable].mean()
        vol_val = vol.item()
        bin_pen = (density_masked * (1.0 - density_masked)).mean()

        design_np = (density_masked.detach() > THRESHOLD).cpu().numpy()
        connected, _ = check_connectivity(design_np, ports)
        bin_vol = float(design_np[is_designable].mean())

        vol_deficit = torch.clamp(vol_growth_target - vol, min=0.0)
        vol_penalty = w_vol_floor * vol_deficit

        loss = vol_penalty + w_bin * bin_pen

        if connected:
            dp, _  = simulate_soft(soft_density, masks, ALPHA_MAX)
            dp_val = dp.item()
            constrict_pen = constriction_penalty(density_masked)
            constrict_val = constrict_pen.item()
            loss = loss + w_dp_p1 * (dp / dp_ref) + w_constrict * constrict_pen
            obj_val = dp_val / dp_ref + w_constrict * constrict_val
            conn_soft_val = 0.0
            mode = "CONNECTED"
        else:
            dp_val = float('nan')
            constrict_val = float('nan')
            obj_val = float('inf')
            conn_soft = soft_connectivity_loss_single(
                soft_density, ports, device, n_iters=conn_iters)
            loss = loss + lambda_conn * conn_soft
            conn_soft_val = conn_soft.item()
            mode = "RECONNECT"

        loss.backward()
        torch.nn.utils.clip_grad_norm_([z], max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        z.data.clamp_(-3.0, 3.0)

        if connected and np.isfinite(dp_val):
            if obj_val < any_obj:
                any_obj, any_dp, any_z, any_vol = obj_val, dp_val, z.detach().clone(), bin_vol
            if abs(bin_vol - target_vol) < vol_tol and obj_val < best_obj:
                best_obj, best_dp, best_z, best_vol = obj_val, dp_val, z.detach().clone(), bin_vol

        if (step % plot_every == 0):
            with torch.no_grad():
                vis = (torch.sigmoid(vae.decode(z.detach(), bc_mask_tensor)[0, 0])
                       > THRESHOLD).float().cpu().numpy()
            plot_design(vis, ports,
                        dp_val if np.isfinite(dp_val) else None, bin_vol, step,
                        title=f"[{run_id}] step {step} | PHASE1:grow | {mode}",
                        save_path=os.path.join(intermediate_dir,
                                               f"{run_id}_step{step:04d}.png"))

        dp_str = f"{dp_val:.5f}" if np.isfinite(dp_val) else "  --  "
        cn_str = f"{constrict_val:.4f}" if np.isfinite(constrict_val) else " -- "
        in_range_str = "OK" if vol_lo <= bin_vol <= vol_hi else "OOB"
        print(f"  [{run_id}] step {step:03d} | PHASE1:grow    | {mode:9s} | "
              f"T={temperature:.3f} | dp={dp_str} | constr={cn_str} | "
              f"vol={vol_val:.3f}(bin {bin_vol:.3f} {in_range_str}) | "
              f"vol_pen={vol_penalty.item():.4f}")

        history.append({
            'step': step, 'phase': 1,
            'dp': dp_val if np.isfinite(dp_val) else float('nan'),
            'constrict': constrict_val if np.isfinite(constrict_val) else float('nan'),
            'vol': vol_val, 'bin_vol': bin_vol,
            'conn': 0.0 if connected else 1.0, 'conn_soft': conn_soft_val,
            'vol_in_range': vol_lo <= bin_vol <= vol_hi,
            'temperature': temperature,
        })

    # ============================================================
    # PHASE 2A — SQUEEZE via gradient projection
    # The protected/combined gradient is now ∇(dp + width), so the squeeze
    # avoids creating throats while reducing volume.
    # ============================================================
    best_obj, best_dp, best_z, best_vol = float('inf'), float('inf'), z.detach().clone(), None
    any_obj,  any_dp,  any_z,  any_vol  = float('inf'), float('inf'), z.detach().clone(), None

    squeeze_steps = 50
    squeeze_optimizer = torch.optim.Adam([z], lr=lr * 0.5)
    w_dp_proj = 0.3   # share of PURE (dp+width)-reduction in combined gradient

    for sq_step in range(squeeze_steps):
        squeeze_optimizer.zero_grad()

        logits = vae.decode(z, bc_mask_tensor)
        soft_density = torch.sigmoid(logits[0, 0] / temp_end)
        density_masked = apply_physical_masks(soft_density, masks)
        vol = density_masked[is_designable].mean()
        vol_val = vol.item()

        design_np = (density_masked.detach() > THRESHOLD).cpu().numpy()
        connected, _ = check_connectivity(design_np, ports)
        bin_vol = float(design_np[is_designable].mean())

        vol_below = torch.clamp(vol_lo - vol, min=0.0)
        vol_above = torch.clamp(vol - vol_hi, min=0.0)
        vol_penalty = 200.0 * (vol_below + vol_above) + \
                       50.0 * (vol_below ** 2 + vol_above ** 2)

        if connected:
            dp, _ = simulate_soft(soft_density, masks, ALPHA_MAX)
            dp_val = dp.item()
            constrict_pen = constriction_penalty(density_masked)
            constrict_val = constrict_pen.item()

            # combined "quality" objective whose gradient we PROTECT
            quality = dp / dp_ref + w_constrict * constrict_pen
            obj_val = dp_val / dp_ref + w_constrict * constrict_val

            # g_vol — from volume penalty
            vol_penalty.backward(retain_graph=True)
            g_vol = z.grad.clone()
            squeeze_optimizer.zero_grad()

            # g_q — from (dp + width)  [this replaces the old g_dp]
            quality.backward()
            g_q = z.grad.clone()
            squeeze_optimizer.zero_grad()

            # Projection: Adam descends along -g_vol.
            # If g_vol·g_q < 0, then -g_vol aligns with +g_q -> quality WORSENS.
            # Project that harmful component out.
            dot = (g_vol * g_q).sum()
            g_q_norm_sq = (g_q * g_q).sum().clamp(min=1e-8)

            if dot.item() < 0:
                g_vol_safe = g_vol - (dot / g_q_norm_sq) * g_q
                proj_status = "PROJECTED"
            else:
                g_vol_safe = g_vol
                proj_status = "ALIGNED"

            combined_grad = (1.0 - w_dp_proj) * g_vol_safe + w_dp_proj * g_q
            z.grad = combined_grad
            mode = "CONNECTED"
        else:
            conn_soft = soft_connectivity_loss_single(
                soft_density, ports, device, n_iters=conn_iters)
            loss = vol_penalty + lambda_conn * conn_soft
            loss.backward()
            dp_val = float('nan')
            constrict_val = float('nan')
            obj_val = float('inf')
            proj_status = "RECONNECT"
            mode = "RECONNECT"

        torch.nn.utils.clip_grad_norm_([z], max_norm=grad_clip)
        squeeze_optimizer.step()
        z.data.clamp_(-3.0, 3.0)

        in_band = vol_lo <= bin_vol <= vol_hi
        if connected and np.isfinite(dp_val) and in_band:
            if obj_val < any_obj:
                any_obj, any_dp, any_z, any_vol = obj_val, dp_val, z.detach().clone(), bin_vol
            if obj_val < best_obj:
                best_obj, best_dp, best_z, best_vol = obj_val, dp_val, z.detach().clone(), bin_vol
                print(f"  [{run_id}] squeeze new best  dp={dp_val:.5f}  "
                      f"constr={constrict_val:.4f}  vol={bin_vol:.3f}  sq_step={sq_step}")

        dp_str = f"{dp_val:.5f}" if np.isfinite(dp_val) else "  --  "
        cn_str = f"{constrict_val:.4f}" if np.isfinite(constrict_val) else " -- "
        in_range_str = "OK" if in_band else "OOB"
        print(f"  [{run_id}] sq_step {sq_step:03d} | PHASE2A:squeeze | {mode:9s} | "
              f"{proj_status:9s} | dp={dp_str} | constr={cn_str} | "
              f"vol(bin {bin_vol:.3f} {in_range_str}) | vol_pen={vol_penalty.item():.4f}")

        history.append({
            'step': phase1_steps + sq_step, 'phase': '2A',
            'dp': dp_val if np.isfinite(dp_val) else float('nan'),
            'constrict': constrict_val if np.isfinite(constrict_val) else float('nan'),
            'vol': vol_val, 'bin_vol': bin_vol,
            'conn': 0.0 if connected else 1.0,
            'vol_in_range': in_band, 'proj_status': proj_status,
            'vol_penalty': vol_penalty.item(),
        })

    # ============================================================
    # PHASE 2B — gradient-guided latent WALK (minimize dp + width)
    # ============================================================
    walk_temp = temp_end

    def full_evaluate(z_tensor, with_grad=False):
        """
        Decode -> mask -> LBM. Returns:
          dp_t       (tensor/float or None if disconnected)
          obj_t      combined objective dp/dp_ref + w_constrict*constrict
                     (tensor if with_grad, else float; None if disconnected)
          vol_t, bin_v, conn, density, soft
        """
        if with_grad:
            z_tensor = z_tensor if z_tensor.requires_grad else \
                       z_tensor.clone().requires_grad_(True)
            logits = vae.decode(z_tensor, bc_mask_tensor)
        else:
            with torch.no_grad():
                logits = vae.decode(z_tensor, bc_mask_tensor)

        soft = torch.sigmoid(logits[0, 0] / walk_temp)
        density = apply_physical_masks(soft, masks)
        vol_t = density[is_designable].mean()

        design = (density.detach() > THRESHOLD).cpu().numpy()
        conn, _ = check_connectivity(design, ports)
        bin_v = float(design[is_designable].mean())

        if not conn:
            return None, None, vol_t, bin_v, False, density, None

        if with_grad:
            dp_t, _ = simulate_soft(soft, masks, ALPHA_MAX)
            constrict_t = constriction_penalty(density)
            obj_t = dp_t / dp_ref + w_constrict * constrict_t
        else:
            with torch.no_grad():
                dp_t, _ = simulate_soft(soft, masks, ALPHA_MAX)
                constrict_t = constriction_penalty(density)
                obj_t = (dp_t / dp_ref + w_constrict * constrict_t)

        return dp_t, obj_t, vol_t, bin_v, conn, density, soft

    # Starting point for the walk
    z_curr = z.detach().clone()
    dp_t, obj_t, vol_t, bin_v, conn, _, _ = full_evaluate(z_curr, with_grad=False)

    if not conn:
        print(f"  [{run_id}] WARNING: squeeze ended disconnected — walk cannot start")
        dp_curr, obj_curr = float('inf'), float('inf')
    else:
        dp_curr  = dp_t.item()
        obj_curr = obj_t.item()
        print(f"  [{run_id}] PHASE 2B starting — dp={dp_curr:.5f}, obj={obj_curr:.4f}, "
              f"vol={bin_v:.3f}, alpha_init={alpha_init}")

    alpha = alpha_init
    walk_step_offset = phase1_steps + squeeze_steps

    for walk_it in range(phase2_iters):
        global_step = walk_step_offset + walk_it
        if not conn:
            break

        # ---- 1. Gradient of OBJECTIVE (dp + width) at current z ----
        # z_grad = z_curr.clone().detach().requires_grad_(True)
        # dp_t, obj_t, vol_t, bin_v, conn, _, _ = full_evaluate(z_grad, with_grad=True)
        # if not conn:
        #     print(f"  [{run_id}] walk iter {walk_it} | current z disconnected — perturbing")
        #     g_norm = None
        # else:
        #     obj_t.backward()
        #     g = z_grad.grad.detach().clone()
        #     g_norm = g / (g.norm() + 1e-8)
        # ---- 1. VOLUME-PRESERVING objective gradient at current z ----
        z_grad = z_curr.clone().detach().requires_grad_(True)
        logits  = vae.decode(z_grad, bc_mask_tensor)
        soft    = torch.sigmoid(logits[0, 0] / walk_temp)
        density = apply_physical_masks(soft, masks)
        vol_t   = density[is_designable].mean()
        design  = (density.detach() > THRESHOLD).cpu().numpy()
        conn, _ = check_connectivity(design, ports)

        if not conn:
            print(f"  [{run_id}] walk iter {walk_it} | current z disconnected — perturbing")
            g_norm = None
        else:
            dp_t        = simulate_soft(soft, masks, ALPHA_MAX)[0]
            constrict_t = constriction_penalty(density)
            obj_t       = dp_t / dp_ref + w_constrict * constrict_t

            # g_obj — direction that changes the objective
            obj_t.backward(retain_graph=True)
            g_obj = z_grad.grad.clone()
            z_grad.grad.zero_()

            # g_vol — direction that changes volume
            vol_t.backward()
            g_vol = z_grad.grad.clone()

            # Project g_obj onto the constant-volume manifold:
            #   g_walk = g_obj - (g_obj·g_vol / |g_vol|²) g_vol
            # Moving along -g_walk reduces obj WITHOUT changing volume (1st order),
            # so large alpha steps stay in-band and become acceptable.
            gv_norm_sq = (g_vol * g_vol).sum().clamp(min=1e-8)
            dot        = (g_obj * g_vol).sum()
            g_walk     = g_obj - (dot / gv_norm_sq) * g_vol
            g_norm     = g_walk / (g_walk.norm() + 1e-8)

        # ---- 2. Backtracking line search (accept on OBJECTIVE improvement) ----
        accepted = False
        trial_alpha = alpha
        attempted = []

        if g_norm is not None:
            for bt in range(max_backtrack):
                z_trial = (z_curr - trial_alpha * g_norm).clamp(-3.0, 3.0)
                dp_t, obj_t, vol_t, bin_v, conn_t, _, _ = full_evaluate(z_trial, with_grad=False)
                vol_v = bin_v
                in_band = vol_lo <= vol_v <= vol_hi
                in_band_loose = abs(vol_v - target_vol) < vol_tol

                obj_trial = obj_t.item() if obj_t is not None else None
                attempted.append({'alpha': trial_alpha,
                                   'dp': dp_t.item() if dp_t is not None else None,
                                   'obj': obj_trial, 'vol': vol_v,
                                   'conn': conn_t, 'in_band': in_band_loose})

                if (conn_t and in_band_loose and obj_trial is not None
                        and obj_trial < obj_curr):
                    z_curr   = z_trial.detach().clone()
                    dp_curr  = dp_t.item()
                    obj_curr = obj_trial
                    alpha = min(alpha * alpha_grow, alpha_max)
                    accepted = True

                    if obj_curr < any_obj:
                        any_obj, any_dp, any_z, any_vol = obj_curr, dp_curr, z_curr.clone(), vol_v
                    if in_band and obj_curr < best_obj:
                        best_obj, best_dp, best_z, best_vol = obj_curr, dp_curr, z_curr.clone(), vol_v
                        print(f"  [{run_id}] new best  dp={dp_curr:.5f}  obj={obj_curr:.4f}  "
                              f"vol={vol_v:.3f}  walk_iter={walk_it} alpha={trial_alpha:.3f}")
                    break

                trial_alpha *= alpha_shrink

        # ---- 3. Perturbation fallback ----
        if not accepted:
            for pt in range(n_perturb_tries):
                z_pert = (z_curr + torch.randn_like(z_curr) * perturb_scale).clamp(-3.0, 3.0)
                dp_t, obj_t, vol_t, bin_v, conn_t, _, _ = full_evaluate(z_pert, with_grad=False)
                vol_v = bin_v
                in_band_loose = abs(vol_v - target_vol) < vol_tol
                obj_trial = obj_t.item() if obj_t is not None else None

                if (conn_t and in_band_loose and obj_trial is not None
                        and obj_trial < obj_curr):
                    z_curr   = z_pert.detach().clone()
                    dp_curr  = dp_t.item()
                    obj_curr = obj_trial
                    accepted = True

                    if obj_curr < any_obj:
                        any_obj, any_dp, any_z, any_vol = obj_curr, dp_curr, z_curr.clone(), vol_v
                    if vol_lo <= vol_v <= vol_hi and obj_curr < best_obj:
                        best_obj, best_dp, best_z, best_vol = obj_curr, dp_curr, z_curr.clone(), vol_v
                        print(f"  [{run_id}] new best (perturb)  dp={dp_curr:.5f}  "
                              f"obj={obj_curr:.4f}  vol={vol_v:.3f}  walk_iter={walk_it}")
                    break
            alpha = max(alpha * alpha_shrink, alpha_min)

        # ---- Logging + plotting ----
        with torch.no_grad():
            _, _, vol_t, bin_v, conn, _, _ = full_evaluate(z_curr, with_grad=False)
        in_range_str = "OK" if vol_lo <= bin_v <= vol_hi else "OOB"
        dp_str  = f"{dp_curr:.5f}"  if np.isfinite(dp_curr)  else "  --  "
        obj_str = f"{obj_curr:.4f}" if np.isfinite(obj_curr) else " -- "
        accept_str = "ACCEPT" if accepted else "REJECT"
        print(f"  [{run_id}] iter {walk_it:03d} | PHASE2B:walk   | {accept_str:6s} | "
              f"alpha={alpha:.3f} | dp={dp_str} | obj={obj_str} | "
              f"vol(bin {bin_v:.3f} {in_range_str}) | backtracks={len(attempted)}")

        if walk_it % plot_every == 0 or walk_it == phase2_iters - 1:
            with torch.no_grad():
                vis = (torch.sigmoid(vae.decode(z_curr, bc_mask_tensor)[0, 0] / walk_temp)
                       > THRESHOLD).float().cpu().numpy()
            plot_design(vis, ports, dp_curr if np.isfinite(dp_curr) else None,
                        bin_v, global_step,
                        title=f"[{run_id}] walk_iter {walk_it} | dp={dp_curr:.5f}",
                        save_path=os.path.join(intermediate_dir,
                                               f"{run_id}_step{global_step:04d}.png"))

        history.append({
            'step': global_step, 'phase': 2,
            'dp': dp_curr if np.isfinite(dp_curr) else float('nan'),
            'obj': obj_curr if np.isfinite(obj_curr) else float('nan'),
            'vol': bin_v, 'bin_vol': bin_v,
            'conn': 0.0 if conn else 1.0,
            'vol_in_range': vol_lo <= bin_v <= vol_hi,
            'alpha': alpha, 'accepted': accepted,
            'n_backtracks': len(attempted), 'temperature': walk_temp,
        })

    # ============================================================
    # Return best
    # ============================================================
    if best_vol is not None:
        final_z, final_dp, final_vol = best_z, best_dp, best_vol
    elif any_vol is not None:
        print(f"  [{run_id}] WARNING: never held vol in [{vol_lo:.2f}, {vol_hi:.2f}]; "
              f"returning lowest-obj connected design (vol={any_vol:.3f}).")
        final_z, final_dp, final_vol = any_z, any_dp, any_vol
    else:
        print(f"  [{run_id}] WARNING: never connected; returning last z.")
        final_z, final_dp, final_vol = z.detach().clone(), float('nan'), None

    return {
        'best_z':  final_z.squeeze(0).cpu().numpy(),
        'best_dp': final_dp,
        'best_vol': final_vol,
        'history': history,
        'run_id':  run_id,
        'phase_reached': 2,
    }


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


# MAIN OPTIMIZER WITH MULTIPLE RESTARTS

def run_latent_grad(ports,
                    vae_path      = "vae_best_new.pth",
                    n_restarts    = 5,
                    n_steps       = 80,
                    lr            = 0.05,
                    lambda_volume = 0.5,
                    lambda_binary = 2.0,
                    temp_start    = 1.0,
                    temp_end      = 0.1):

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

    run_tag          = ports_to_tag(ports) + f"_lv{lambda_volume}"
    intermediate_dir = f"./latgrad_intermediates/{run_tag}"
    os.makedirs(intermediate_dir, exist_ok=True)

    # Load VAE 
    vae = FluidVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(
        torch.load(vae_path, map_location=DEVICE, weights_only=True))
    vae.eval()

    # Freeze decoder weights — only z is optimized
    for param in vae.parameters():
        param.requires_grad_(False)

    # BC setup 
    bc_mask_np     = make_bc_mask(ports)
    bc_mask_tensor = torch.FloatTensor(bc_mask_np).unsqueeze(0).to(DEVICE)
    masks          = build_bc_masks(Nx, Ny, WALL, ports)

    # Run restarts 
    restart_results = []

    for restart in range(n_restarts):
        print(f"\nRestart {restart+1}/{n_restarts}")

        # Each restart uses a different random z initialization
        # Initialization strategy:
        #   restart 0: z = 0  (prior mean — safe, near-average topology)
        #   restart 1: z ~ N(0,1)  (random sample from prior)
        #   restart 2+: z ~ N(0, 0.5)  (tighter sampling, explore near-prior)
        if restart == 0:
            z_init = torch.zeros(1, LATENT_DIM)
            #z_init = (torch.randn(1, LATENT_DIM) * 0.5).clamp(-3, 3)
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
            lambda_volume  = 100,
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

    # Select best across restarts 
    feasible = [r for r in restart_results if r['connected']]

    if not feasible:
        print("\nNo feasible binary topology found across all restarts.")
        print("Consider: more restarts, higher temp_end, lower lambda_binary")
        return None

    # Best = lowest dp on binary evaluation
    best = min(feasible, key=lambda r: r['dp_binary'])

    print(f"Best result: restart={best['run_id']} | "
          f"dp={best['dp_binary']:.5f} | vol={best['vol_binary']:.3f}")
    print(f"Soft dp was: {best['best_dp']:.5f} | "
          f"Gap: {best['dp_binary'] - best['best_dp']:+.5f}")

    # Save outputs 

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

    print(f"{'Restart':<10} {'Best soft dp':<15} {'Binary dp':<12} "
          f"{'Vol':<8} {'Connected'}")
    
    for r in restart_results:
        dp_b = f"{r['dp_binary']:.5f}" if r['connected'] else "N/A"
        vol  = f"{r['vol_binary']:.3f}" if r['connected'] else "N/A"
        mark = "✓" if r['connected'] else "✗"
        best_mark = " ← best" if r == best else ""
        print(f"  {r['run_id']:<8} {r['best_dp']:<15.5f} {dp_b:<12} "
              f"{vol:<8} {mark}{best_mark}")


    return best['best_z'], best['design_np'], best['dp_binary'], best['vol_binary']


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
    parser.add_argument('--temp_end',      type=float, default=0.25,
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

    best_z, best_design_np, best_design_bin, best_vol =run_latent_grad(
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