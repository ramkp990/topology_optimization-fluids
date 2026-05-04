"""
fluid_latent_analysis.py
Latent space analysis for the fluid CVAE.

Produces:
  1. UMAP coloured by pressure_drop and volume_fraction
  2. UMAP coloured by port configuration properties
  3. Pearson correlation heatmap (latent dims vs scalar properties)
  4. Per-sample reconstruction loss vs pressure_drop
  5. UMAP coloured by reconstruction loss
  6. CMA-ES trajectory on UMAP (if npz files exist)
"""


import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import pearsonr
import glob
import os
import umap

from vae_fluid_multiple import FluidVAE, make_bc_mask, make_loaders, is_feasible_vae

# ── config ────────────────────────────────────────────────────────────────
VAE_PATH         = "vae_best_new.pth"
H5_PATH          = "./data/new1/dataset_all_merged.h5"
CMAES_RESULT_DIR = "cmaes_intermediates_fluid"
OUT_DIR          = "analysis/fluid_latent"
LATENT_DIM       = 32
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE       = 256
# ──────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


# ── load model ────────────────────────────────────────────────────────────
model = FluidVAE(latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"Loaded VAE from {VAE_PATH}")


# ── load dataset ─────────────────────────────────────────────────────────
train_loader, val_loader, test_loader, ds = make_loaders(
    H5_PATH, batch_size=BATCH_SIZE, seed=42
)


# pull everything out of the dataset directly
all_densities    = []
all_pressure     = []
all_volume       = []
all_ports        = []
all_bc_masks     = []

print("Loading all samples from dataset...")
for i in range(len(ds)):
    density, bc_mask, metrics, ports = ds[i]
    all_densities.append(density.numpy())      # [1, 64, 64]
    all_bc_masks.append(bc_mask.numpy())       # [2, 64, 64]
    all_pressure.append(metrics[0].item())     # pressure_drop
    all_volume.append(metrics[1].item())       # volume_fraction
    all_ports.append(ports)

all_densities = np.stack(all_densities)        # [N, 1, 64, 64]
all_bc_masks  = np.stack(all_bc_masks)         # [N, 2, 64, 64]
pressure      = np.array(all_pressure)         # [N]
volume        = np.array(all_volume)           # [N]
N             = len(pressure)

print(f"  {N} designs loaded")
print(f"  Pressure: [{pressure.min():.4f}, {pressure.max():.4f}]")
print(f"  Volume:   [{volume.min():.3f},  {volume.max():.3f}]")


# ── extract port scalar features ─────────────────────────────────────────
# For each design: n_inlets, n_outlets, whether any inlet is on top/bottom
# (left/right = horizontal walls, top/bottom = vertical walls)
# These give us something to correlate with latent dims

n_inlets_arr    = np.zeros(N, dtype=np.float32)
#n_outlets_arr   = np.zeros(N, dtype=np.float32)
inlet_wall_h    = np.zeros(N, dtype=np.float32)  # 1 if any inlet on left/right
outlet_wall_h   = np.zeros(N, dtype=np.float32)  # 1 if outlet on left/right
inlet_center_y  = np.zeros(N, dtype=np.float32)  # mean inlet center position
outlet_center_y = np.zeros(N, dtype=np.float32)  # mean outlet center position

for i, ports in enumerate(all_ports):
    inlets  = [p for p in ports if p["type"] == "inlet"]
    outlets = [p for p in ports if p["type"] == "outlet"]

    n_inlets_arr[i]  = len(inlets)
    #n_outlets_arr[i] = len(outlets)

    if inlets:
        inlet_wall_h[i]   = float(any(p["wall"] in ["left","right"] for p in inlets))
        inlet_center_y[i] = np.mean([p["center"] for p in inlets])

    if outlets:
        outlet_wall_h[i]   = float(any(p["wall"] in ["left","right"] for p in outlets))
        outlet_center_y[i] = np.mean([p["center"] for p in outlets])

port_feature_names = [
    "n_inlets",# "n_outlets",
    "inlet_on_horiz_wall", "outlet_on_horiz_wall",
    "inlet_center_pos", "outlet_center_pos"
]
port_features = np.column_stack([
    n_inlets_arr, #n_outlets_arr,
    inlet_wall_h, outlet_wall_h,
    inlet_center_y, outlet_center_y
])


# ── encode all designs ────────────────────────────────────────────────────
print("Encoding all designs to latent space...")
all_mu = []

density_tensor = torch.tensor(all_densities).float()
bc_tensor      = torch.tensor(all_bc_masks).float()

with torch.no_grad():
    for i in range(0, N, BATCH_SIZE):
        d  = density_tensor[i:i+BATCH_SIZE].to(DEVICE)
        bc = bc_tensor[i:i+BATCH_SIZE].to(DEVICE)
        mu, _ = model.encode(d, bc)
        all_mu.append(mu.cpu().numpy())

mu_all = np.concatenate(all_mu, axis=0)   # [N, 32]
# filter to one BC family
left_right_mask = np.array([
    any(p["wall"] == "left" for p in ports if p["type"] == "inlet") and
    any(p["wall"] == "right" for p in ports if p["type"] == "outlet")
    for ports in all_ports
])

mu_subset      = mu_all[left_right_mask]
pressure   = pressure[left_right_mask]
volume     = volume[left_right_mask]

np.save(os.path.join(OUT_DIR, "mu_all.npy"), mu_subset)
print(f"  Encoded → μ shape {mu_subset.shape}")




# ── UMAP ──────────────────────────────────────────────────────────────────
print("Running UMAP...")
reducer = umap.UMAP(n_components=2, n_neighbors=15,
                    min_dist=0.1, random_state=42)
reducer = umap.UMAP(n_components=2, n_neighbors=15,
                        min_dist=0.1, random_state=42)
umap_2d = reducer.fit_transform(mu_all)
umap_2d = reducer.fit_transform(mu_subset)
np.save(os.path.join(OUT_DIR, "umap_2d.npy"), umap_2d)
print("  UMAP done")


# ── plotting helper ───────────────────────────────────────────────────────
def scatter(coords, color_vals, title, fname, cmap="plasma",
            vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=color_vals, cmap=cmap, s=2, alpha=0.6,
                    rasterized=True, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label=title)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ── UMAP coloured by physics ──────────────────────────────────────────────
print("\nPlotting UMAP by physics properties...")
scatter(umap_2d, pressure, "Pressure drop (l.u.)",
        "umap_pressure.png", cmap="plasma_r")
scatter(umap_2d, volume,   "Volume fraction",
        "umap_volume.png",   cmap="viridis")

# log-scale pressure — useful since range is [0.005, 0.174]
scatter(umap_2d, np.log10(pressure), "log₁₀(Pressure drop)",
        "umap_pressure_log.png", cmap="plasma_r")

# ── UMAP coloured by port properties ─────────────────────────────────────
print("Plotting UMAP by port properties...")
scatter(umap_2d, n_inlets_arr,    "Number of inlets",
        "umap_n_inlets.png",    cmap="coolwarm")
scatter(umap_2d, inlet_wall_h,    "Inlet on horizontal wall",
        "umap_inlet_horiz.png", cmap="RdBu")
scatter(umap_2d, outlet_wall_h,   "Outlet on horizontal wall",
        "umap_outlet_horiz.png",cmap="RdBu")
scatter(umap_2d, inlet_center_y,  "Inlet center position",
        "umap_inlet_center.png",cmap="coolwarm")
scatter(umap_2d, outlet_center_y, "Outlet center position",
        "umap_outlet_center.png",cmap="coolwarm")


# ── Pearson correlation: latent dims vs all scalars ───────────────────────
print("\nComputing Pearson correlations...")

targets      = np.column_stack([port_features, pressure, volume])
target_names = port_feature_names + ["pressure_drop", "volume_frac"]

N_targets = len(target_names)
corr_matrix = np.zeros((LATENT_DIM, N_targets))
pval_matrix = np.zeros((LATENT_DIM, N_targets))

for z in range(LATENT_DIM):
    for t in range(N_targets):
        r, p = pearsonr(mu_all[:, z], targets[:, t])
        corr_matrix[z, t] = r
        pval_matrix[z, t] = p

np.save(os.path.join(OUT_DIR, "corr_matrix.npy"), corr_matrix)

# heatmap
fig, ax = plt.subplots(figsize=(max(12, N_targets * 0.8), 10))
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Pearson r")

ax.set_xticks(range(N_targets))
ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(LATENT_DIM))
ax.set_yticklabels([f"z{i}" for i in range(LATENT_DIM)], fontsize=7)
ax.set_xlabel("Property")
ax.set_ylabel("Latent dimension")
ax.set_title("Pearson correlation: latent dims vs design properties")

sig = pval_matrix < 0.01
for zi in range(LATENT_DIM):
    for ti in range(N_targets):
        if sig[zi, ti]:
            ax.plot(ti, zi, "k.", markersize=2)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "disentanglement_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved disentanglement_heatmap.png")

# print top correlations
abs_corr = np.abs(corr_matrix)
flat_idx = np.argsort(abs_corr.ravel())[::-1][:20]

print("\nTop 20 (latent dim, property) correlations:")
print(f"{'Rank':<5} {'z_dim':<8} {'property':<25} {'r':>8} {'p':>12}")
print("-" * 60)
for rank, idx in enumerate(flat_idx):
    zi, ti = divmod(idx, N_targets)
    print(f"{rank+1:<5} z{zi:<7} {target_names[ti]:<25} "
          f"{corr_matrix[zi,ti]:>8.4f} {pval_matrix[zi,ti]:>12.2e}")

print("\nBest latent dimension per property:")
for ti, name in enumerate(target_names):
    best_z = int(np.argmax(np.abs(corr_matrix[:, ti])))
    r_val  = corr_matrix[best_z, ti]
    print(f"  {name:<25} → z{best_z:<3}  r={r_val:+.4f}")


# ── per-sample reconstruction loss ───────────────────────────────────────
print("\nComputing per-sample reconstruction loss...")
recon_losses = []

with torch.no_grad():
    for i in range(0, N, BATCH_SIZE):
        d  = density_tensor[i:i+BATCH_SIZE].to(DEVICE)
        bc = bc_tensor[i:i+BATCH_SIZE].to(DEVICE)
        recon, _, _ = model(d, bc)
        recon_prob  = torch.sigmoid(recon)
        loss = F.binary_cross_entropy(
            recon_prob, d, reduction="none"
        ).mean(dim=[1, 2, 3])
        recon_losses.append(loss.cpu().numpy())

recon_losses = np.concatenate(recon_losses)
np.save(os.path.join(OUT_DIR, "recon_losses.npy"), recon_losses)

# plot recon loss vs pressure drop
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(np.log10(pressure), recon_losses,
                c=volume, cmap="viridis",
                s=3, alpha=0.5, rasterized=True)
plt.colorbar(sc, ax=ax, label="Volume fraction")
ax.set_xlabel("log₁₀(Pressure drop)")
ax.set_ylabel("Reconstruction loss (BCE)")
ax.set_title("Reconstruction difficulty vs pressure drop\n"
             "(colour = volume fraction)")

sort_idx    = np.argsort(np.log10(pressure))
lp_s        = np.log10(pressure)[sort_idx]
loss_s      = recon_losses[sort_idx]
window      = max(1, len(lp_s) // 50)
running_med = np.array([
    np.median(loss_s[max(0, i-window):i+window])
    for i in range(len(loss_s))
])
ax.plot(lp_s, running_med, color="red", linewidth=1.5, label="Running median")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "recon_vs_pressure.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved recon_vs_pressure.png")

# UMAP side-by-side: pressure drop vs recon loss
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sc1 = axes[0].scatter(umap_2d[:, 0], umap_2d[:, 1],
                       c=np.log10(pressure), cmap="plasma_r",
                       s=2, alpha=0.5, rasterized=True)
plt.colorbar(sc1, ax=axes[0], label="log₁₀(Pressure drop)")
axes[0].set_title("UMAP — coloured by pressure drop")
axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

sc2 = axes[1].scatter(umap_2d[:, 0], umap_2d[:, 1],
                       c=recon_losses, cmap="hot_r",
                       s=2, alpha=0.5, rasterized=True)
plt.colorbar(sc2, ax=axes[1], label="Reconstruction loss")
axes[1].set_title("UMAP — coloured by reconstruction loss")
axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "umap_pressure_vs_recon.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved umap_pressure_vs_recon.png")


# ── feasibility check on reconstructions ─────────────────────────────────
print("\nChecking reconstruction feasibility on full dataset...")
feasible_count = 0
violation_types = {}

with torch.no_grad():
    for i in range(0, N, BATCH_SIZE):
        d  = density_tensor[i:i+BATCH_SIZE].to(DEVICE)
        bc = bc_tensor[i:i+BATCH_SIZE].to(DEVICE)
        recon, _, _ = model(d, bc)
        recon_prob  = torch.sigmoid(recon)

        for j in range(recon_prob.shape[0]):
            recon_np = recon_prob[j, 0].cpu().numpy()
            ports    = all_ports[i + j]
            ok, reason = is_feasible_vae(recon_np, ports)
            if ok:
                feasible_count += 1
            else:
                violation_types[reason] = violation_types.get(reason, 0) + 1

pct = 100 * feasible_count / N
print(f"  Feasibility: {feasible_count}/{N} ({pct:.1f}%)")
for vtype, count in violation_types.items():
    print(f"    {vtype}: {count}")


# ── CMA-ES trajectory ─────────────────────────────────────────────────────
npz_files = sorted(glob.glob(
    os.path.join(CMAES_RESULT_DIR, "**/*.npz"), recursive=True))
if not npz_files:
    npz_files = sorted(glob.glob(
        os.path.join(CMAES_RESULT_DIR, "*.npz")))

if not npz_files:
    print("\nNo CMA-ES npz files found — skipping trajectory plot.")
    print(f"  Expected npz files with 'best_z' key in: {CMAES_RESULT_DIR}")
else:
    traj_zs = []
    traj_dp = []
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        if "best_z" in d:
            traj_zs.append(d["best_z"])
            # compliance/pressure stored alongside if available
            if "pressure_drop" in d:
                traj_dp.append(float(d["pressure_drop"]))
            elif "compliance" in d:
                traj_dp.append(float(d["compliance"]))
            else:
                traj_dp.append(np.nan)

    if traj_zs:
        traj_zs = np.stack(traj_zs)
        traj_2d = reducer.transform(traj_zs)

        fig, ax = plt.subplots(figsize=(8, 7))

        sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1],
                        c=np.log10(pressure), cmap="plasma_r",
                        s=2, alpha=0.3, rasterized=True, zorder=1)
        plt.colorbar(sc, ax=ax, label="log₁₀(Pressure drop)")

        n_steps = len(traj_2d)
        colors  = cm.cool(np.linspace(0, 1, n_steps))
        ax.plot(traj_2d[:, 0], traj_2d[:, 1],
                color="white", linewidth=1.5, zorder=2, alpha=0.8)
        ax.scatter(traj_2d[:, 0], traj_2d[:, 1],
                   c=colors, s=40, edgecolors="k",
                   linewidths=0.5, zorder=3)

        ax.scatter(*traj_2d[0],  s=150, marker="*", color="cyan",
                   edgecolors="k", linewidths=0.8, zorder=4, label="Start")
        ax.scatter(*traj_2d[-1], s=120, marker="D", color="lime",
                   edgecolors="k", linewidths=0.8, zorder=4, label="Best found")

        ax.set_title("CMA-ES trajectory in UMAP latent space\n"
                     "(background = training designs, colour = log pressure drop)")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "umap_cmaes_trajectory.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved umap_cmaes_trajectory.png  ({n_steps} steps)")
    else:
        print("  No 'best_z' keys found in npz files.")


print(f"\nAll outputs saved to {OUT_DIR}")