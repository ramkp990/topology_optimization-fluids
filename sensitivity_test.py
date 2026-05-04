import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vae_fluid_multiple import FluidVAE, make_bc_mask
from new_generate_dataset_multiple import sample_ports, build_bc_masks

# =========================================================
# CONFIG
# =========================================================
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
VAE_PATH   = "vae_best_new.pth"
LATENT_DIM = 32
Nx, Ny     = 64, 64
WALL       = 4
N_SAMPLES  = 100   # z vectors to sample per BC config
N_BC_CONFIGS = 8   # number of different BC configs to test across


# =========================================================
# LOAD MODEL
# =========================================================
def load_model(path, latent_dim, device):
    model = FluidVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded VAE from {path}")
    return model


# =========================================================
# TEST 1: OUTPUT STD ACROSS Z SAMPLES
# How much does z actually vary the output for a fixed BC?
# =========================================================
def test_output_diversity(model, bc_mask_tensor, device, n=N_SAMPLES, label=""):
    """
    Sample n random z vectors, decode each, measure pixel-level
    variance across all outputs.
    
    Healthy: mean_std > 0.10  (z meaningfully controls output)
    Dead:    mean_std < 0.05  (all outputs look the same)
    """
    outputs = []
    with torch.no_grad():
        for _ in range(n):
            z   = torch.randn(1, model.latent_dim, device=device)
            out = torch.sigmoid(model.decode(z, bc_mask_tensor))
            outputs.append(out.squeeze().cpu().numpy())

    stack    = np.stack(outputs)          # [N, 64, 64]
    mean_img = stack.mean(axis=0)
    std_img  = stack.std(axis=0)

    mean_std = std_img.mean()
    max_std  = std_img.max()
    mean_vol = (stack > 0.5).mean()

    status = "RICH" if mean_std > 0.10 else ("MARGINAL" if mean_std > 0.05 else "DEAD")
    print(f"  [{label}] Output diversity: mean_std={mean_std:.4f}  "
          f"max_std={max_std:.4f}  mean_vol={mean_vol:.3f}  [{status}]")

    return stack, std_img, mean_std


# =========================================================
# TEST 2: PAIRWISE Z SENSITIVITY
# How different are two random outputs on average?
# =========================================================
def test_pairwise_sensitivity(model, bc_mask_tensor, device, n=N_SAMPLES, label=""):
    """
    Sample n pairs of random z, compare outputs.
    This is more robust than the single z=0 vs z=rand test.
    
    Healthy: mean_diff > 0.08
    Dead:    mean_diff < 0.03
    """
    diffs = []
    with torch.no_grad():
        for _ in range(n):
            z1   = torch.randn(1, model.latent_dim, device=device)
            z2   = torch.randn(1, model.latent_dim, device=device)
            out1 = torch.sigmoid(model.decode(z1, bc_mask_tensor))
            out2 = torch.sigmoid(model.decode(z2, bc_mask_tensor))
            diffs.append((out1 - out2).abs().mean().item())

    mean_diff = np.mean(diffs)
    p10       = np.percentile(diffs, 10)
    p90       = np.percentile(diffs, 90)

    status = "RICH" if mean_diff > 0.08 else ("MARGINAL" if mean_diff > 0.03 else "DEAD")
    print(f"  [{label}] Pairwise diff:    mean={mean_diff:.4f}  "
          f"p10={p10:.4f}  p90={p90:.4f}  [{status}]")

    return mean_diff, diffs


# =========================================================
# TEST 3: LATENT DIMENSION ACTIVITY
# Which z dimensions are actually being used?
# =========================================================
def test_latent_activity(model, val_loader, device, n_batches=5):
    """
    Run the encoder on real data and inspect the posterior.
    
    Active dim:   std(mu_i) >> 1  — encoder varies this dimension
    Inactive dim: std(mu_i) ≈ 0  — encoder ignores this dimension
    
    Healthy: most dims active (std > 0.1)
    Dead:    most dims inactive (std ≈ 0)
    """
    all_mu     = []
    all_logvar = []

    model.eval()
    with torch.no_grad():
        for i, (density, bc_mask, _, _) in enumerate(val_loader):
            if i >= n_batches:
                break
            density = density.to(device)
            bc_mask = bc_mask.to(device)
            mu, logvar = model.encode(density, bc_mask)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())

    all_mu     = np.concatenate(all_mu,     axis=0)   # [N, latent_dim]
    all_logvar = np.concatenate(all_logvar, axis=0)

    mu_std     = all_mu.std(axis=0)       # [latent_dim] — how much each dim varies
    mean_var   = np.exp(all_logvar).mean(axis=0)  # average posterior variance

    n_active   = (mu_std > 0.1).sum()
    n_total    = model.latent_dim

    print(f"\n  Latent dimension activity ({n_active}/{n_total} active):")
    print(f"  mu_std:    min={mu_std.min():.4f}  "
          f"mean={mu_std.mean():.4f}  max={mu_std.max():.4f}")
    print(f"  post_var:  min={mean_var.min():.4f}  "
          f"mean={mean_var.mean():.4f}  max={mean_var.max():.4f}")

    # Print per-dim activity
    print(f"\n  Per-dim mu_std (sorted):")
    sorted_idx = np.argsort(mu_std)[::-1]
    for rank, idx in enumerate(sorted_idx):
        bar    = "█" * int(mu_std[idx] * 40)
        status = "✅" if mu_std[idx] > 0.1 else "💀"
        print(f"    dim {idx:2d}: {mu_std[idx]:.4f} {bar} {status}")

    return mu_std, mean_var


# =========================================================
# TEST 4: INTERPOLATION SMOOTHNESS
# Does z-space interpolate smoothly between two designs?
# Good VAE: smooth transition. Collapsed: no change.
# =========================================================
def test_interpolation(model, bc_mask_tensor, device, n_steps=10, label=""):
    """
    Linearly interpolate z from z1 to z2 and measure
    how much the output changes at each step.
    
    Healthy: smooth monotonic change
    Dead:    output barely changes along the path
    """
    with torch.no_grad():
        z1 = torch.randn(1, model.latent_dim, device=device)
        z2 = torch.randn(1, model.latent_dim, device=device)

        step_diffs = []
        prev_out   = None

        for t in np.linspace(0, 1, n_steps):
            z   = (1 - t) * z1 + t * z2
            out = torch.sigmoid(model.decode(z, bc_mask_tensor))
            out_np = out.squeeze().cpu().numpy()

            if prev_out is not None:
                step_diffs.append(abs(out_np - prev_out).mean())
            prev_out = out_np

    mean_step = np.mean(step_diffs)
    status = "SMOOTH" if mean_step > 0.01 else "FLAT"
    print(f"  [{label}] Interpolation:    mean_step_diff={mean_step:.4f}  [{status}]")
    return step_diffs


# =========================================================
# PLOTTING
# =========================================================
def plot_diversity_grid(stacks, std_imgs, bc_labels,
                        save_path="sensitivity_diversity.png"):
    """
    For each BC config: show mean output + std heatmap + 4 random samples.
    """
    n_configs = len(stacks)
    cols      = 2 + 4   # mean, std, 4 samples
    fig, axes = plt.subplots(n_configs, cols,
                             figsize=(cols * 2.5, n_configs * 2.5))

    if n_configs == 1:
        axes = axes[np.newaxis, :]

    for row, (stack, std_img, label) in enumerate(zip(stacks, std_imgs, bc_labels)):
        mean_img = stack.mean(axis=0)

        axes[row, 0].imshow(mean_img.T, cmap='gray_r', origin='lower',
                            vmin=0, vmax=1)
        axes[row, 0].set_title(f"Mean\n{label}", fontsize=6)
        axes[row, 0].axis('off')

        im = axes[row, 1].imshow(std_img.T, cmap='hot', origin='lower',
                                 vmin=0, vmax=0.3)
        axes[row, 1].set_title(f"Std\n(max={std_img.max():.3f})", fontsize=6)
        axes[row, 1].axis('off')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046)

        sample_idxs = np.random.choice(len(stack), 4, replace=False)
        for col, idx in enumerate(sample_idxs):
            axes[row, 2 + col].imshow(stack[idx].T, cmap='gray_r',
                                       origin='lower', vmin=0, vmax=1)
            axes[row, 2 + col].set_title(f"Sample {col+1}", fontsize=6)
            axes[row, 2 + col].axis('off')

    plt.suptitle("VAE Latent Sensitivity: Mean | Std | Random Samples", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary(all_mean_stds, all_mean_diffs, bc_labels,
                 save_path="sensitivity_summary.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = range(len(bc_labels))

    axes[0].bar(x, all_mean_stds, color='steelblue', alpha=0.8)
    axes[0].axhline(0.10, color='green', linestyle='--', label='target (0.10)')
    axes[0].axhline(0.05, color='red',   linestyle='--', label='collapse (0.05)')
    axes[0].set_xticks(x); axes[0].set_xticklabels(bc_labels, rotation=30, ha='right', fontsize=7)
    axes[0].set_title('Output Std Across z Samples (per BC config)')
    axes[0].set_ylabel('Mean pixel std'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].bar(x, all_mean_diffs, color='darkorange', alpha=0.8)
    axes[1].axhline(0.08, color='green', linestyle='--', label='target (0.08)')
    axes[1].axhline(0.03, color='red',   linestyle='--', label='collapse (0.03)')
    axes[1].set_xticks(x); axes[1].set_xticklabels(bc_labels, rotation=30, ha='right', fontsize=7)
    axes[1].set_title('Pairwise Output Diff (per BC config)')
    axes[1].set_ylabel('Mean |out1 - out2|'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =========================================================
# MAIN
# =========================================================
def run_sensitivity_test():
    print(f"Device: {DEVICE}")
    model = load_model(VAE_PATH, LATENT_DIM, DEVICE)

    # Load val loader just for latent activity test
    from vae_fluid_multiple import make_loaders
    _, val_loader, _, _ = make_loaders(
        "./data/new/dataset_all_merged.h5",
        batch_size=32, seed=42
    )

    print("\n" + "=" * 60)
    print("TEST 3: LATENT DIMENSION ACTIVITY")
    print("=" * 60)
    mu_std, post_var = test_latent_activity(model, val_loader, DEVICE)

    # Sample random BC configs
    print("\n" + "=" * 60)
    print("TESTS 1, 2, 4: PER-BC-CONFIG SENSITIVITY")
    print("=" * 60)

    stacks        = []
    std_imgs      = []
    bc_labels     = []
    all_mean_stds = []
    all_mean_diffs= []

    for cfg_idx in range(N_BC_CONFIGS):
        # Sample a random valid BC config
        from new_generate_dataset_multiple import sample_ports, ports_overlap
        while True:
            n_in  = np.random.choice([1, 2])
            #only one outlet
            n_out = 1
            ports = sample_ports(n_in, "inlet") + sample_ports(n_out, "outlet")
            overlap = any(
                ports_overlap(ports[i], ports[j])
                for i in range(len(ports))
                for j in range(i+1, len(ports))
            )
            # check if outlet present in any inlet wall
            if any(
                p_out['type'] == 'outlet' and
                any(p_in['type'] == 'inlet' and p_out['wall'] == p_in['wall']
                    for p_in in ports)
                for p_out in ports
            ):
                overlap = True
                
            if not overlap:
                break

        bc_np     = make_bc_mask(ports)
        bc_tensor = torch.FloatTensor(bc_np).unsqueeze(0).to(DEVICE)  # [1,2,64,64]

        in_desc  = [f"{p['wall']}@{p['center']}" for p in ports if p['type']=='inlet']
        out_desc = [f"{p['wall']}@{p['center']}" for p in ports if p['type']=='outlet']
        label    = f"in:{in_desc} out:{out_desc}"
        short    = f"cfg{cfg_idx}"

        print(f"\n  Config {cfg_idx}: {label}")

        stack, std_img, mean_std = test_output_diversity(
            model, bc_tensor, DEVICE, label=short)
        mean_diff, _             = test_pairwise_sensitivity(
            model, bc_tensor, DEVICE, label=short)
        _                        = test_interpolation(
            model, bc_tensor, DEVICE, label=short)

        stacks.append(stack)
        std_imgs.append(std_img)
        bc_labels.append(short)
        all_mean_stds.append(mean_std)
        all_mean_diffs.append(mean_diff)

    # ── Global summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GLOBAL SUMMARY")
    print("=" * 60)
    print(f"  mean_std  across all configs: {np.mean(all_mean_stds):.4f}  "
          f"(target >0.10, collapse <0.05)")
    print(f"  mean_diff across all configs: {np.mean(all_mean_diffs):.4f}  "
          f"(target >0.08, collapse <0.03)")
    print(f"  active latent dims:           "
          f"{(mu_std > 0.1).sum()}/{LATENT_DIM}")

    overall = "READY FOR CMA-ES ✅" if (
        np.mean(all_mean_stds)  > 0.08 and
        np.mean(all_mean_diffs) > 0.06 and
        (mu_std > 0.1).sum()    > LATENT_DIM // 4
    ) else "NOT READY — latent space too collapsed ❌"
    print(f"\n  Verdict: {overall}")

    # ── Save plots ────────────────────────────────────────────
    plot_diversity_grid(stacks, std_imgs, bc_labels)
    plot_summary(all_mean_stds, all_mean_diffs, bc_labels)

    return {
        "mean_std_per_config":  all_mean_stds,
        "mean_diff_per_config": all_mean_diffs,
        "mu_std_per_dim":       mu_std,
        "n_active_dims":        int((mu_std > 0.1).sum()),
    }


if __name__ == "__main__":
    results = run_sensitivity_test()