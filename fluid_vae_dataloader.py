import h5py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
Nx, Ny = 64, 64
WALL   = 4

# Interior mask — computed once, used in loss
# True for designable interior cells, False for wall pixels
# Shape: [1, 1, 64, 64] for broadcasting with [B, 1, 64, 64]
_interior = torch.zeros(1, 1, Nx, Ny, dtype=torch.bool)
_interior[0, 0, WALL:-WALL, WALL:-WALL] = True
INTERIOR_MASK = _interior  # moved to device in train_vae
INTERIOR_MASK = None 

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
def load_fluid_data(h5_path, batch_size=32, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Load fluid designs from HDF5 and return train/val/test loaders.

    Convention:
      - density saved as [N, 64, 64] in [x, y] order by the generator
      - we keep it as [x, y] throughout — NO transpose applied here
      - BC params normalized to [0, 1] by dividing by Ny=64

    Returns:
        train_loader, val_loader, test_loader, info
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    with h5py.File(h5_path, 'r') as f:
        densities    = f['density'][:]           # [N, 64, 64]  [x, y]
        pressure_drops = f['pressure_drop'][:]
        volumes      = f['volume_fraction'][:]
        inlet_y      = f['bc_inlet_y'][:]
        outlet_y     = f['bc_outlet_y'][:]
        height_diff  = f['bc_height_diff'][:]

    print(f"✅ Loaded {len(densities)} designs from {h5_path}")

    # [N, 1, 64, 64] — add channel dim, NO transpose
    X = torch.FloatTensor(densities).unsqueeze(1)

    # BC: normalize inlet_y and outlet_y to [0, 1]
    # height_diff kept as signed normalized value
    bc_tensor = torch.FloatTensor(np.stack([
        inlet_y  / Ny,         # [0, 1]
        outlet_y / Ny,         # [0, 1]
    ], axis=1))                # [N, 2]

    metrics_tensor = torch.FloatTensor(
        np.stack([pressure_drops, volumes], axis=1)
    )                          # [N, 2]

    dataset = TensorDataset(X, bc_tensor, metrics_tensor)

    n_total = len(dataset)
    n_train = int(train_frac * n_total)
    n_val   = int(val_frac   * n_total)
    n_test  = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    info = {
        'total': n_total, 'train': n_train, 'val': n_val, 'test': n_test,
        'volume_range':   (float(volumes.min()),       float(volumes.max())),
        'pressure_range': (float(pressure_drops.min()), float(pressure_drops.max())),
        'inlet_y_range':  (int(inlet_y.min()),         int(inlet_y.max())),
        'outlet_y_range': (int(outlet_y.min()),        int(outlet_y.max())),
    }

    print(f"📊 Split: Train={n_train}, Val={n_val}, Test={n_test}")
    print(f"   Volume:   [{info['volume_range'][0]:.3f},  {info['volume_range'][1]:.3f}]")
    print(f"   Pressure: [{info['pressure_range'][0]:.4f}, {info['pressure_range'][1]:.4f}]")
    print(f"   Inlet Y:  [{info['inlet_y_range'][0]}, {info['inlet_y_range'][1]}]")
    print(f"   Outlet Y: [{info['outlet_y_range'][0]}, {info['outlet_y_range'][1]}]")

    return train_loader, val_loader, test_loader, info


# ---------------------------------------------------------
# FEASIBILITY CHECK
# ---------------------------------------------------------
def is_feasible(rho, inlet_y, outlet_y, inlet_h=6, outlet_h=6,
                min_vol=0.10, max_vol=0.40, threshold=0.5):
    """
    Feasibility check in [x, y] convention.
    rho: numpy array [64, 64] in [x, y] order
    """
    geom   = (rho > threshold).astype(np.uint8)
    volume = geom.mean()

    diag = {'volume': volume, 'connected': False,
            'inlet_ok': False, 'outlet_ok': False}

    if volume < min_vol:
        return False, f"Volume too low: {volume:.3f}", diag
    if volume > max_vol:
        return False, f"Volume too high: {volume:.3f}", diag

    inlet_lo  = max(WALL, inlet_y  - inlet_h  // 2)
    inlet_hi  = min(Ny - WALL, inlet_y  + inlet_h  // 2)
    outlet_lo = max(WALL, outlet_y - outlet_h // 2)
    outlet_hi = min(Ny - WALL, outlet_y + outlet_h // 2)

    # Check fluid presence at inlet (left wall, x=0..2)
    inlet_fluid  = geom[0:3, inlet_lo:inlet_hi].sum()
    outlet_fluid = geom[Nx-3:Nx, outlet_lo:outlet_hi].sum()

    diag['inlet_ok']  = (inlet_fluid  > 0)
    diag['outlet_ok'] = (outlet_fluid > 0)

    if inlet_fluid == 0:
        return False, "No fluid at inlet region", diag
    if outlet_fluid == 0:
        return False, "No fluid at outlet region", diag

    # BFS connectivity in [x, y]
    inlet_cells = [(x, y) for x in range(0, 3)
                           for y in range(inlet_lo, inlet_hi)
                           if geom[x, y] == 1]
    if not inlet_cells:
        return False, "No inlet fluid cells", diag

    visited = set(inlet_cells)
    queue   = deque(inlet_cells)
    reached = False

    while queue:
        x, y = queue.popleft()
        if x >= Nx - 3 and outlet_lo <= y < outlet_hi:
            reached = True
            break
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < Nx and 0 <= ny < Ny:
                if geom[nx, ny] == 1 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    diag['connected'] = reached
    if not reached:
        return False, "No path from inlet to outlet", diag

    return True, "OK", diag


# ---------------------------------------------------------
# BC-CONDITIONED VAE MODEL
# ---------------------------------------------------------
class FluidVAE(nn.Module):
    """
    BC-conditioned VAE.
    BC vector (inlet_y_norm, outlet_y_norm) is injected:
      - In encoder: concatenated after flattening conv features
      - In decoder: concatenated with z before fc projection

    This forces the model to learn topology conditioned on port positions,
    which is essential for CMA-ES to search meaningful designs for a
    specific BC configuration.
    """
    def __init__(self, latent_dim=32, bc_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.bc_dim     = bc_dim
        conv_flat       = 256 * 4 * 4   # 64→32→16→8→4, 256 channels

        # ── Encoder ──────────────────────────────────────
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1,   32,  4, stride=2, padding=1),  # [B,1,64,64]→[B,32,32,32]
            nn.LeakyReLU(0.2),
            nn.Conv2d(32,  64,  4, stride=2, padding=1),  # →[B,64,16,16]
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,  128, 4, stride=2, padding=1),  # →[B,128,8,8]
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # →[B,256,4,4]
            nn.LeakyReLU(0.2),
            nn.Flatten()                                   # →[B, 256*4*4]
        )
        # BC injected here: conv_flat + bc_dim → latent
        self.fc_mu     = nn.Linear(conv_flat + bc_dim, latent_dim)
        self.fc_logvar = nn.Linear(conv_flat + bc_dim, latent_dim)

        # ── Decoder ──────────────────────────────────────
        # BC injected here: latent_dim + bc_dim → conv_flat
        self.dec_fc = nn.Linear(latent_dim + bc_dim, conv_flat)

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4→8
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),  # 8→16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),  # 16→32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32,  1,   4, stride=2, padding=1),  # 32→64
            nn.Sigmoid()
        )

    def encode(self, x, bc):
        h = self.enc_conv(x)                    # [B, conv_flat]
        h = torch.cat([h, bc], dim=-1)           # [B, conv_flat + bc_dim]
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, bc):
        h = self.dec_fc(torch.cat([z, bc], dim=-1))  # [B, conv_flat]
        h = h.view(-1, 256, 4, 4)
        return self.dec_conv(h)                       # [B, 1, 64, 64]

    def forward(self, x, bc):
        mu, logvar = self.encode(x, bc)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z, bc)
        return recon, mu, logvar


# ---------------------------------------------------------
# LOSS FUNCTION
# ---------------------------------------------------------
def binary_penalty(rho):
    """Push outputs toward 0 or 1, penalise grey values."""
    return torch.mean(rho * (1.0 - rho))


def vae_loss(recon, target, mu, logvar, beta=0.1, w_bin=0.5,
             interior_mask=None):
    """
    Total loss = reconstruction (interior only) + beta*KL + binary penalty.

    Reconstruction is masked to interior cells only.
    Wall pixels are always 0 and add no useful signal.

    Args:
        recon, target: [B, 1, 64, 64]
        mu, logvar:    encoder outputs
        beta:          KL weight
        w_bin:         binary penalty weight
        interior_mask: [1, 1, 64, 64] bool tensor on same device
    """
    if interior_mask is not None:
        mask   = interior_mask.expand_as(recon)
        recon_loss = F.binary_cross_entropy(
            recon[mask], target[mask], reduction='mean'
        )
    else:
        recon_loss = F.binary_cross_entropy(recon, target, reduction='mean')

    kld      = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    bin_loss = binary_penalty(recon)

    total = recon_loss + beta * kld + w_bin * bin_loss
    return total, recon_loss, kld, bin_loss


# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
def train_vae(train_loader, val_loader,
              epochs=400, lr=1e-3, latent_dim=32,
              beta=0.1, w_bin=0.5,
              save_path="vae_best.pth",
              device='cuda'):

    global INTERIOR_MASK
    #INTERIOR_MASK = INTERIOR_MASK.to(device)

    model     = FluidVAE(latent_dim=latent_dim, bc_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=25, factor=0.5, verbose=True
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_recon': [], 'val_recon': [],
        'train_kld': [],   'val_kld': [],
        'train_bin': [],   'val_bin': [],
    }
    best_val_loss = float('inf')

    print(f"\n🚀 Training BC-conditioned VAE on {device}")
    print(f"   Epochs={epochs} | lr={lr} | beta={beta} | w_bin={w_bin} | latent={latent_dim}")
    print("=" * 60)

    for epoch in range(epochs):

        # ── Train ─────────────────────────────────────────
        model.train()
        t_loss = t_recon = t_kld = t_bin = 0.0

        for density, bc, _ in train_loader:
            density = density.to(device)   # [B, 1, 64, 64]
            bc      = bc.to(device)        # [B, 2]

            optimizer.zero_grad()
            recon, mu, logvar = model(density, bc)
            loss, rl, kl, bl  = vae_loss(
                recon, density, mu, logvar,
                beta=beta, w_bin=w_bin,
                interior_mask=INTERIOR_MASK
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss  += loss.item()
            t_recon += rl.item()
            t_kld   += kl.item()
            t_bin   += bl.item()

        n = len(train_loader)
        t_loss /= n; t_recon /= n; t_kld /= n; t_bin /= n

        # ── Validate ──────────────────────────────────────
        model.eval()
        v_loss = v_recon = v_kld = v_bin = 0.0

        with torch.no_grad():
            for density, bc, _ in val_loader:
                density = density.to(device)
                bc      = bc.to(device)
                recon, mu, logvar = model(density, bc)
                loss, rl, kl, bl  = vae_loss(
                    recon, density, mu, logvar,
                    beta=beta, w_bin=w_bin,
                    interior_mask=INTERIOR_MASK
                )
                v_loss  += loss.item()
                v_recon += rl.item()   # ← correctly from val loop
                v_kld   += kl.item()
                v_bin   += bl.item()

        n = len(val_loader)
        v_loss /= n; v_recon /= n; v_kld /= n; v_bin /= n

        scheduler.step(v_loss)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_recon'].append(t_recon)
        history['val_recon'].append(v_recon)
        history['train_kld'].append(t_kld)
        history['val_kld'].append(v_kld)
        history['train_bin'].append(t_bin)
        history['val_bin'].append(v_bin)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Ep {epoch+1:03d} | "
                  f"Train: total={t_loss:.4f} recon={t_recon:.4f} "
                  f"kl={t_kld:.4f} bin={t_bin:.4f} | "
                  f"Val: total={v_loss:.4f} recon={v_recon:.4f}")

    model.load_state_dict(
        torch.load(save_path, weights_only=True, map_location=device)
    )
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")
    return model, history


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------
def evaluate_vae(model, test_loader, device='cuda'):
    """Evaluate VAE reconstructions on test set."""
    model.eval()
    feasible_count  = 0
    total_count     = 0
    violation_types = {}
    reconstructions = []
    originals       = []
    bc_params_list  = []

    print("\n🔍 Evaluating reconstructions...")

    with torch.no_grad():
        for density, bc, _ in test_loader:
            density_dev = density.to(device)
            bc_dev      = bc.to(device)

            recon, _, _ = model(density_dev, bc_dev)

            for i in range(len(density)):
                # Both in [x, y] convention — no transpose
                recon_np = recon[i, 0].cpu().numpy()     # [64, 64] [x, y]
                orig_np  = density[i, 0].cpu().numpy()   # [64, 64] [x, y]

                # Denormalize BC
                inlet_y  = int(round(bc[i, 0].item() * Ny))
                outlet_y = int(round(bc[i, 1].item() * Ny))

                is_ok, reason, diag = is_feasible(recon_np, inlet_y, outlet_y)

                if is_ok:
                    feasible_count += 1
                else:
                    violation_types[reason] = violation_types.get(reason, 0) + 1

                total_count += 1
                reconstructions.append(recon_np)
                originals.append(orig_np)
                bc_params_list.append((inlet_y, outlet_y))

    print(f"\n📊 Feasibility Results:")
    print(f"   Feasible: {feasible_count}/{total_count} "
          f"({100*feasible_count/total_count:.1f}%)")
    print(f"\n   Violation breakdown:")
    for vtype, count in violation_types.items():
        print(f"      {vtype}: {count}")

    return reconstructions, originals, bc_params_list


# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------
def plot_reconstructions(originals, reconstructions, bc_params_list,
                         save_path="recon_samples.png"):
    n_samples = min(6, len(originals))
    fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))

    for i in range(n_samples):
        inlet_y, outlet_y = bc_params_list[i]

        # Transpose [x,y] → [y,x] only for imshow display
        axes[0, i].imshow(originals[i].T,  cmap='gray_r', origin='lower', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original\nin={inlet_y} out={outlet_y}', fontsize=7)
        axes[0, i].axis('off')

        axes[1, i].imshow(reconstructions[i].T, cmap='gray_r', origin='lower', vmin=0, vmax=1)
        axes[1, i].set_title('Reconstructed', fontsize=7)
        axes[1, i].axis('off')

    plt.suptitle("Top: Original  |  Bottom: VAE Reconstruction", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {save_path}")


def plot_history(history, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'],  label='Train')
    axes[0].plot(history['val_loss'],    label='Val')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history['train_recon'], label='Train')
    axes[1].plot(history['val_recon'],   label='Val')   # ← correctly from val loop
    axes[1].set_title('Reconstruction Loss (interior only)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(history['train_kld'],  label='KLD (train)')
    axes[2].plot(history['train_bin'],  label='Binary (train)')
    axes[2].set_title('Aux Losses')
    axes[2].set_xlabel('Epoch')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {save_path}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")

    # ── Load data ─────────────────────────────────────────
    # Switch between dataset_final and dataset_all here
    H5_PATH = "./data/dataset_all_merged.h5"   # or dataset_all_run1.h5

    train_loader, val_loader, test_loader, data_info = load_fluid_data(
        h5_path=H5_PATH,
        batch_size=32,
    )

    # ── Train ─────────────────────────────────────────────
    model, history = train_vae(
        train_loader, val_loader,
        epochs=200,
        lr=1e-3,
        latent_dim=32,
        beta=0.1,
        w_bin=0.5,
        save_path="vae_best.pth",
        device=device
    )

    # ── Plot training curves ───────────────────────────────
    plot_history(history)

    # ── Evaluate ──────────────────────────────────────────
    reconstructions, originals, bc_params = evaluate_vae(
        model, test_loader, device
    )

    # ── Plot reconstructions ───────────────────────────────
    plot_reconstructions(originals, reconstructions, bc_params)

    print("\n✅ All done! Check recon_samples.png and training_curves.png")