import h5py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
Nx, Ny     = 64, 64
WALL       = 4
INLET_H    = int(0.1 * Nx)   # = 6  — must match generator
OUTLET_H   = int(0.1 * Ny)   # = 6

# ---------------------------------------------------------
# BC MASK HELPER
# ---------------------------------------------------------
def make_bc_mask(inlet_y, outlet_y,
                 inlet_h=INLET_H, outlet_h=OUTLET_H,
                 wall=WALL, nx=Nx, ny=Ny):
    """
    Build a [2, Nx, Ny] BC mask image.
      Channel 0: inlet  slot pixels = 1  (left wall strip)
      Channel 1: outlet slot pixels = 1  (right wall strip)
      Everything else = 0

    This gives the VAE exact spatial supervision of port locations
    instead of relying on scalar interpolation.

    Args:
        inlet_y, outlet_y: integer center positions
    Returns:
        mask: float32 numpy array [2, Nx, Ny]
    """
    mask = np.zeros((2, nx, ny), dtype=np.float32)

    inlet_lo  = max(wall, inlet_y  - inlet_h  // 2)
    inlet_hi  = min(ny - wall, inlet_y  + inlet_h  // 2)
    outlet_lo = max(wall, outlet_y - outlet_h // 2)
    outlet_hi = min(ny - wall, outlet_y + outlet_h // 2)

    # Channel 0: inlet — left wall strip (x = 0..WALL-1)
    mask[0, 0:wall, inlet_lo:inlet_hi]   = 1.0

    # Channel 1: outlet — right wall strip (x = Nx-WALL..Nx-1)
    mask[1, nx-wall:nx, outlet_lo:outlet_hi] = 1.0

    return mask   # [2, Nx, Ny]


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
class FluidDataset(Dataset):
    """
    Loads fluid topology designs and builds BC mask images on the fly.

    Returns per sample:
        density:  [1, Nx, Ny]  float32 — topology field [x, y]
        bc_mask:  [2, Nx, Ny]  float32 — inlet/outlet mask image
        metrics:  [2]          float32 — (pressure_drop, volume_fraction)
    """
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.densities     = f['density'][:]          # [N, 64, 64]
            self.pressure_drop = f['pressure_drop'][:]    # [N]
            self.volume        = f['volume_fraction'][:]  # [N]
            self.inlet_y       = f['bc_inlet_y'][:]       # [N]  integers
            self.outlet_y      = f['bc_outlet_y'][:]      # [N]  integers

        print(f"✅ Loaded {len(self.densities)} designs from {h5_path}")
        print(f"   Pressure: [{self.pressure_drop.min():.4f}, "
              f"{self.pressure_drop.max():.4f}]")
        print(f"   Volume:   [{self.volume.min():.3f}, "
              f"{self.volume.max():.3f}]")
        print(f"   Inlet Y:  [{self.inlet_y.min()}, {self.inlet_y.max()}]")
        print(f"   Outlet Y: [{self.outlet_y.min()}, {self.outlet_y.max()}]")

    def __len__(self):
        return len(self.densities)

    def __getitem__(self, idx):
        # Density [1, Nx, Ny] — no transpose, [x,y] convention
        density = torch.FloatTensor(self.densities[idx]).unsqueeze(0)

        # BC mask [2, Nx, Ny] — built from exact integer port positions
        bc_mask_np = make_bc_mask(
            int(self.inlet_y[idx]),
            int(self.outlet_y[idx])
        )
        bc_mask = torch.FloatTensor(bc_mask_np)

        # Metrics
        metrics = torch.FloatTensor([
            self.pressure_drop[idx],
            self.volume[idx],
        ])

        return density, bc_mask, metrics


def make_loaders(h5_path, batch_size=32,
                 train_frac=0.7, val_frac=0.15, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds      = FluidDataset(h5_path)
    n       = len(ds)
    n_train = int(train_frac * n)
    n_val   = int(val_frac   * n)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"📊 Split: Train={n_train}, Val={n_val}, Test={n_test}")
    return train_loader, val_loader, test_loader, ds


# ---------------------------------------------------------
# FEASIBILITY CHECK
# ---------------------------------------------------------
def is_feasible(rho, inlet_y, outlet_y, inlet_h=INLET_H, outlet_h=OUTLET_H,
                min_vol=0.10, max_vol=0.40, threshold=0.5):
    """
    Feasibility check in [x, y] convention.
    rho: numpy array [64, 64]
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

    inlet_fluid  = geom[0:3, inlet_lo:inlet_hi].sum()
    outlet_fluid = geom[Nx-3:Nx, outlet_lo:outlet_hi].sum()

    diag['inlet_ok']  = (inlet_fluid  > 0)
    diag['outlet_ok'] = (outlet_fluid > 0)

    if inlet_fluid == 0:
        return False, "No fluid at inlet region", diag
    if outlet_fluid == 0:
        return False, "No fluid at outlet region", diag

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
            nx2, ny2 = x+dx, y+dy
            if 0 <= nx2 < Nx and 0 <= ny2 < Ny:
                if geom[nx2, ny2] == 1 and (nx2, ny2) not in visited:
                    visited.add((nx2, ny2))
                    queue.append((nx2, ny2))

    diag['connected'] = reached
    if not reached:
        return False, "No path from inlet to outlet", diag

    return True, "OK", diag


# ---------------------------------------------------------
# BC-MASK-CONDITIONED VAE
# ---------------------------------------------------------
class FluidVAE(nn.Module):
    """
    BC-mask-conditioned VAE.

    Instead of scalars (inlet_y/64, outlet_y/64), conditioning uses
    a [2, 64, 64] BC mask image showing exact port pixel locations.

    Architecture:
      Encoder:
        - density [1,64,64] + bc_mask [2,64,64] → concat → [3,64,64]
        - shared conv stack → flatten → fc_mu, fc_logvar

      Decoder:
        - bc_mask [2,64,64] → small CNN → flat feature vector
        - concat(z, bc_features) → fc → reshape → deconv stack → [1,64,64]

    This gives the decoder exact spatial supervision of port locations,
    solving the port misalignment problem that scalar conditioning has.
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        conv_flat       = 256 * 4 * 4   # after 4× stride-2 convs: 64→4

        # ── Encoder ──────────────────────────────────────
        # Input: density [1] + bc_mask [2] = 3 channels
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3,   32,  4, stride=2, padding=1),  # [B,3,64,64]→[B,32,32,32]
            nn.LeakyReLU(0.2),
            nn.Conv2d(32,  64,  4, stride=2, padding=1),  # →[B,64,16,16]
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,  128, 4, stride=2, padding=1),  # →[B,128,8,8]
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # →[B,256,4,4]
            nn.LeakyReLU(0.2),
            nn.Flatten()                                   # →[B, 256*4*4]
        )
        self.fc_mu     = nn.Linear(conv_flat, latent_dim)
        self.fc_logvar = nn.Linear(conv_flat, latent_dim)

        # ── BC encoder (decoder side) ─────────────────────
        # Encodes bc_mask [2,64,64] → spatial feature vector
        # Separate from the encoder above — decoder needs its own
        # spatial understanding of the port locations
        self.bc_enc = nn.Sequential(
            nn.Conv2d(2,  16,  4, stride=2, padding=1),  # [B,2,64,64]→[B,16,32,32]
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32,  4, stride=2, padding=1),  # →[B,32,16,16]
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64,  4, stride=2, padding=1),  # →[B,64,8,8]
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # →[B,128,4,4]
            nn.LeakyReLU(0.2),
            nn.Flatten()                                  # →[B, 128*4*4]
        )
        bc_flat = 128 * 4 * 4

        # ── Decoder ──────────────────────────────────────
        # Input: concat(z [latent_dim], bc_features [bc_flat])
        self.dec_fc = nn.Linear(latent_dim + bc_flat, conv_flat)

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

    def encode(self, x, bc_mask):
        """
        x:       [B, 1, 64, 64]  density
        bc_mask: [B, 2, 64, 64]  BC mask image
        """
        inp = torch.cat([x, bc_mask], dim=1)   # [B, 3, 64, 64]
        h   = self.enc_conv(inp)               # [B, conv_flat]
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, bc_mask):
        """
        z:       [B, latent_dim]
        bc_mask: [B, 2, 64, 64]
        """
        bc_feat = self.bc_enc(bc_mask)                       # [B, bc_flat]
        h       = self.dec_fc(torch.cat([z, bc_feat], dim=1))# [B, conv_flat]
        h       = h.view(-1, 256, 4, 4)
        return self.dec_conv(h)                               # [B, 1, 64, 64]

    def forward(self, x, bc_mask):
        mu, logvar = self.encode(x, bc_mask)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z, bc_mask)
        return recon, mu, logvar


# ---------------------------------------------------------
# LOSS
# ---------------------------------------------------------
def binary_penalty(rho):
    return torch.mean(rho * (1.0 - rho))


def vae_loss(recon, target, mu, logvar, beta=0.1, w_bin=0.5):
    """
    Total loss = BCE reconstruction + beta*KL + binary penalty.
    Full image reconstruction (no interior mask needed —
    wall pixels in training data are already 0, so BCE
    correctly pushes decoder to output 0 there too).
    """
    recon_loss = F.binary_cross_entropy(recon, target, reduction='mean')
    kld        = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    bin_loss   = binary_penalty(recon)

    total = recon_loss + beta * kld + w_bin * bin_loss
    return total, recon_loss, kld, bin_loss


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
def train_vae(train_loader, val_loader,
              epochs=400, lr=1e-3, latent_dim=32,
              beta=0.1, w_bin=0.5,
              save_path="vae_best.pth",
              device='cuda'):

    model     = FluidVAE(latent_dim=latent_dim).to(device)
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

    print(f"\n🚀 Training BC-mask-conditioned VAE on {device}")
    print(f"   Epochs={epochs} | lr={lr} | beta={beta} | "
          f"w_bin={w_bin} | latent={latent_dim}")
    print(f"   BC conditioning: image mask [2,64,64] — exact port locations")
    print("=" * 60)

    for epoch in range(epochs):

        # ── Train ─────────────────────────────────────────
        model.train()
        t_loss = t_recon = t_kld = t_bin = 0.0

        for density, bc_mask, _ in train_loader:
            density = density.to(device)    # [B, 1, 64, 64]
            bc_mask = bc_mask.to(device)    # [B, 2, 64, 64]

            optimizer.zero_grad()
            recon, mu, logvar = model(density, bc_mask)
            loss, rl, kl, bl  = vae_loss(
                recon, density, mu, logvar,
                beta=beta, w_bin=w_bin
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
            for density, bc_mask, _ in val_loader:
                density = density.to(device)
                bc_mask = bc_mask.to(device)
                recon, mu, logvar = model(density, bc_mask)
                loss, rl, kl, bl  = vae_loss(
                    recon, density, mu, logvar,
                    beta=beta, w_bin=w_bin
                )
                v_loss  += loss.item()
                v_recon += rl.item()
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
    model.eval()
    feasible_count  = 0
    total_count     = 0
    violation_types = {}
    reconstructions = []
    originals       = []
    bc_params_list  = []

    print("\n🔍 Evaluating reconstructions...")

    with torch.no_grad():
        for density, bc_mask, metrics in test_loader:
            density_dev = density.to(device)
            bc_mask_dev = bc_mask.to(device)

            recon, _, _ = model(density_dev, bc_mask_dev)

            for i in range(len(density)):
                recon_np = recon[i, 0].cpu().numpy()     # [64,64] [x,y]
                orig_np  = density[i, 0].cpu().numpy()

                # Recover inlet/outlet from bc_mask image
                # Channel 0: inlet strip on left wall — find y range
                inlet_mask_np  = bc_mask[i, 0].cpu().numpy()  # [64,64]
                outlet_mask_np = bc_mask[i, 1].cpu().numpy()

                inlet_ys  = np.where(inlet_mask_np[0:WALL, :].sum(axis=0) > 0)[0]
                outlet_ys = np.where(outlet_mask_np[Nx-WALL:Nx, :].sum(axis=0) > 0)[0]

                inlet_y  = int(inlet_ys.mean())  if len(inlet_ys)  > 0 else 32
                outlet_y = int(outlet_ys.mean()) if len(outlet_ys) > 0 else 32

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

        axes[0, i].imshow(originals[i].T, cmap='gray_r',
                          origin='lower', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original\nin={inlet_y} out={outlet_y}',
                              fontsize=7)
        axes[0, i].axis('off')

        axes[1, i].imshow(reconstructions[i].T, cmap='gray_r',
                          origin='lower', vmin=0, vmax=1)
        axes[1, i].set_title('Reconstructed', fontsize=7)
        axes[1, i].axis('off')

    plt.suptitle("Top: Original  |  Bottom: VAE Reconstruction", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {save_path}")


def plot_history(history, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history['train_recon'], label='Train')
    axes[1].plot(history['val_recon'],   label='Val')
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(history['train_kld'], label='KLD')
    axes[2].plot(history['train_bin'], label='Binary')
    axes[2].set_title('Aux Losses (train)')
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

    H5_PATH = "./data/dataset_all_merged.h5"

    train_loader, val_loader, test_loader, ds = make_loaders(
        h5_path=H5_PATH,
        batch_size=32,
    )

    model, history = train_vae(
        train_loader, val_loader,
        epochs=400,
        lr=1e-3,
        latent_dim=32,
        beta=0.1,
        w_bin=0.5,
        save_path="vae_best_new.pth",
        device=device
    )

    plot_history(history)

    reconstructions, originals, bc_params = evaluate_vae(
        model, test_loader, device
    )

    plot_reconstructions(originals, reconstructions, bc_params)

    print("\n✅ All done!")