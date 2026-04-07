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


# =========================================================
# CONSTANTS
# =========================================================
Nx, Ny  = 64, 64
WALL    = 4
INLET_H  = int(0.1 * Nx)   # = 6
OUTLET_H = int(0.1 * Ny)   # = 6


# =========================================================
# BC MASK HELPER
# =========================================================
def make_bc_mask(ports, nx=Nx, ny=Ny, wall=WALL):
    """
    Build a [2, Nx, Ny] BC mask image.
      Channel 0: all inlet  pixels = 1
      Channel 1: all outlet pixels = 1
    """
    mask = np.zeros((2, nx, ny), dtype=np.float32)
    ch = {"inlet": 0, "outlet": 1}

    for p in ports:
        r         = p["range"]       # slice object
        wall_side = p["wall"]
        c         = ch[p["type"]]

        if wall_side == "left":
            mask[c, 0:wall, r]        = 1.0
        elif wall_side == "right":
            mask[c, nx-wall:nx, r]    = 1.0
        elif wall_side == "bottom":
            mask[c, r, 0:wall]        = 1.0
        elif wall_side == "top":
            mask[c, r, ny-wall:ny]    = 1.0

    return mask   # [2, Nx, Ny]


# =========================================================
# DIFFERENTIABLE CONNECTIVITY LOSS
# =========================================================
def make_port_mask(ports, port_type, nx=Nx, ny=Ny, wall=WALL, device='cpu'):
    """
    Build a [1, 1, Nx, Ny] binary mask marking the first fluid
    cells of all ports of the given type.
    """
    mask = torch.zeros(1, 1, nx, ny, device=device)
    for p in ports:
        if p["type"] != port_type:
            continue
        r    = p["range"]
        wall_side = p["wall"]
        if wall_side == "left":
            mask[0, 0, wall, r]          = 1.0
        elif wall_side == "right":
            mask[0, 0, nx-wall-1, r]     = 1.0
        elif wall_side == "bottom":
            mask[0, 0, r, wall]          = 1.0
        elif wall_side == "top":
            mask[0, 0, r, ny-wall-1]     = 1.0
    return mask



def path_connectivity_loss(recon, ports_batch, device, n_samples=20):
    """Smoother connectivity loss using sigmoid-softened path penalty."""
    B = recon.shape[0]
    losses = []
    
    for i in range(B):
        ports = ports_batch[i]
        topo = recon[i, 0]  # [Nx, Ny]
        
        inlets = [p for p in ports if p["type"] == "inlet"]
        outlets = [p for p in ports if p["type"] == "outlet"]
        if not inlets or not outlets:
            losses.append(torch.tensor(0.0, device=device))
            continue
        
        path_losses = []
        for inlet in inlets:
            for outlet in outlets:
                # Get port centers (same as before)
                def get_center(port):
                    r = port["range"]
                    c = (r.start + r.stop) / 2.0
                    if port["wall"] == "left":
                        return torch.tensor([WALL, c], device=device)
                    elif port["wall"] == "right":
                        return torch.tensor([Nx-WALL-1, c], device=device)
                    elif port["wall"] == "bottom":
                        return torch.tensor([c, WALL], device=device)
                    else:
                        return torch.tensor([c, Ny-WALL-1], device=device)
                
                p0 = get_center(inlet)
                p1 = get_center(outlet)
                
                # Sample path points
                t_vals = torch.linspace(0, 1, n_samples, device=device)
                xs = (p0[0] + t_vals * (p1[0] - p0[0])).clamp(WALL, Nx-WALL-1).long()
                ys = (p0[1] + t_vals * (p1[1] - p0[1])).clamp(WALL, Ny-WALL-1).long()
                
                path_vals = topo[xs, ys]  # [n_samples]
                
                # SOFT penalty: use mean + sigmoid to avoid harsh gradients
                # Want path_vals.mean() > 0.7 → loss ≈ 0
                path_mean = path_vals.mean()
                soft_penalty = torch.sigmoid(10.0 * (0.7 - path_mean))  # Smooth step
                path_losses.append(soft_penalty)
        
        if path_losses:
            losses.append(torch.stack(path_losses).mean())
        else:
            losses.append(torch.tensor(0.0, device=device))
    
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


# =========================================================
# PORT CELLS HELPER  (needed by check_connectivity)
# =========================================================
def port_cells(port):
    """Return list of (x,y) tuples for the first fluid cells of a port."""
    r    = port["range"]
    wall = port["wall"]
    if wall == "left":
        return [(WALL, y) for y in range(r.start, r.stop)]
    if wall == "right":
        return [(Nx - WALL - 1, y) for y in range(r.start, r.stop)]
    if wall == "bottom":
        return [(x, WALL) for x in range(r.start, r.stop)]
    if wall == "top":
        return [(x, Ny - WALL - 1) for x in range(r.start, r.stop)]
    return []


# =========================================================
# CUSTOM COLLATE  (ports_list contains slice objects which
# the default PyTorch collate cannot handle)
# =========================================================
def collate_fn(batch):
    """
    Each item: (density, bc_mask, metrics, ports_list)
    density/bc_mask/metrics → stacked tensors as usual
    ports_list              → kept as a plain Python list of lists
    """
    densities  = torch.stack([b[0] for b in batch])
    bc_masks   = torch.stack([b[1] for b in batch])
    metrics    = torch.stack([b[2] for b in batch])
    ports_list = [b[3] for b in batch]   # list of port-dict lists
    return densities, bc_masks, metrics, ports_list

def make_wall_mask():
    mask = np.zeros((Nx,Ny), np.float32)
    mask[:WALL,:] = 1
    mask[-WALL:,:] = 1
    mask[:,:WALL] = 1
    mask[:,-WALL:] = 1
    return torch.tensor(mask)[None,None]

# =========================================================
# DATASET
# =========================================================
class FluidDataset(Dataset):
    """
    Returns per sample:
        density:    [1, Nx, Ny]   float32
        bc_mask:    [2, Nx, Ny]   float32
        metrics:    [2]           float32  (pressure_drop, volume_fraction)
        ports_list: list of port dicts for this sample
    """
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.densities     = f['density'][:]        # [N, 64, 64]
            self.pressure_drop = f['pressure_drop'][:]  # [N]
            self.volume        = f['volume_fraction'][:] # [N]
            raw_meta           = f['metadata'][:]

        # Parse port dicts from JSON metadata
        self.ports_list = []
        for m in raw_meta:
            meta  = json.loads(m.decode('utf-8'))
            ports = []
            for desc in meta.get("inlets", []):
                wall_side, center = desc.rsplit("_", 1)
                center = int(center)
                if wall_side in ["left", "right"]:
                    lo = max(WALL, center - INLET_H//2)
                    hi = min(Ny - WALL, center + INLET_H//2)
                else:  # top/bottom → slice in x direction
                    lo = max(WALL, center - INLET_H//2)
                    hi = min(Nx - WALL, center + INLET_H//2)
                ports.append({
                    "type": "inlet", "wall": wall_side,
                    "range": slice(lo, hi), "center": center
                })
            for desc in meta.get("outlets", []):
                wall_side, center = desc.rsplit("_", 1)
                center = int(center)
                if wall_side in ["left", "right"]:
                    lo = max(WALL, center - INLET_H//2)
                    hi = min(Ny - WALL, center + INLET_H//2)
                else:  # top/bottom → slice in x direction
                    lo = max(WALL, center - INLET_H//2)
                    hi = min(Nx - WALL, center + INLET_H//2)
                ports.append({
                    "type": "outlet", "wall": wall_side,
                    "range": slice(lo, hi), "center": center
                })
            self.ports_list.append(ports)


        print(f"Loaded {len(self.densities)} designs from {h5_path}")
        print(f"  Pressure: [{self.pressure_drop.min():.4f}, "
              f"{self.pressure_drop.max():.4f}]")
        print(f"  Volume:   [{self.volume.min():.3f}, "
              f"{self.volume.max():.3f}]")

    def __len__(self):
        return len(self.densities)

    def __getitem__(self, idx):
        density = torch.FloatTensor(self.densities[idx]).unsqueeze(0)  # [1,64,64]

        bc_mask_np = make_bc_mask(self.ports_list[idx])
        bc_mask    = torch.FloatTensor(bc_mask_np)                     # [2,64,64]

        metrics = torch.FloatTensor([
            self.pressure_drop[idx],
            self.volume[idx],
        ])

        # BUG 2 FIX: return ports_list so evaluate_vae can use it
        return density, bc_mask, metrics, self.ports_list[idx]


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
                              shuffle=True,  drop_last=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    print(f"Split: Train={n_train}, Val={n_val}, Test={n_test}")
    return train_loader, val_loader, test_loader, ds


# =========================================================
# FEASIBILITY
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
    """BFS from every inlet to see if any outlet is reachable."""
    binary = (density_np > 0.5).astype(np.uint8)

    inlet_ports  = [p for p in ports if p["type"] == "inlet"]
    outlet_ports = [p for p in ports if p["type"] == "outlet"]

    outlet_cells = set()
    for p in outlet_ports:
        outlet_cells |= set(map(tuple, port_cells(p)))

    for inlet in inlet_ports:
        starts  = [tuple(c) for c in port_cells(inlet)]
        visited = bfs(binary, starts)
        if not any(cell in outlet_cells for cell in visited):
            return False   # this inlet never reaches an outlet
    return True


def is_feasible_vae(density_np, ports,
                    vol_min=0.10, vol_max=0.40):
    """
    Lightweight feasibility for VAE evaluation.
    Skips pressure check (VAE has no pressure info at inference time).
    """
    vol = float((density_np > 0.5).mean())
    if vol < vol_min: return False, f"vol_low:{vol:.3f}"
    if vol > vol_max: return False, f"vol_high:{vol:.3f}"
    if not check_connectivity(density_np, ports):
        return False, "disconnected"
    return True, "OK"


# =========================================================
# MODEL
# =========================================================
class FluidVAE(nn.Module):

    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        conv_flat       = 256 * 4 * 4

        # Encoder: density[1] + bc_mask[2] = 3 channels
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3,   32,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1), 
            nn.Conv2d(32,  64,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,  128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )

        self.fc_mu     = nn.Linear(conv_flat, latent_dim)
        self.fc_logvar = nn.Linear(conv_flat, latent_dim)

        # Separate BC encoder on decoder side
        self.bc_enc = nn.Sequential(
            nn.Conv2d(2,  16,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        bc_flat = 128 * 4 * 4

        # Decoder
        self.dec_fc = nn.Linear(latent_dim + bc_flat, conv_flat)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1), 
            nn.ConvTranspose2d(32,  1,   4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, bc_mask):
        inp = torch.cat([x, bc_mask], dim=1)
        h   = self.enc_conv(inp)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, bc_mask):
        bc_feat = self.bc_enc(bc_mask)
        h       = self.dec_fc(torch.cat([z, bc_feat], dim=1))
        h       = h.view(-1, 256, 4, 4)
        return self.dec_conv(h)

    def forward(self, x, bc_mask):
        mu, logvar = self.encode(x, bc_mask)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z, bc_mask)
        return recon, mu, logvar


# =========================================================
# LOSS
# =========================================================
def binary_penalty(rho):
    return torch.mean(rho * (1.0 - rho))


# =========================================================
# UPDATED LOSS
# =========================================================
def vae_loss(recon, target, mu, logvar,
             beta=1.0, w_bin=2.0, w_sharp=1.0,
             w_conn=0.0,           # start at 0, anneal up
             ports_batch=None,
             device='cpu'):

    recon_loss = F.binary_cross_entropy(recon, target, reduction='mean')
    kld        = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    bin_loss   = torch.mean(recon * (1.0 - recon))
    sharp_loss = torch.mean(torch.min(recon, 1.0 - recon))

    conn_loss = torch.tensor(0.0, device=device)
    if w_conn > 0.0 and ports_batch is not None:
        conn_loss = path_connectivity_loss(recon, ports_batch, device)

    total = (recon_loss
             + beta    * kld
             + w_bin   * bin_loss
             + w_sharp * sharp_loss
             + w_conn  * conn_loss)

    return total, recon_loss, kld, bin_loss, conn_loss


# =========================================================
# TRAINING
# =========================================================
def train_vae(train_loader, val_loader,
              epochs=400, lr=1e-3, latent_dim=16,
              beta=0.05, w_bin=5.0, w_sharp=1.0,
              w_conn_max=2.0,          # final connectivity weight
              w_conn_warmup=1,
              save_path="vae_best.pth",
              device='cuda'):

    model     = FluidVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4 )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=25, factor=0.5, verbose=True
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_recon': [], 'val_recon': [],
        'train_kld': [],   'val_kld': [],
        'train_bin': [],   'val_bin': [],
        'train_conn': [],  
    }
    best_val_loss = float('inf')

    print(f"\nTraining BC-mask-conditioned VAE on {device}")
    print(f"  Epochs={epochs} | lr={lr} | beta={beta} | "
          f"w_bin={w_bin} | latent={latent_dim}")
    print("=" * 60)

    patience_counter = 0
    EARLY_STOP_PATIENCE = 20

    for epoch in range(epochs):

        if epoch < w_conn_warmup:
            w_conn = 0.0
        else:
            progress = (epoch - w_conn_warmup) / max(1, epochs - w_conn_warmup)
            w_conn   = w_conn_max * progress

        # --- Train ---
        model.train()
        t_loss = t_recon = t_kld = t_bin = t_conn = 0.0

        # BUG 4 FIX: unpack 4 values; use _ for ports (not needed in training)
        for density, bc_mask, _, _ports in train_loader:
            density = density.to(device)
            bc_mask = bc_mask.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(density, bc_mask)
            #WALL_MASK_DEV = make_wall_mask().to(device)
            #recon = recon * (1 - WALL_MASK_DEV) + density * WALL_MASK_DEV
            loss, rl, kl, bl, cl = vae_loss(
                recon, density, mu, logvar,
                beta=beta, w_bin=w_bin, w_sharp=w_sharp,
                w_conn=w_conn,
                ports_batch=_ports,
                device=device
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss  += loss.item()
            t_recon += rl.item()
            t_kld   += kl.item()
            t_bin   += bl.item()
            t_conn  += cl.item()

        n = len(train_loader)
        t_loss /= n; t_recon /= n; t_kld /= n; t_bin /= n; t_conn /= n

        # --- Validate ---
        model.eval()
        v_loss = v_recon = v_kld = v_bin = v_conn = 0.0   # add v_conn

        with torch.no_grad():
            for density, bc_mask, _, _ports in val_loader:
                density = density.to(device)
                bc_mask = bc_mask.to(device)
                recon, mu, logvar = model(density, bc_mask)

                # FIX: unpack 5 values, pass w_conn=0 for val (no connectivity penalty)
                loss, rl, kl, bl, cl = vae_loss(
                    recon, density, mu, logvar,
                    beta=beta, w_bin=w_bin, w_sharp=w_sharp,
                    w_conn=0.0,          # don't penalize connectivity in val
                    ports_batch=_ports,
                    device=device
                )
                v_loss  += loss.item()
                v_recon += rl.item()
                v_kld   += kl.item()
                v_bin   += bl.item()
                v_conn  += cl.item()   # will always be 0 since w_conn=0, but keeps symmetry

        n = len(val_loader)
        v_loss /= n; v_recon /= n; v_kld /= n; v_bin /= n; v_conn /= n
        scheduler.step(v_loss)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_recon'].append(t_recon)
        history['val_recon'].append(v_recon)
        history['train_kld'].append(t_kld)
        history['val_kld'].append(v_kld)
        history['train_bin'].append(t_bin)
        history['val_bin'].append(v_bin)
        history['train_conn'].append(t_conn)  

        # in train_vae, after val loss calculation:


        if v_loss < best_val_loss:
            best_val_loss  = v_loss
            best_val_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}. Best was epoch {best_val_epoch}")
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Ep {epoch+1:03d} | w_conn={w_conn:.2f} | "
                  f"Train: total={t_loss:.4f} recon={t_recon:.4f} "
                  f"kl={t_kld:.4f} bin={t_bin:.4f} conn={t_conn:.4f} | "
                  f"Val: total={v_loss:.4f}")
            


    model.load_state_dict(
        torch.load(save_path, weights_only=True, map_location=device)
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model, history


# =========================================================
# EVALUATION
# =========================================================
def evaluate_vae(model, test_loader, device='cuda'):
    model.eval()
    feasible_count  = 0
    total_count     = 0
    violation_types = {}
    reconstructions = []
    originals       = []
    ports_out       = []   # stores ports list per sample for plotting

    print("\nEvaluating reconstructions...")

    with torch.no_grad():
        # BUG 5 FIX: unpack 4 values; ports_batch is a list of port-dict lists
        for density, bc_mask, metrics, ports_batch in test_loader:
            density_dev = density.to(device)
            bc_mask_dev = bc_mask.to(device)

            recon, _, _ = model(density_dev, bc_mask_dev)

            for i in range(len(density)):
                recon_np = recon[i, 0].cpu().numpy()   # [64,64]
                orig_np  = density[i, 0].cpu().numpy()
                ports    = ports_batch[i]               # list of port dicts

                # BUG 6 FIX: use correct is_feasible signature
                # (no pressure check — VAE has no pressure info)
                is_ok, reason = is_feasible_vae(recon_np, ports)

                if is_ok:
                    feasible_count += 1
                else:
                    violation_types[reason] = violation_types.get(reason, 0) + 1

                total_count += 1
                reconstructions.append(recon_np)
                originals.append(orig_np)
                ports_out.append(ports)

    pct = 100 * feasible_count / total_count if total_count > 0 else 0
    print(f"\nFeasibility: {feasible_count}/{total_count} ({pct:.1f}%)")
    print("  Violation breakdown:")
    for vtype, count in violation_types.items():
        print(f"    {vtype}: {count}")

    return reconstructions, originals, ports_out


# =========================================================
# PLOTTING
# =========================================================
def plot_reconstructions(originals, reconstructions, ports_out,
                         save_path="recon_samples.png"):
    n_samples = min(6, len(originals))
    fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))

    for i in range(n_samples):
        # BUG 7 FIX: ports_out[i] is now a list of port dicts, not (inlet_y, outlet_y)
        ports = ports_out[i]
        inlets  = [p for p in ports if p["type"] == "inlet"]
        outlets = [p for p in ports if p["type"] == "outlet"]
        in_str = [f"{p['wall']}@{p['center']}" for p in inlets]
        out_str = [f"{p['wall']}@{p['center']}" for p in outlets]

        label = f"in: {in_str}\nout: {out_str}"

        axes[0, i].imshow(originals[i].T, cmap='gray_r',
                          origin='lower', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original\n{label}', fontsize=6)
        axes[0, i].axis('off')

        axes[1, i].imshow(reconstructions[i].T, cmap='gray_r',
                          origin='lower', vmin=0, vmax=1)
        axes[1, i].set_title('Reconstructed', fontsize=7)
        axes[1, i].axis('off')

    plt.suptitle("Top: Original  |  Bottom: VAE Reconstruction", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


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
    print(f"Saved: {save_path}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    H5_PATH = "./data/new/dataset_all_merged.h5"

    train_loader, val_loader, test_loader, ds = make_loaders(
        h5_path=H5_PATH,
        batch_size=32,
        seed=np.random.randint(0, 10000)  # ← Random seed each run
    )

    model, history = train_vae(
        train_loader, val_loader,
        epochs=200,
        lr=1e-3,
        latent_dim=32,       # restored from 16
        beta=0.5,            # increased from 0.05
        w_bin=5.0,
        w_sharp=1.0,
        w_conn_max=1.0,      # reduced from 2.0
        w_conn_warmup=0,    # delayed from 1
        save_path="vae_best_new.pth",
        device=device
    )

    plot_history(history)

    reconstructions, originals, ports_out = evaluate_vae(
        model, test_loader, device
    )

    plot_reconstructions(originals, reconstructions, ports_out)

    print("\nAll done!")