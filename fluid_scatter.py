'''
import pandas as pd, matplotlib.pyplot as plt
import os
os.makedirs("figs", exist_ok=True)        # add this
plt.tight_layout(); plt.savefig("figs/fluid_scatter.png", dpi=150)
df = pd.read_csv("master_results.csv").dropna(subset=["lg_dp","eso_dp"])
m = max(df.eso_dp.max(), df.lg_dp.max())
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(df.eso_dp, df.lg_dp, s=18, alpha=0.6, color="#1a6fb5")
ax.plot([0,m],[0,m], "k--", lw=1, label="parity")
ax.set_xlabel(r"$\Delta p$  (LBM-ESO)"); ax.set_ylabel(r"$\Delta p$  (latent gradient)")
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig("figs/fluid_scatter.png", dpi=150)
'''

"""
resimulate_eso.py
=================
Re-run the LBM forward pass on each saved (post-removal) ESO topology to obtain
the CORRECT pressure drop for that exact design, fixing the off-by-one in
LBM_Multiple.main() where the reported Δp belonged to the pre-removal topology.

Prints the corrected ESO Δp per config (and a CSV block at the end to paste).

Usage:
    python resimulate_eso.py
    python resimulate_eso.py "/path/to/.../sweep_results/2026*/cfg*/lbm_eso/lbm_final_*.npy"
"""

import os
import re
import sys
import glob
import time

import numpy as np
import torch

# ----------------------------------------------------------------------
# 0. Default location of the saved ESO topologies (override via argv[1])
# ----------------------------------------------------------------------
DEFAULT_GLOB = ("/Users/ramankp/Documents/thesis/output_8/fluid/comparison/"
                "sweep_results/best/2026*/cfg*/lbm_eso/lbm_final_*.npy")

PATTERN = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GLOB
ALPHA_MAX = 100.0   # same alpha_max_constant the sweep used

# ----------------------------------------------------------------------
# 1. Import LBM_Multiple with a dummy CLI so its argparse doesn't crash.
#    (It builds masks for these dummy ports at import; we overwrite them
#     per config below, so the dummy choice is irrelevant.)
# ----------------------------------------------------------------------
sys.argv = ["LBM_Multiple.py",
            "--port", "inlet",  "left",  "32", "6",
            "--port", "outlet", "right", "32", "6"]
try:
    import LBM_Multiple as L
except Exception as e:
    print("ERROR: could not import LBM_Multiple.py — run this script from the "
          "directory that contains it (or add it to PYTHONPATH).")
    print(f"       {e}")
    sys.exit(1)

torch.set_grad_enabled(False)   # forward-only; we never call backward here

# ----------------------------------------------------------------------
# 2. Tag -> ports, using LBM_Multiple's OWN make_slot so the port slices
#    (and therefore the masks) are built identically to the solver.
#    Sampled ports keep a >=9-cell margin, so no edge clamping is needed.
# ----------------------------------------------------------------------
_WALLMAP = {"l": "left", "r": "right", "t": "top", "b": "bottom"}

def ports_from_tag(tag):
    ports = []
    for tok in str(tag).split("_"):
        if len(tok) < 3:
            continue
        ptype = "inlet" if tok[0] == "i" else "outlet" if tok[0] == "o" else None
        wall  = _WALLMAP.get(tok[1])
        if ptype is None or wall is None:
            continue
        try:
            center = int(tok[2:])
        except ValueError:
            continue
        ports.append({"type": ptype, "wall": wall,
                            "range": L.make_slot(center, L.PORT_HEIGHT), "center": center})
    return ports

# ----------------------------------------------------------------------
# 3. Reset LBM_Multiple's module-global masks/ports for one config,
#    matching the module-level setup in LBM_Multiple exactly.
# ----------------------------------------------------------------------
def set_config(ports):
    W = L.WALL_THICKNESS
    L.solid_mask   = torch.zeros(L.Nx, L.Ny, device=L.device, dtype=torch.bool)
    L.orifice_mask = torch.zeros_like(L.solid_mask)
    L.fluid_mask   = torch.zeros_like(L.solid_mask)

    # outer walls
    L.solid_mask[0:W, :]  = True
    L.solid_mask[-W:, :]  = True
    L.solid_mask[:, 0:W]  = True
    L.solid_mask[:, -W:]  = True

    # ports
    L.ports        = ports
    L.inlet_ports  = [p for p in ports if p["type"] == "inlet"]
    L.outlet_ports = [p for p in ports if p["type"] == "outlet"]
    for p in ports:
        L.carve_port(p)   # mutates L.solid_mask / L.orifice_mask / L.fluid_mask

    L.fluid_dilated = torch.nn.functional.max_pool2d(
        L.orifice_mask.float().unsqueeze(0).unsqueeze(0),
        3, stride=1, padding=1)[0, 0].bool()

def tag_from_cfgdir(npy_path):
    # .../cfgNN_<tag>/lbm_eso/lbm_final_*.npy  ->  <tag>
    cfg_dir  = os.path.dirname(os.path.dirname(npy_path))
    cfg_name = os.path.basename(cfg_dir)
    m = re.match(r"^cfg\d+_(.+)$", cfg_name)
    return (m.group(1) if m else cfg_name), cfg_name

# ----------------------------------------------------------------------
# 4. Walk the saved topologies, re-simulate, print corrected Δp.
# ----------------------------------------------------------------------
files = sorted(glob.glob(PATTERN))
if not files:
    print(f"No files matched:\n  {PATTERN}")
    sys.exit(1)

print(f"Found {len(files)} ESO topology file(s).\n")
print(f"{'cfg':<22} {'tag':<28} {'corrected_dp':>14}")
print("-" * 66)

rows = []   # (tag, cfg_name, dp)
for i, npy_path in enumerate(files):
    tag, cfg_name = tag_from_cfgdir(npy_path)
    ports = ports_from_tag(tag)
    if not ports or not any(p["type"] == "inlet" for p in ports) \
                 or not any(p["type"] == "outlet" for p in ports):
        print(f"{cfg_name:<22} {tag:<28} {'SKIP (bad tag)':>14}")
        continue

    arr = np.load(npy_path)
    if arr.shape != (L.Nx, L.Ny):
        print(f"{cfg_name:<22} {tag:<28} {'SKIP shape '+str(arr.shape):>14}")
        continue

    set_config(ports)
    L.topology = torch.tensor(arr, dtype=torch.float32, device=L.device)

    t0 = time.time()
    dp = L.simulate(ALPHA_MAX).item()
    dt = time.time() - t0

    rows.append((tag, cfg_name, dp))
    print(f"{cfg_name:<22} {tag:<28} {dp:>14.6f}   ({dt:.1f}s)")

# ----------------------------------------------------------------------
# 5. CSV block for easy paste into your master sheet.
# ----------------------------------------------------------------------
print("\n=== CSV (tag,eso_dp_corrected) ===")
print("tag,eso_dp_corrected")
for tag, cfg_name, dp in rows:
    print(f"{tag},{dp:.6f}")

print(f"\n{len(rows)} configs re-simulated.")