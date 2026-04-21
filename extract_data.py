import h5py
import json
import glob
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm

DATA_DIR = "./data/new"
OUT_FILE = "./data/new1/filtered_one_outlet.h5"

os.makedirs("./data", exist_ok=True)

def wall_from_port_string(p):
    for wall in ["left", "right", "top", "bottom"]:
        if p.startswith(wall):
            return wall
    raise ValueError(f"Unknown port string: {p}")

def get_ports(meta):
    # dataset_all uses "inlets"/"outlets"
    # dataset_final uses "inlet_centers"/"outlet_centers" ← broken key, skip
    inlets  = meta.get("inlets",  [])
    outlets = meta.get("outlets", [])
    return inlets, outlets

def is_valid_bc(inlets, outlets):
    if len(outlets) != 1:
        return False
    if len(inlets) not in [1, 2]:
        return False
    outlet_wall = wall_from_port_string(outlets[0])
    inlet_walls = [wall_from_port_string(i) for i in inlets]
    if outlet_wall in inlet_walls:
        return False
    return True

# Only use dataset_all files — dataset_final has broken metadata keys
files = sorted(glob.glob(f"{DATA_DIR}/dataset_all_run*.h5"))
print(f"Found {len(files)} dataset_all files")

filtered_density, filtered_dp, filtered_vol, filtered_meta = [], [], [], []
total_seen = total_valid = 0

for file in files:
    print(f"\nReading {file}")
    with h5py.File(file, "r") as hf:
        density  = hf["density"][:]
        dp       = hf["pressure_drop"][:]
        vol      = hf["volume_fraction"][:]
        metadata = hf["metadata"][:]

        for i in tqdm(range(len(density))):
            total_seen += 1
            try:
                meta = json.loads(metadata[i].decode())
            except Exception:
                continue

            inlets, outlets = get_ports(meta)

            # Skip samples with missing port info (dataset_final contamination)
            if not inlets and not outlets:
                continue

            if not is_valid_bc(inlets, outlets):
                continue

            # Keep ALL feasible snapshots — don't deduplicate by BC config
            # Diversity across iterations is what the VAE needs
            filtered_density.append(density[i].astype(np.float32))
            filtered_dp.append(float(dp[i]))
            filtered_vol.append(float(vol[i]))
            filtered_meta.append(meta)
            total_valid += 1

print(f"\nTotal seen: {total_seen} | Valid kept: {total_valid}")

if total_valid == 0:
    raise RuntimeError("Nothing passed the filter — check metadata keys")

with h5py.File(OUT_FILE, "w") as hf:
    hf.create_dataset("density",
        data=np.stack(filtered_density), compression="gzip")
    hf.create_dataset("pressure_drop",
        data=np.array(filtered_dp, dtype=np.float32))
    hf.create_dataset("volume_fraction",
        data=np.array(filtered_vol, dtype=np.float32))
    hf.create_dataset("metadata",
        data=[json.dumps(m).encode() for m in filtered_meta])
    hf.attrs["num_designs"] = total_valid
    hf.attrs["created_on"]  = datetime.now().isoformat()
    hf.attrs["filter"]      = "1 outlet, 1-2 inlets, different walls, all iterations kept"

print(f"Saved → {OUT_FILE}")

# -------------------------------------------------------
# BUILD CHECKPOINT FOR GENERATOR
# -------------------------------------------------------

CHECKPOINT_OUT = "./data/new1/generation_checkpoint.json"

used_bc_configs = set()

for m in filtered_meta:
    inlets, outlets = get_ports(m)
    if not inlets or not outlets:
        continue
    bc_key = (tuple(sorted(inlets)), tuple(sorted(outlets)))
    used_bc_configs.add(bc_key)

print(f"Unique BC configs found: {len(used_bc_configs)}")

checkpoint = {
    "total_designs": len(used_bc_configs),  # generator counts BC configs
    "used_bc_configs": [
        [list(k[0]), list(k[1])] for k in sorted(used_bc_configs)
    ],
    "run_number": 0,
    "created_from_filter": datetime.now().isoformat(),
    "source_dataset": OUT_FILE
}

os.makedirs(os.path.dirname(CHECKPOINT_OUT), exist_ok=True)

with open(CHECKPOINT_OUT, "w") as f:
    json.dump(checkpoint, f, indent=2)

print("\nCheckpoint written →", CHECKPOINT_OUT)