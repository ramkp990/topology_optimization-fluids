"""
Merge multiple dataset_final HDF5 files into one.
Usage: python merge_datasets.py
"""

import h5py
import numpy as np
import os
from datetime import datetime

# ---------------------------------------------------------
# Files to merge
# ---------------------------------------------------------
INPUT_FILES = [
    "./data/dataset_all_run1.h5",
    "./data/dataset_all_run2.h5",
    "./data/dataset_all_run4.h5",
]
OUTPUT_FILE = "./data/dataset_all_merged.h5"

# ---------------------------------------------------------
# Keys to merge (arrays)
# ---------------------------------------------------------
ARRAY_KEYS = [
    'density',
    'pressure_drop',
    'volume_fraction',
    'bc_inlet_y',
    'bc_outlet_y',
    'bc_height_diff',
    'eso_iteration',
    'optimization_id',
    'is_intermediate',
]

# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------
data = {k: [] for k in ARRAY_KEYS}
metadata_all = []
total = 0

print("📂 Loading files...")
for path in INPUT_FILES:
    if not os.path.exists(path):
        print(f"   ⚠️  Skipping (not found): {path}")
        continue

    with h5py.File(path, 'r') as f:
        n = f['density'].shape[0]
        print(f"   {path}: {n} designs")

        for k in ARRAY_KEYS:
            if k in f:
                data[k].append(f[k][:])
            else:
                # Some older files may be missing target_volume etc.
                # Fill with zeros so shape is consistent
                print(f"      ⚠️  Key '{k}' missing — filling with zeros")
                shape = (n,) if k != 'density' else (n, 64, 64)
                data[k].append(np.zeros(shape))

        # Metadata strings (optional key)
        if 'metadata' in f:
            meta = [m.decode('utf-8') for m in f['metadata'][:]]
            metadata_all.extend(meta)
        else:
            metadata_all.extend(['{}'] * n)

        total += n

if total == 0:
    print("❌ No data found. Check file paths.")
    exit()

# ---------------------------------------------------------
# Concatenate
# ---------------------------------------------------------
print(f"\n🔗 Merging {total} designs...")
merged = {}
for k in ARRAY_KEYS:
    if data[k]:
        merged[k] = np.concatenate(data[k], axis=0)

# Re-index optimization_id to be globally unique
merged['optimization_id'] = np.arange(total)

# ---------------------------------------------------------
# Duplicate check
# ---------------------------------------------------------
inlet_y  = merged['bc_inlet_y']
outlet_y = merged['bc_outlet_y']
configs  = list(zip(inlet_y.tolist(), outlet_y.tolist()))
unique   = set(configs)
n_dupes  = len(configs) - len(unique)

print(f"   Total designs:     {total}")
print(f"   Unique BC configs: {len(unique)}")
print(f"   Duplicates:        {n_dupes}")
if n_dupes > 0:
    print(f"   ⚠️  Duplicates exist — same (inlet_y, outlet_y) across runs.")
    print(f"      This is fine for training but worth knowing.")

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with h5py.File(OUTPUT_FILE, 'w') as f:
    # Main arrays
    f.create_dataset('density',          data=merged['density'],         compression='gzip')
    f.create_dataset('pressure_drop',    data=merged['pressure_drop'])
    f.create_dataset('volume_fraction',  data=merged['volume_fraction'])
    f.create_dataset('bc_inlet_y',       data=merged['bc_inlet_y'])
    f.create_dataset('bc_outlet_y',      data=merged['bc_outlet_y'])
    f.create_dataset('bc_height_diff',   data=merged['bc_height_diff'])
    f.create_dataset('eso_iteration',    data=merged['eso_iteration'])
    f.create_dataset('optimization_id',  data=merged['optimization_id'])
    f.create_dataset('is_intermediate',  data=merged['is_intermediate'])

    # Metadata strings
    f.create_dataset('metadata', data=[m.encode('utf-8') for m in metadata_all])

    # Attributes
    f.attrs['num_designs']   = total
    f.attrs['dataset_type']  = 'merged_final'
    f.attrs['source_files']  = str(INPUT_FILES)
    f.attrs['nx']            = 64
    f.attrs['ny']            = 64
    f.attrs['n_duplicates']  = n_dupes
    f.attrs['timestamp']     = datetime.now().isoformat()

print(f"\n✅ Saved: {OUTPUT_FILE}")
print(f"   Designs:        {total}")
print(f"   Pressure range: [{merged['pressure_drop'].min():.4f}, "
      f"{merged['pressure_drop'].max():.4f}]")
print(f"   Volume range:   [{merged['volume_fraction'].min():.3f}, "
      f"{merged['volume_fraction'].max():.3f}]")
print(f"   Inlet Y range:  [{merged['bc_inlet_y'].min()}, "
      f"{merged['bc_inlet_y'].max()}]")
print(f"   Outlet Y range: [{merged['bc_outlet_y'].min()}, "
      f"{merged['bc_outlet_y'].max()}]")