"""
Merge multiple dataset_all HDF5 files into one.
Compatible with NEW generic-port dataset format.
"""

import h5py
import numpy as np
import os
from datetime import datetime

INPUT_FILES = [
    "./data/new1/filtered_one_outlet.h5",
    "./data/new1/dataset_all_run2.h5",
    "./data/new1/dataset_all_run3.h5",
    "./data/new1/dataset_all_run4.h5",
    "./data/new1/dataset_all_run5.h5",
    "./data/new1/dataset_all_run6.h5",
    "./data/new1/dataset_all_run7.h5",
    "./data/new1/dataset_all_run8.h5",
    "./data/new1/dataset_all_run9.h5",
    "./data/new1/dataset_all_run10.h5",
    "./data/new1/dataset_all_run11.h5",
    "./data/new1/dataset_all_run12.h5",
    "./data/new1/dataset_all_run13.h5",
    "./data/new1/dataset_all_run14.h5",
    "./data/new1/dataset_all_run15.h5",
    "./data/new1/dataset_all_run16.h5",
    "./data/new1/dataset_all_run17.h5",
    "./data/new1/dataset_all_run18.h5",
    "./data/new1/dataset_all_run19.h5",
    "./data/new1/dataset_all_run20.h5",
    "./data/new1/dataset_all_run21.h5",
]

OUTPUT_FILE = "./data/new1/dataset_all_merged.h5"

ARRAY_KEYS = [
    "density",
    "pressure_drop",
    "volume_fraction",
    "eso_iteration",
    "optimization_id",
]

data = {k: [] for k in ARRAY_KEYS}
metadata_all = []
total = 0

print("📂 Loading files...")
for path in INPUT_FILES:
    if not os.path.exists(path):
        print(f"   ⚠️ Skipping missing file: {path}")
        continue

    with h5py.File(path, "r") as f:
        n = f["density"].shape[0]
        print(f"   {path}: {n} designs")

        # --- REQUIRED fields (always exist) ---
        data["density"].append(f["density"][:])
        data["pressure_drop"].append(f["pressure_drop"][:])
        data["volume_fraction"].append(f["volume_fraction"][:])

        # --- OPTIONAL fields (new schema) ---
        if "eso_iteration" in f:
            data["eso_iteration"].append(f["eso_iteration"][:])
        else:
            print("      ↳ adding dummy eso_iteration")
            data["eso_iteration"].append(np.full(n, -1, dtype=np.int32))

        if "optimization_id" in f:
            data["optimization_id"].append(f["optimization_id"][:])
        else:
            print("      ↳ adding dummy optimization_id")
            data["optimization_id"].append(np.full(n, -1, dtype=np.int32))

        metadata_all.extend([m.decode("utf-8") for m in f["metadata"][:]])
        total += n

if total == 0:
    print("❌ No data found.")
    exit()

print(f"\n🔗 Merging {total} designs...")

merged = {k: np.concatenate(data[k], axis=0) for k in ARRAY_KEYS}
metadata_all = np.array(metadata_all, dtype="S")

print("💾 Writing merged file...")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with h5py.File(OUTPUT_FILE, "w") as f:
    for k,v in merged.items():
        f.create_dataset(k, data=v, compression="gzip")

    f.create_dataset("metadata", data=metadata_all)

    f.attrs["num_designs"] = total
    f.attrs["merged_at"] = datetime.now().isoformat()
    f.attrs["dataset_type"] = "all_feasible_designs"

print("✅ Done!")
print(f"Saved → {OUTPUT_FILE}")