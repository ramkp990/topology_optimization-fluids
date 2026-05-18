import os
import glob
import re
import argparse
import numpy as np


# ---------------------------------------------
# Extract base tag from filename
# ---------------------------------------------
def extract_base_tag(filename):
    """
    Examples:
      lbm_final_np_ir36_ib50_ol10.npy           -> ir36_ib50_ol10
      cmaes_final_np_ir36_ib50_ol10_lv0.4.npy   -> ir36_ib50_ol10
      latgrad_final_np_ir36_ib50_ol10_lv0.4.npy -> ir36_ib50_ol10
    """

    name = os.path.basename(filename)

    # remove prefix
    name = re.sub(r"^(lbm|latgrad|cmaes)_final_np_", "", name)

    # remove .npy
    name = name.replace(".npy", "")

    # remove optional _lv...
    name = re.sub(r"_lv[0-9.]+$", "", name)

    return name


# ---------------------------------------------
# Scan folder and group files by tag
# ---------------------------------------------
def group_files(folder):
    files = glob.glob(os.path.join(folder, "*_final_np_*.npy"))

    groups = {}
    for f in files:
        tag = extract_base_tag(f)
        groups.setdefault(tag, []).append(f)

    return groups


# ---------------------------------------------
# Load helpers
# ---------------------------------------------
def try_load(path):
    try:
        arr = np.load(path)
        print("  ✓", os.path.basename(path))
        return arr
    except Exception:
        print("  ✗ Failed:", os.path.basename(path))
        return None


# ---------------------------------------------
# Merge one tag
# ---------------------------------------------
def merge_tag(folder, tag, files):
    print("\n==============================")
    print("Merging:", tag)

    save_dict = {}

    for f in files:
        name = os.path.basename(f)

        if name.startswith("lbm_"):
            save_dict["final_np_lbm_eso"] = try_load(f)

        elif name.startswith("latgrad_"):
            save_dict["final_np_latent_grad"] = try_load(f)

        elif name.startswith("cmaes_"):
            save_dict["final_np_cmaes"] = try_load(f)

    # remove missing entries
    save_dict = {k: v for k, v in save_dict.items() if v is not None}

    if not save_dict:
        print("⚠ Nothing usable — skipping")
        return

    out_path = os.path.join(folder, f"comparison_{tag}.npz")
    np.savez(out_path, **save_dict)

    print("✔ Saved:", out_path)


# ---------------------------------------------
# Main
# ---------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="./comparison_results",)
    parser.add_argument("--tag", default=None,
                        help="Merge only one tag")
    args = parser.parse_args()

    groups = group_files(args.folder)

    if args.tag:
        if args.tag not in groups:
            print("Tag not found:", args.tag)
            return
        merge_tag(args.folder, args.tag, groups[args.tag])
    else:
        print("Found tags:", list(groups.keys()))
        for tag, files in groups.items():
            merge_tag(args.folder, tag, files)


if __name__ == "__main__":
    main()