import numpy as np
import argparse
import json
from pathlib import Path


# -------------------------------------------------------
# Compute surface area (perimeter) of fluid region
# -------------------------------------------------------
def compute_perimeter(fluid):
    """
    fluid: binary array (1=fluid, 0=solid)
    returns discrete perimeter length
    """
    # pad with solid so edges count as walls
    padded = np.pad(fluid, 1, mode="constant", constant_values=0)

    up    = padded[:-2, 1:-1]
    down  = padded[2:, 1:-1]
    left  = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    center = padded[1:-1, 1:-1]

    # Count fluid pixels touching solid
    boundary = (
        (center == 1) & (up == 0) +
        (center == 1) & (down == 0) +
        (center == 1) & (left == 0) +
        (center == 1) & (right == 0)
    )

    return boundary.sum()


# -------------------------------------------------------
# Surface area to volume ratio
# -------------------------------------------------------
def surface_to_volume_ratio(arr):
    fluid = (arr > 0.5).astype(np.uint8)

    volume = fluid.sum()
    perimeter = compute_perimeter(fluid)

    ratio = perimeter / volume
    return ratio, perimeter, volume


# -------------------------------------------------------
def main(tag):
    folder = "./comparison_results"
    fname = f"comparison_{tag}.npz"
    path = Path(folder) / fname

    #path = Path(fname)

    if not path.exists():
        raise FileNotFoundError(f"{fname} not found")

    data = np.load(path)
    print(data.files)

    results = {}

    print("\nSurface Area / Volume Ratio")
    print("-----------------------------------")

    for method in ["final_np_lbm_eso", "final_np_latent_grad", "final_np_cmaes"]:
        arr = data[method]
        ratio, perimeter, volume = surface_to_volume_ratio(arr)

        results[method] = {
            "ratio": float(ratio),
            "perimeter": int(perimeter),
            "volume": int(volume),
        }

        print(f"{method:8s}  SA/V = {ratio:.4f}   perimeter={perimeter}   volume={volume}")

    # Save results for thesis plots later
    out_file = f"metric_sav_{tag}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved → {out_file}")


# -------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("tag", help="e.g. ir21_ib50_ot10")
    args = parser.parse_args()

    main(args.tag)