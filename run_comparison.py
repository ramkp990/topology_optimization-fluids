"""
random_sweep.py
===============
Generate N_CONFIGS random port configurations. For each config, run
LBM ESO / Latent Gradient / CMA-ES simultaneously in parallel threads
(each method is an independent subprocess), then collect results.

Usage:
    python random_sweep.py                        # 20 configs, 1 inlet
    python random_sweep.py --n_configs 10 --n_inlets 2
    python random_sweep.py --skip_lbm --skip_cmaes   # grad-opt only
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

# ─── constants ────────────────────────────────────────────────────────────────
Nx, Ny         = 64, 64
WALL_THICKNESS = 4
PORT_HEIGHT    = int(0.10 * Ny)   # 6
WALLS          = ["left", "right", "top", "bottom"]

METHOD_KEYS    = ["lbm_eso", "latent_grad", "cmaes"]

# Skip flags passed to comparison.py for each method
# When running method X, skip the other two
_SKIP_FLAGS = {
    "lbm_eso":     ["--skip_grad",  "--skip_cmaes"],
    "latent_grad": ["--skip_lbm",   "--skip_cmaes"],
    "cmaes":       ["--skip_lbm",   "--skip_grad"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# PORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def make_slot(center, height=PORT_HEIGHT):
    return slice(int(center - height // 2), int(center + height // 2))

def random_center(wall):
    margin   = WALL_THICKNESS + PORT_HEIGHT // 2 + 2
    axis_max = Ny if wall in ("left", "right") else Nx
    return int(np.random.randint(margin, axis_max - margin))

def sample_ports(n, port_type):
    ports, attempts = [], 0
    while len(ports) < n and attempts < 50:
        wall   = np.random.choice(WALLS)
        center = random_center(wall)
        ok = all(
            abs(center - (p["range"].start + p["range"].stop) // 2) >= PORT_HEIGHT + 5
            for p in ports if p["wall"] == wall
        )
        if ok:
            ports.append({"type": port_type, "wall": wall,
                          "range": make_slot(center), "center": center})
        attempts += 1
    return ports

def ports_overlap(a, b, gap=5):
    if a["wall"] != b["wall"]:
        return False
    return (a["range"].start - gap) < b["range"].stop and \
           b["range"].start < (a["range"].stop + gap)

def generate_valid_ports(n_inlets=1, seed=None, max_attempts=200):
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        np.random.seed(int(rng.integers(0, 2**31)))
        ports = sample_ports(n_inlets, "inlet") + sample_ports(1, "outlet")

        overlap   = any(ports_overlap(ports[i], ports[j])
                        for i in range(len(ports))
                        for j in range(i + 1, len(ports)))
        same_wall = bool({p["wall"] for p in ports if p["type"] == "outlet"} &
                         {p["wall"] for p in ports if p["type"] == "inlet"})

        if not overlap and not same_wall:
            return ports
    raise RuntimeError(f"Could not sample valid ports in {max_attempts} attempts.")

def ports_to_tag(ports):
    return "_".join(f"{p['type'][0]}{p['wall'][0]}{p['center']}" for p in ports)

def ports_to_desc(ports):
    return " | ".join(f"{p['type']}@{p['wall']}:{p['center']}" for p in ports)


# ═══════════════════════════════════════════════════════════════════════════════
# SUBPROCESS — chunk-based reader (avoids tqdm \r deadlock)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_subprocess_capture(cmd, log_path, cfg_seed):
    """
    Run cmd, capture all output to log_path.
    Returns (returncode, full_text).
    Output is NOT streamed live here — each method runs in its own thread
    so interleaving would be unreadable. Logs are printed after all finish.
    """
    env = dict(os.environ, SWEEP_SEED=str(cfg_seed))
    chunks = []
    proc   = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, bufsize=0, env=env)

    def _reader(pipe):
        for chunk in iter(lambda: pipe.read(256), b""):
            chunks.append(chunk)

    t = threading.Thread(target=_reader, args=(proc.stdout,))
    t.start()
    proc.wait()
    t.join()

    full_text = b"".join(chunks).decode("utf-8", errors="replace")
    Path(log_path).write_text(full_text, encoding="utf-8")
    return proc.returncode, full_text


def run_single_method(method, ports, args, cfg_output_dir):
    """
    Run comparison.py for exactly one method (other two skipped).
    Returns dict with keys: method, success, bin_dp, log_text.
    """
    method_dir = os.path.join(cfg_output_dir, method)
    os.makedirs(method_dir, exist_ok=True)

    cmd = [sys.executable, "comparison.py"]
    for p in ports:
        cmd += ["--port", p["type"], p["wall"], str(p["center"])]
    cmd += [
        "--vae_path",            args.vae_path,
        "--output_dir",          method_dir,
        "--target_volume",       str(args.target_volume),
        "--n_restarts",          str(args.n_restarts),
        "--n_steps",             str(args.n_steps),
        "--lr",                  str(args.lr),
        "--lambda_volume_grad",  str(args.lambda_volume_grad),
        "--lambda_binary",       str(args.lambda_binary),
        "--temp_start",          str(args.temp_start),
        "--temp_end",            str(args.temp_end),
        "--lambda_volume_cmaes", str(args.lambda_volume_cmaes),
        "--max_gen",             str(args.max_gen),
        "--popsize",             str(args.popsize),
        "--sigma0",              str(args.sigma0),
    ]
    # Skip the other two methods
    cmd += _SKIP_FLAGS[method]

    log_path    = os.path.join(method_dir, "run.log")
    rc, log_txt = _run_subprocess_capture(cmd, log_path, cfg_seed=args.seed)

    bin_dp = None
    if method == "lbm_eso":
        bin_dp = _parse_lbm_dp(log_txt)
    elif method == "latent_grad":
        bin_dp = _load_npz_dp("latgrad", ports_to_tag(ports), args.lambda_volume_grad)
    elif method == "cmaes":
        bin_dp = _load_npz_dp("cmaes",   ports_to_tag(ports), args.lambda_volume_cmaes)

    return {
        "method":   method,
        "dir":      method_dir,
        "success":  rc == 0,
        "bin_dp":   bin_dp,
        "log_text": log_txt,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_lbm_dp(text):
    m = re.search(r"Final\s+.p\s*=\s*([0-9]+\.[0-9]+)", text)
    if m:
        return float(m.group(1))
    matches = re.findall(r".p=([0-9]+\.[0-9]+)", text)
    return float(matches[-1]) if matches else None

def _load_npz_dp1(prefix, tag, lv):
    p = Path(f"{prefix}_result_{tag}_lv{lv}.npz")
    if p.exists():
        data = np.load(p, allow_pickle=True)
        return float(data["best_dp"]) if "best_dp" in data else None
    return None

def _load_npz_dp(prefix, tag, lv):
    import glob
    matches = glob.glob(f"{prefix}_result_{tag}_lv*.npz")   # any lv format
    if matches:
        data = np.load(matches[0], allow_pickle=True)
        return float(data["best_dp"]) if "best_dp" in data else None
    return None

def _load_topo(method_dir, tag, method):
    """Load topology array from the per-method comparison_{tag}.npz."""
    # comparison.py saves final_np_{method_key} — map to the right key
    method_key_map = {
        "lbm_eso":     "lbm_eso",
        "latent_grad": "latent_grad",
        "cmaes":       "cmaes",
    }
    path = Path(method_dir) / f"comparison_{tag}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    key  = f"final_np_{method_key_map[method]}"
    return data[key] if key in data else None

def _compute_bin_vol(arr, ports):
    if arr is None:
        return None
    try:
        from new_generate_dataset_multiple import build_bc_masks
        masks  = build_bc_masks(Nx, Ny, WALL_THICKNESS, ports)
        is_des = (~masks["solid_mask"] &
                  ~masks["fluid_mask"] &
                  ~masks["orifice_mask"]).cpu().numpy()
        return float((arr > 0.5)[is_des].mean())
    except Exception:
        return float((arr > 0.5).mean())


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.sweep_dir) / timestamp
    sweep_dir.mkdir(parents=True, exist_ok=True)

    csv_path = sweep_dir / "sweep_results.csv"
    npz_path = sweep_dir / "sweep_results.npz"

    active_methods = [m for m in METHOD_KEYS
                      if not getattr(args, f"skip_{m.replace('latent_grad','grad').replace('lbm_eso','lbm').replace('cmaes','cmaes')}")]

    # map CLI skip flags correctly
    def _is_active(m):
        if m == "lbm_eso":     return not args.skip_lbm
        if m == "latent_grad": return not args.skip_grad
        if m == "cmaes":       return not args.skip_cmaes
    active_methods = [m for m in METHOD_KEYS if _is_active(m)]

    fieldnames = ["config_idx", "tag", "ports_desc", "success"]
    for m in active_methods:
        fieldnames += [f"{m}_bin_vol", f"{m}_bin_dp"]

    all_rows  = []
    all_topos = {}

    print(f"\n{'='*70}")
    print(f"  Random Port Sweep — {args.n_configs} configs")
    print(f"  Methods (parallel): {active_methods}")
    print(f"  Sweep dir: {sweep_dir}")
    print(f"{'='*70}\n")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for cfg_idx in range(args.n_configs):
            print(f"\n{'─'*70}")
            print(f"  CONFIG {cfg_idx+1:02d} / {args.n_configs}")
            print(f"{'─'*70}")

            seed = args.seed + cfg_idx if args.seed is not None else None
            try:
                ports = generate_valid_ports(n_inlets=args.n_inlets, seed=seed)
            except RuntimeError as e:
                print(f"  [SKIP] {e}")
                continue

            tag  = ports_to_tag(ports)
            desc = ports_to_desc(ports)
            print(f"  Ports : {desc}")
            print(f"  Tag   : {tag}")
            print(f"  Launching {len(active_methods)} method(s) in parallel...\n")

            cfg_output_dir = str(sweep_dir / f"cfg{cfg_idx+1:02d}_{tag}")

            # ── Run all active methods simultaneously ─────────────────────
            method_results = {}   # method → result dict

            with ThreadPoolExecutor(max_workers=len(active_methods)) as ex:
                futures = {
                    ex.submit(run_single_method, m, ports, args, cfg_output_dir): m
                    for m in active_methods
                }
                for fut in as_completed(futures):
                    m   = futures[fut]
                    res = fut.result()
                    method_results[m] = res
                    status = "✓" if res["success"] else "✗"
                    dp_str = f"{res['bin_dp']:.6f}" if res["bin_dp"] is not None else "N/A"
                    print(f"  {status} [{m:12s}] done  |  bin_dp={dp_str}")

            # ── Print logs in order after all methods finish ───────────────
            print(f"\n{'─'*30} LOGS {'─'*30}")
            for m in active_methods:
                print(f"\n=== {m.upper()} ===")
                print(method_results[m]["log_text"])

            # ── Collect vol + build row ───────────────────────────────────
            row = {
                "config_idx": cfg_idx + 1,
                "tag":        tag,
                "ports_desc": desc,
                "success":    all(r["success"] for r in method_results.values()),
            }
            all_topos[tag] = {}

            for m in active_methods:
                res  = method_results[m]
                topo = _load_topo(res["dir"], tag, m)
                all_topos[tag][m] = topo

                bin_vol = _compute_bin_vol(topo, ports)
                bin_dp  = res["bin_dp"]

                row[f"{m}_bin_vol"] = f"{bin_vol:.4f}" if bin_vol is not None else "N/A"
                row[f"{m}_bin_dp"]  = f"{bin_dp:.6f}"  if bin_dp  is not None else "N/A"

            writer.writerow(row)
            csvfile.flush()
            all_rows.append(row)

            print(f"\n  Results for config {cfg_idx+1:02d}:")
            for m in active_methods:
                print(f"    {m:12s}  vol={row[f'{m}_bin_vol']:>8}  "
                      f"dp={row[f'{m}_bin_dp']:>10}")

    # ── Combined NPZ ──────────────────────────────────────────────────────────
    save_dict = {"rows_json": json.dumps(all_rows)}
    for tag, md in all_topos.items():
        for m, arr in md.items():
            if arr is not None:
                save_dict[f"{tag}__{m}"] = arr
    np.savez(str(npz_path), **save_dict)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SWEEP COMPLETE")
    print(f"  CSV : {csv_path}")
    print(f"  NPZ : {npz_path}")
    print(f"{'='*70}")
    hdr = f"  {'cfg':<5} {'tag':<35}"
    for m in active_methods:
        hdr += f"  {'vol':>7} {'dp':>10}"
    print(hdr)
    print("  " + "─" * (42 + len(active_methods) * 20))
    for row in all_rows:
        line = f"  {row['config_idx']:<5} {row['tag']:<35}"
        for m in active_methods:
            line += f"  {row.get(f'{m}_bin_vol','N/A'):>7}  {row.get(f'{m}_bin_dp','N/A'):>10}"
        print(line)
    print(f"\n  {len(all_rows)}/{args.n_configs} configs completed.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--n_configs",  type=int,   default=20)
    p.add_argument("--n_inlets",   type=int,   default=1)
    p.add_argument("--seed",       type=int,   default=10000)
    p.add_argument("--sweep_dir",  default="sweep_results")

    p.add_argument("--skip_lbm",   action="store_true")
    p.add_argument("--skip_grad",  action="store_true")
    p.add_argument("--skip_cmaes", action="store_true")

    p.add_argument("--vae_path",            default="vae_best_new.pth")
    p.add_argument("--target_volume",       type=float, default=0.20)

    p.add_argument("--n_restarts",          type=int,   default=1)
    p.add_argument("--n_steps",             type=int,   default=200)
    p.add_argument("--lr",                  type=float, default=0.05)
    p.add_argument("--lambda_volume_grad",  type=float, default=0.5)
    p.add_argument("--lambda_binary",       type=float, default=2.0)
    p.add_argument("--temp_start",          type=float, default=1.0)
    p.add_argument("--temp_end",            type=float, default=0.1)

    p.add_argument("--lambda_volume_cmaes", type=float, default=1)
    p.add_argument("--max_gen",             type=int,   default=40)
    p.add_argument("--popsize",             type=int,   default=24)
    p.add_argument("--sigma0",              type=float, default=0.5)

    return p.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())