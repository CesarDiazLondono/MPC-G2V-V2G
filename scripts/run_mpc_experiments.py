"""
Run eMPC_V2G and OCMF_V2G with EV2Gym and save a summary CSV.

Usage
  python scripts/run_mpc_experiments.py --config paper_baseline.yaml --H 10 --seeds 0 1
"""

import argparse
import csv
import os
from pathlib import Path

from ev2gym.models.ev2gym_env import EV2Gym
from eMPC import eMPC_V2G
from ocmf_mpc import OCMF_V2G


def run_once(cfg: str, algo: str, H: int, seed: int) -> dict:
    env = EV2Gym(config_file=cfg, verbose=False, save_replay=False, save_plots=False)
    env.reset(seed=seed)
    agent = eMPC_V2G(env, control_horizon=H) if algo == "eMPC_V2G" else OCMF_V2G(env, control_horizon=H)

    stats = {}
    for _ in range(env.simulation_length):
        a = agent.get_action(env)
        out = env.step(a, visualize=False)
        if isinstance(out, tuple) and len(out) == 5:
            _, _, done, _, stats = out
        elif isinstance(out, tuple) and len(out) == 4:
            _, _, done, stats = out
        else:
            done = False
        if done:
            break

    return {
        "algo": algo,
        "seed": seed,
        "profits": float(stats.get("total_profits", 0.0)),
        "served": int(stats.get("total_ev_served", 0)),
        "satisfaction": float(stats.get("average_user_satisfaction", 0.0)),
        "degradation": float(stats.get("battery_degradation", 0.0)),
        "energy_c": float(stats.get("total_energy_charged", 0.0)),
        "energy_d": float(stats.get("total_energy_discharged", 0.0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="paper_baseline.yaml")
    p.add_argument("--H", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--out", default="analysis_out/mpc_summary.csv")
    args = p.parse_args()

    cfg = Path(args.config)
    if not cfg.exists():
        alt = Path(__file__).resolve().parents[1] / cfg.name
        if alt.exists():
            cfg = alt
        else:
            raise FileNotFoundError(f"Config not found: {args.config}")

    rows = []
    for algo in ["eMPC_V2G", "OCMF_V2G"]:
        for s in args.seeds:
            print(f"Running {algo} seed={s} H={args.H} ...")
            rows.append(run_once(str(cfg), algo, args.H, s))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
