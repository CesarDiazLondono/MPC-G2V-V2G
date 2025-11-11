"""
Compare SoC evolution at one EVSE port for two MPC baselines: eMPC_V2G and OCMF_V2G.

This reproduces the single-panel SoC plot with two step-like lines (one per MPC),
using the same scenario and seed. The script auto-picks a "busy" port if the
requested one has no EVs connected during the day.

Usage (requires Gurobi for MPC)
  python scripts/plot_soc_mpc_compare.py \
      --config paper_baseline.yaml --seed 0 \
      --cs 0 --port 0 --H 10 \
      --out analysis_out/soc_MPC_compare.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ev2gym.models.ev2gym_env import EV2Gym

# Import MPC agents (needs gurobipy installed)
try:
    from eMPC import eMPC_V2G  # type: ignore
    from ocmf_mpc import OCMF_V2G  # type: ignore
except ModuleNotFoundError:
    import sys as _sys
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from eMPC import eMPC_V2G  # type: ignore
    from ocmf_mpc import OCMF_V2G  # type: ignore


def _time_axis(env) -> np.ndarray:
    start_h = int(getattr(env, "hour", 0))
    start_m = int(getattr(env, "minute", 0))
    ts_min = float(getattr(env, "timescale", 15))
    steps = int(getattr(env, "simulation_length", 96))
    minutes = start_h * 60 + start_m + np.arange(steps) * ts_min
    return (minutes % (24 * 60)) / 60.0


def _soc_of(ev) -> float:
    if ev is None:
        return np.nan
    cap = float(getattr(ev, "current_capacity", np.nan))
    bcap = float(getattr(ev, "battery_capacity", np.nan))
    if not np.isfinite(cap) or not np.isfinite(bcap) or bcap <= 0:
        return np.nan
    return cap / bcap


def _scan_connectivity(cfg_path: str, seed: int) -> Dict[Tuple[int, int], np.ndarray]:
    env = EV2Gym(config_file=cfg_path, verbose=False, save_replay=False, save_plots=False)
    env.reset(seed=seed)
    L = env.simulation_length
    soc_map: Dict[Tuple[int, int], np.ndarray] = {}
    for ci, cs in enumerate(env.charging_stations):
        soc_map.update({(ci, pj): np.full(L, np.nan, dtype=float) for pj in range(cs.n_ports)})
    for t in range(L):
        for ci, cs in enumerate(env.charging_stations):
            evs = getattr(cs, "evs_connected", [])
            for pj in range(min(len(evs), cs.n_ports)):
                soc_map[(ci, pj)][t] = _soc_of(evs[pj])
        # step with zero action (no charge/discharge) just to advance time
        a = np.zeros(env.number_of_ports)
        out = env.step(a, visualize=False)
        done = bool(out[2]) if isinstance(out, tuple) else False
        if done:
            break
    return soc_map


def _choose_port(cfg_path: str, seed: int, fallback: Tuple[int, int]) -> Tuple[int, int]:
    soc_map = _scan_connectivity(cfg_path, seed)
    def score(arr: np.ndarray) -> int:
        finite = np.isfinite(arr).astype(int)
        sess = int(np.sum((finite[1:] == 1) & (finite[:-1] == 0)) + (finite[0] == 1))
        occ = int(finite.sum())
        return sess * 1_000_000 + occ
    best = max(soc_map.items(), key=lambda kv: score(kv[1]))[0] if soc_map else fallback
    return best


def _track_with_agent(cfg_path: str, seed: int, H: int, port: Tuple[int, int], agent_kind: str) -> Tuple[np.ndarray, np.ndarray]:
    env = EV2Gym(config_file=cfg_path, verbose=False, save_replay=False, save_plots=False)
    env.reset(seed=seed)
    xs = _time_axis(env)
    cs_idx, pj = port
    if agent_kind.lower() == "empc":
        agent = eMPC_V2G(env, control_horizon=H, verbose=False)
    elif agent_kind.lower() == "ocmf":
        agent = OCMF_V2G(env, control_horizon=H, verbose=False)
    else:
        raise ValueError("agent_kind must be 'empc' or 'ocmf'")

    soc = np.full(env.simulation_length, np.nan, dtype=float)
    for t in range(env.simulation_length):
        try:
            cs = env.charging_stations[cs_idx]
            evs = getattr(cs, "evs_connected", [])
            ev = evs[pj] if 0 <= pj < len(evs) else None
        except Exception:
            ev = None
        soc[t] = _soc_of(ev)

        a = agent.get_action(env)
        out = env.step(a, visualize=False)
        done = bool(out[2]) if isinstance(out, tuple) else False
        if done:
            break
    return xs, soc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="paper_baseline.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--H", type=int, default=10, help="Control horizon for MPC")
    parser.add_argument("--cs", type=int, default=0)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--out", type=str, default="analysis_out/soc_MPC_compare.png")
    args = parser.parse_args()

    cfg = Path(args.config)
    if not cfg.exists():
        alt = Path(__file__).resolve().parents[1] / cfg.name
        if alt.exists():
            cfg = alt
        else:
            raise FileNotFoundError(f"Config not found: {args.config}")

    # pick port if empty
    pick = _choose_port(str(cfg), args.seed, (args.cs, args.port))

    xs, soc_empc = _track_with_agent(str(cfg), args.seed, args.H, pick, "empc")
    _,  soc_ocmf = _track_with_agent(str(cfg), args.seed, args.H, pick, "ocmf")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4.6, 2.8), dpi=170)
    plt.plot(xs, soc_empc, color="#e377c2", linewidth=1.8, label="eMPC_V2G")
    plt.plot(xs, soc_ocmf, color="#1f77b4", linewidth=1.8, linestyle="--", label="OCMF_V2G")
    plt.ylim(0.0, 1.02)
    plt.xlim(0, 24)
    ticks = np.arange(0, 25, 4)
    plt.xticks(ticks, [f"{int(t)}:00" for t in ticks])
    plt.xlabel("Time")
    plt.ylabel("SoC")
    plt.grid(True, alpha=0.35)
    plt.title(f"CS {pick[0]}, Port {pick[1]} (seed={args.seed})")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
