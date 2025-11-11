"""
Plot SoC evolution at a single EVSE port for AFAP vs ALAP.

This reproduces the "top-left" style panel: time on X, SoC on Y,
two lines (AFAP solid, ALAP dashed) for the same port under the
same scenario/seed.

Usage
  python scripts/plot_soc_rule_baselines.py \
      --config paper_baseline.yaml --seed 0 \
      --cs 0 --port 0 --out analysis_out/soc_AFAP_ALAP.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict, Tuple as TTuple

import numpy as np
import matplotlib.pyplot as plt

from ev2gym.models.ev2gym_env import EV2Gym

# Ensure project root (parent of 'scripts') is importable when running this file directly
try:
    from rule_based import AFAP, ALAP  # type: ignore
except ModuleNotFoundError:
    import sys as _sys
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from rule_based import AFAP, ALAP  # type: ignore


def _time_axis(env) -> np.ndarray:
    # hour:minute start from config; env.timescale is minutes per step
    start_h = int(getattr(env, "hour", 0))
    start_m = int(getattr(env, "minute", 0))
    ts_min = float(getattr(env, "timescale", 15))
    steps = int(getattr(env, "simulation_length", 96))
    minutes = start_h * 60 + start_m + np.arange(steps) * ts_min
    return (minutes % (24 * 60)) / 60.0  # hours in [0,24)


def _get_soc(ev) -> float:
    if ev is None:
        return np.nan
    cap = float(getattr(ev, "current_capacity", np.nan))
    max_cap = float(
        getattr(ev, "max_battery_capacity", getattr(ev, "battery_capacity", np.nan))
    )
    if not np.isfinite(cap) or not np.isfinite(max_cap) or max_cap <= 0:
        return np.nan
    return cap / max_cap


def _track_soc(env: EV2Gym, policy, cs_idx: int, port_idx: int) -> np.ndarray:
    soc = np.full(env.simulation_length, np.nan, dtype=float)
    agent = policy(env)
    for t in range(env.simulation_length):
        # read SoC BEFORE action (state at time t)
        try:
            cs = env.charging_stations[cs_idx]
            evs = getattr(cs, "evs_connected", [])
            ev = evs[port_idx] if 0 <= port_idx < len(evs) else None
        except Exception:
            ev = None
        soc[t] = _get_soc(ev)

        a = agent.get_action(env)
        out = env.step(a, visualize=False)
        if isinstance(out, tuple):
            if len(out) == 5:
                _, _, done, _, _ = out
            else:
                _, _, done, _ = out
        else:
            done = False
        if done:
            break
    return soc


def _track_all_ports(env: EV2Gym, policy) -> Dict[TTuple[int, int], np.ndarray]:
    agent = policy(env)
    L = env.simulation_length
    soc_map: Dict[TTuple[int, int], np.ndarray] = {}
    # pre-allocate arrays for each port
    for ci, cs in enumerate(env.charging_stations):
        for pj in range(cs.n_ports):
            soc_map[(ci, pj)] = np.full(L, np.nan, dtype=float)

    for t in range(L):
        # record SoC before action
        for ci, cs in enumerate(env.charging_stations):
            evs = getattr(cs, "evs_connected", [])
            for pj in range(min(len(evs), cs.n_ports)):
                soc_map[(ci, pj)][t] = _get_soc(evs[pj])

        a = agent.get_action(env)
        out = env.step(a, visualize=False)
        if isinstance(out, tuple):
            done = bool(out[2])
        else:
            done = False
        if done:
            break
    return soc_map


def run_once(cfg_path: str, seed: int, cs_idx: int, port_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int,int]]:
    env = EV2Gym(config_file=cfg_path, verbose=False, save_replay=False, save_plots=False)
    env.reset(seed=seed)
    xs = _time_axis(env)

    # First try requested port; if empty, auto-pick a port with most valid SoC
    soc_afap = _track_soc(env, AFAP, cs_idx, port_idx)
    pick = (cs_idx, port_idx)
    if np.isnan(soc_afap).all():
        # re-create env and scan
        env = EV2Gym(config_file=cfg_path, verbose=False, save_replay=False, save_plots=False)
        env.reset(seed=seed)
        soc_map = _track_all_ports(env, AFAP)
        # pick port emphasizing more sessions, then occupancy length
        def _count_sessions(arr: np.ndarray) -> int:
            finite = np.isfinite(arr).astype(int)
            if finite.size == 0:
                return 0
            return int(np.sum((finite[1:] == 1) & (finite[:-1] == 0)) + (finite[0] == 1))

        best_key = None
        best_score = -1
        for k, arr in soc_map.items():
            cnt = int(np.isfinite(arr).sum())
            sess = _count_sessions(arr)
            score = sess * 1_000_000 + cnt
            if score > best_score:
                best_score = score
                best_key = k
        if best_key is None:
            best_key = (0, 0)
        pick = best_key
        # re-run AFAP on the chosen port for aligned length
        env = EV2Gym(config_file=cfg_path, verbose=False, save_replay=False, save_plots=False)
        env.reset(seed=seed)
        soc_afap = _track_soc(env, AFAP, pick[0], pick[1])

    # ALAP on the chosen port
    env2 = EV2Gym(config_file=cfg_path, verbose=False, save_replay=False, save_plots=False)
    env2.reset(seed=seed)
    soc_alap = _track_soc(env2, ALAP, pick[0], pick[1])
    return xs, soc_afap, soc_alap, pick


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="paper_baseline.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cs", type=int, default=0, help="Charging station index")
    parser.add_argument("--port", type=int, default=0, help="Port index within the CS")
    parser.add_argument("--out", type=str, default="analysis_out/soc_AFAP_ALAP.png")
    args = parser.parse_args()

    cfg = Path(args.config)
    if not cfg.exists():
        alt = Path(__file__).resolve().parents[1] / cfg.name
        if alt.exists():
            cfg = alt
        else:
            raise FileNotFoundError(f"Config not found: {args.config}")

    xs, afap, alap, pick = run_once(str(cfg), args.seed, args.cs, args.port)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Style closer to reference: 0-24h axis, clearer ticks, thicker lines
    plt.figure(figsize=(4.6, 2.8), dpi=170)
    plt.plot(xs, afap, label="AFAP", color="#d62728", linewidth=1.8)
    plt.plot(xs, alap, label="ALAP", color="#1f77b4", linestyle="--", linewidth=1.8)
    plt.ylim(0, 1.02)
    plt.xlim(0, 24)
    ticks = np.arange(0, 25, 4)
    plt.xticks(ticks, [f"{int(t)}:00" for t in ticks])
    plt.xlabel("Time")
    plt.ylabel("SoC")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.title(f"SoC at CS {pick[0]}, Port {pick[1]} (seed={args.seed})")
    plt.tight_layout()
    outp = Path(args.out)
    plt.savefig(outp)
    plt.close()
    print(f"Saved: {outp.as_posix()}")


if __name__ == "__main__":
    main()
