"""
Lightweight CLI to run EV2Gym evaluations and save plots.

Examples
- python main_eval.py --agent eMPC_V2G --config paper_baseline.yaml --H 10 --seed 0
- python main_eval.py --agent OCMF_V2G --config V2G_MPC.yaml --H 10 --no-plots
"""

import argparse
from pathlib import Path

from ev2gym.models.ev2gym_env import EV2Gym
from ocmf_mpc import OCMF_V2G, OCMF_G2V
from eMPC import eMPC_V2G, eMPC_G2V
from rule_based import AFAP, ALAP


def build_agent(name: str, env, H: int, verbose: bool = False):
    name = (name or "eMPC_V2G").strip().lower()
    if name == "empc_v2g":
        return eMPC_V2G(env, control_horizon=H, verbose=verbose)
    if name == "empc_g2v":
        return eMPC_G2V(env, control_horizon=H, verbose=verbose)
    if name == "ocmf_v2g":
        return OCMF_V2G(env, control_horizon=H, verbose=verbose)
    if name == "ocmf_g2v":
        return OCMF_G2V(env, control_horizon=H, verbose=verbose)
    if name == "afap":
        return AFAP(env, verbose=verbose)
    if name == "alap":
        return ALAP(env, verbose=verbose)
    raise ValueError(
        "Unknown agent '{name}'. Choices: eMPC_V2G, eMPC_G2V, OCMF_V2G, OCMF_G2V, AFAP, ALAP"
    )


def main():
    parser = argparse.ArgumentParser(description="Run EV2Gym evaluation and save plots")
    parser.add_argument("--config", type=str, default="paper_baseline.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--agent",
        type=str,
        default="eMPC_V2G",
        help="Agent: eMPC_V2G|eMPC_G2V|OCMF_V2G|OCMF_G2V|AFAP|ALAP",
    )
    parser.add_argument("--H", type=int, default=10, help="Control horizon for MPC-based agents")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for environment reset")
    parser.add_argument("--plots", dest="plots", action="store_true", help="Enable plotting during env.step")
    parser.add_argument("--no-plots", dest="plots", action="store_false", help="Disable plotting during env.step")
    parser.set_defaults(plots=True)
    parser.add_argument("--verbose", action="store_true", help="Verbose agent/env logs")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        here = Path(__file__).resolve().parent
        alt = here / cfg_path.name
        if alt.exists():
            cfg_path = alt
        else:
            raise FileNotFoundError(f"Config not found: {args.config}")

    env = EV2Gym(
        config_file=str(cfg_path),
        load_from_replay_path=None,
        verbose=args.verbose,
        save_replay=False,
        save_plots=args.plots,
    )

    # Reset
    state, _ = env.reset(seed=args.seed)

    # Some quick environment stats
    ev_profiles = getattr(env, "EVs_profiles", [])
    try:
        max_time_of_stay = max(
            ev.time_of_departure - ev.time_of_arrival for ev in ev_profiles
        ) if ev_profiles else 0
        min_time_of_stay = min(
            ev.time_of_departure - ev.time_of_arrival for ev in ev_profiles
        ) if ev_profiles else 0
    except Exception:
        max_time_of_stay = min_time_of_stay = 0
    print(f"Number of EVs: {len(ev_profiles)}")
    print(f"Max time of stay: {max_time_of_stay}")
    print(f"Min time of stay: {min_time_of_stay}")

    agent = build_agent(args.agent, env, H=args.H, verbose=args.verbose)

    stats = {}
    for _ in range(env.simulation_length):
        actions = agent.get_action(env)
        out = env.step(actions, visualize=args.plots)
        if isinstance(out, tuple) and len(out) == 5:
            new_state, reward, done, _, stats = out
        elif isinstance(out, tuple) and len(out) == 4:
            new_state, reward, done, stats = out
        else:
            raise RuntimeError("Unexpected env.step() return signature")
        if done:
            break

    print("\nSimulation finished at step:", getattr(env, "current_step", "?"))
    if isinstance(stats, dict):
        print("Stats summary:")
        for k in [
            "total_profits",
            "average_user_satisfaction",
            "battery_degradation",
            "total_ev_served",
            "total_energy_charged",
            "total_energy_discharged",
            "tracking_error",
        ]:
            if k in stats:
                print(f"  - {k}: {stats[k]}")


if __name__ == "__main__":
    main()

