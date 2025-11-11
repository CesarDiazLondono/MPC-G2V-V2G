"""
Evaluate a trained PPO/SAC model on EV2Gym and generate plots if enabled.

Usage (bash):
  python scripts/eval_rl.py --algo ppo --config paper_baseline.yaml \
         --model results/ppo_ev2gym.zip
"""

import argparse
from pathlib import Path
import numpy as np

from ev2gym.models.ev2gym_env import EV2Gym
from rl_agents import EV2GymSB3Wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac"], required=True)
    ap.add_argument("--config", default="paper_baseline.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--det", action="store_true", help="Deterministic actions")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    env = EV2Gym(config_file=str(cfg_path), save_replay=False, save_plots=True, verbose=True)

    try:
        from stable_baselines3 import PPO, SAC  # noqa: F401
    except Exception as e:
        raise SystemExit("Please install stable-baselines3 to evaluate RL baselines: pip install stable-baselines3")

    wrapped = EV2GymSB3Wrapper(env)
    if args.algo == "ppo":
        from stable_baselines3 import PPO as Algo
    else:
        from stable_baselines3 import SAC as Algo

    model_path = Path(args.model)
    model = Algo.load(str(model_path), env=wrapped)

    obs, info = wrapped.reset()
    for t in range(env.simulation_length):
        action, _ = model.predict(obs, deterministic=args.det)
        obs, reward, done, truncated, info = wrapped.step(action)
        # Ask the core env to plot/record if it supports it
        if done or truncated:
            break

    print("Evaluation finished at step", env.current_step)


if __name__ == "__main__":
    main()

