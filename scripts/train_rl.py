"""
Train PPO or SAC on EV2Gym using Stable-Baselines3.

Usage (bash):
  python scripts/train_rl.py --algo ppo --config paper_baseline.yaml \
         --timesteps 200000 --save results/ppo_ev2gym

Requires: stable-baselines3, gym/gymnasium installed.
"""

import argparse
from pathlib import Path

from ev2gym.models.ev2gym_env import EV2Gym
from rl_agents import EV2GymSB3Wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--config", default="paper_baseline.yaml")
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--save", default="results/sb3_model")
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    env = EV2Gym(config_file=str(cfg_path), save_replay=False, save_plots=False, verbose=False)

    # Import lazily to give clear error if missing
    try:
        from stable_baselines3 import PPO, SAC  # noqa: F401
    except Exception as e:
        raise SystemExit("Please install stable-baselines3 to train RL baselines: pip install stable-baselines3")

    wrapped = EV2GymSB3Wrapper(env)
    if args.algo == "ppo":
        from stable_baselines3 import PPO as Algo
    else:
        from stable_baselines3 import SAC as Algo

    model = Algo("MlpPolicy", wrapped, verbose=args.verbose)
    model.learn(total_timesteps=int(args.timesteps))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print("Saved:", out)


if __name__ == "__main__":
    main()

