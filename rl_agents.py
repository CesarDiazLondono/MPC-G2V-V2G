"""
RL baselines (PPO, SAC) integration for EV2Gym via Stable-Baselines3.

This module provides:
- EV2GymSB3Wrapper: a light Gym-compatible wrapper around EV2Gym.
- PPOAgent, SACAgent: thin helpers to train and act with SB3 models.

Notes
- Requires stable-baselines3 and gym/gymnasium installed in your environment.
- Actions are continuous Box[-1, 1] per EVSE port, matching EV2Gym usage in
  this repo (same as MPC agents’ normalized actions).
"""

from __future__ import annotations

import numpy as np
from typing import Any, Tuple, Optional

# Try gymnasium first, fallback to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    import gym  # type: ignore
    from gym import spaces  # type: ignore


class EV2GymSB3Wrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, core_env):
        super().__init__()
        self.core_env = core_env

        # Derive observation shape from a real reset
        obs, info = self._safe_reset_core()
        self._obs_shape = np.array(obs).shape
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._obs_shape, dtype=np.float32
        )

        # Actions: one per port, normalized [-1, 1]
        n_ports = int(getattr(core_env, "number_of_ports"))
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_ports,), dtype=np.float32
        )

    # --- Core-to-gym adapters ---
    def _safe_reset_core(self) -> Tuple[np.ndarray, dict]:
        out = self.core_env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None and hasattr(self.core_env, "seed"):
            try:
                self.core_env.seed(seed)
            except Exception:
                pass
        obs, info = self._safe_reset_core()
        return obs, info

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # EV2Gym returns (obs, reward, done, [truncated], info/stats)
        out = self.core_env.step(a, visualize=False)
        if isinstance(out, tuple):
            if len(out) == 5:
                obs, reward, done, truncated, info = out
            elif len(out) == 4:
                obs, reward, done, info = out
                truncated = False
            else:
                # fallback best-effort
                obs, reward, done = out[0], out[1], out[2]
                truncated = False
                info = out[-1] if len(out) > 3 else {}
        else:
            raise RuntimeError("Unexpected env.step return signature")

        obs = np.asarray(obs, dtype=np.float32)
        return obs, float(reward), bool(done), bool(truncated), dict(info)


class _SB3Base:
    def __init__(self, env, algo_name: str, policy: str = "MlpPolicy", verbose: int = 0):
        self.env = env
        self.algo_name = algo_name
        self.policy = policy
        self.verbose = verbose
        self.model = None

    def _import_algo(self):
        try:
            from stable_baselines3 import PPO, SAC  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "stable-baselines3 not installed. Please `pip install stable-baselines3`"
            ) from e

    def train(self, timesteps: int = 10000, save_path: Optional[str] = None, algo_kwargs: Optional[dict] = None):
        self._import_algo()
        from stable_baselines3 import PPO, SAC

        wrapped = EV2GymSB3Wrapper(self.env)
        Algo = PPO if self.algo_name.lower() == "ppo" else SAC
        self.model = Algo(self.policy, wrapped, verbose=self.verbose, **(algo_kwargs or {}))
        self.model.learn(total_timesteps=int(timesteps))
        if save_path:
            self.model.save(save_path)
        return self.model

    def load(self, path: str):
        self._import_algo()
        from stable_baselines3 import PPO, SAC
        Algo = PPO if self.algo_name.lower() == "ppo" else SAC
        wrapped = EV2GymSB3Wrapper(self.env)
        self.model = Algo.load(path, env=wrapped)
        return self.model

    def get_action(self, env):
        if self.model is None:
            # conservative default
            return np.zeros(env.number_of_ports, dtype=np.float32)
        # try to read current obs via a lightweight wrapper without resetting the env
        # assumption: main loop keeps the latest observation in `env.state`; if not, we fallback to zeros
        try:
            obs = np.asarray(getattr(env, "state", None), dtype=np.float32)
            if obs is None or obs.size == 0:
                raise AttributeError
        except Exception:
            # cannot retrieve current obs; return neutral action
            return np.zeros(env.number_of_ports, dtype=np.float32)
        act, _ = self.model.predict(obs, deterministic=True)
        act = np.clip(np.asarray(act, dtype=np.float32), -1.0, 1.0)
        return act


class PPOAgent(_SB3Base):
    def __init__(self, env, verbose: int = 0):
        super().__init__(env, algo_name="ppo", verbose=verbose)


class SACAgent(_SB3Base):
    def __init__(self, env, verbose: int = 0):
        super().__init__(env, algo_name="sac", verbose=verbose)

