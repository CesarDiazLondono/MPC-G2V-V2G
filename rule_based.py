"""
Simple rule-based baselines: AFAP and ALAP.

AFAP (As Fast As Possible):
- Whenever an EV is connected and below its desired energy at departure,
  charge at maximum power; otherwise do nothing.

ALAP (As Late As Possible):
- Delay charging and only start when it becomes necessary to finish
  before departure at maximum power.

Both return actions normalized to [-1, 1] per EVSE port, compatible with
the env.step(actions) used by existing MPC agents in this repo.
"""

from __future__ import annotations

from typing import List
import numpy as np


def _flatten_ports(env) -> List:
    """Yield ports in the same order as MPC uses (per CS, then per port)."""
    for cs in env.charging_stations:
        for port_idx in range(cs.n_ports):
            yield cs, port_idx


def _get_ev_connected(cs, port_idx):
    # evs_connected aligns with ports order for a CS
    if hasattr(cs, "evs_connected"):
        if 0 <= port_idx < len(cs.evs_connected):
            return cs.evs_connected[port_idx]
    return None


def _ev_max_power(cs) -> float:
    # Use station’s port power limit (kW)
    return float(getattr(cs, "get_max_power")())


def _delta_hours(env) -> float:
    # timescale is in minutes
    return float(env.timescale) / 60.0


def _desired_energy_kwh(ev) -> float:
    # Accept both fractional desired_capacity and absolute target in kWh
    cap_kwh = float(getattr(ev, "max_battery_capacity", getattr(ev, "battery_capacity", 0)))
    desired = getattr(ev, "desired_capacity", None)
    if desired is None:
        # Fallback: aim for 80% if unspecified
        return 0.8 * cap_kwh
    try:
        desired_val = float(desired)
    except Exception:
        return 0.8 * cap_kwh
    if cap_kwh > 0 and desired_val <= 1.5:
        return desired_val * cap_kwh
    return desired_val


class AFAP:
    def __init__(self, env, verbose: bool = False):
        self.env = env
        self.verbose = verbose

    def get_action(self, env):
        actions = np.zeros(env.number_of_ports, dtype=float)
        idx = 0
        dt_h = _delta_hours(env)
        for cs, port_idx in _flatten_ports(env):
            ev = _get_ev_connected(cs, port_idx)
            if ev is None:
                actions[idx] = 0.0
            else:
                target = _desired_energy_kwh(ev)
                curr = float(getattr(ev, "current_capacity", 0.0))
                if curr + 1e-6 < target:
                    actions[idx] = 1.0  # full charging rate (normalized)
                else:
                    actions[idx] = 0.0
            idx += 1
        if self.verbose:
            print("AFAP actions:", actions)
        return actions


class ALAP:
    def __init__(self, env, verbose: bool = False):
        self.env = env
        self.verbose = verbose

    def get_action(self, env):
        actions = np.zeros(env.number_of_ports, dtype=float)
        idx = 0
        dt_h = _delta_hours(env)
        t_now = env.current_step
        for cs, port_idx in _flatten_ports(env):
            ev = _get_ev_connected(cs, port_idx)
            if ev is None:
                actions[idx] = 0.0
                idx += 1
                continue

            target = _desired_energy_kwh(ev)
            curr = float(getattr(ev, "current_capacity", 0.0))
            dep = int(getattr(ev, "time_of_departure", t_now))
            rem_steps = max(0, dep - t_now)
            if rem_steps <= 0:
                actions[idx] = 0.0
                idx += 1
                continue

            Pmax = _ev_max_power(cs)  # kW
            # simplest: ignore efficiency here (often 1.0 in baseline)
            max_deliverable = Pmax * dt_h * rem_steps
            needed = max(0.0, target - curr)

            # If waiting one more step makes it impossible to reach target, start now
            if needed > Pmax * dt_h * (rem_steps - 1) + 1e-9:
                actions[idx] = 1.0
            else:
                actions[idx] = 0.0
            idx += 1

        if self.verbose:
            print("ALAP actions:", actions)
        return actions

