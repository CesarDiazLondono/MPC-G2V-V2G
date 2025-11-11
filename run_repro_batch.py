# run_repro_batch.py
import os, yaml, copy, csv
from ev2gym.models.ev2gym_env import EV2Gym
from eMPC import eMPC_V2G

BASE_CFG = "paper_baseline.yaml"
OUT_CSV = "paper_summary.csv"

M_VALUES = [0.8, 0.9, 1.0, 1.1, 1.2]
H_VALUES = [10, 40]     # 2.5h 与 10h
REPEATS = 10            # 论文里常用 50，这里先 10 跑通再加大

def run_once(cfg, H, seed, m):
    cfg2 = copy.deepcopy(cfg)
    cfg2["discharge_price_factor"] = m
    tmp_cfg = "_tmp_run.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg2, f, sort_keys=False, allow_unicode=True)

    env = EV2Gym(config_file=tmp_cfg, verbose=False, save_replay=False, save_plots=False)
    state, _ = env.reset(seed=seed)

    agent = eMPC_V2G(env, control_horizon=H, verbose=False)
    stats = None
    for _ in range(env.simulation_length):
        a = agent.get_action(env)
        _, _, done, _, stats = env.step(a, visualize=False)
        if done:
            break

    os.remove(tmp_cfg)
    # 把常用指标取出来
    return {
        "m": m, "H": H, "seed": seed,
        "profits": float(stats.get("total_profits", 0.0)),
        "n_served": int(stats.get("total_ev_served", 0)),
        "energy_c": float(stats.get("total_energy_charged", 0.0)),
        "energy_d": float(stats.get("total_energy_discharged", 0.0)),
        "satisfaction": float(stats.get("average_user_satisfaction", 0.0)),
        "degradation": float(stats.get("battery_degradation", 0.0)),
        "track_err": float(stats.get("tracking_error", 0.0)),
    }

if __name__ == "__main__":
    with open(BASE_CFG, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    rows = []
    for H in H_VALUES:
        for m in M_VALUES:
            for seed in range(REPEATS):
                r = run_once(base, H=H, seed=seed, m=m)
                print(r)
                rows.append(r)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"Saved: {OUT_CSV}")
