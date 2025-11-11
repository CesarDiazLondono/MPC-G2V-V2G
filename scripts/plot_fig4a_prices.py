# scripts/plot_fig4a_prices.py
import os, argparse, yaml, numpy as np, matplotlib.pyplot as plt

def get_synthetic_prices(step_minutes=15):
    """构造一个近似论文图4(a)的电价曲线"""
    T = int(24 * 60 / step_minutes)
    t = np.arange(T)
    # 模拟：凌晨低价、中午高峰、晚上次高峰
    base = 0.15 + 0.1 * np.sin(2 * np.pi * (t - 20) / T) + 0.05 * np.sin(4 * np.pi * (t - 20) / T)
    base = np.clip(base, 0.05, 0.35)
    return base

def main(cfg_path: str, m_override: float|None, row3: bool, out_path: str|None):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    step  = int(cfg.get("timescale", 15))
    m = float(cfg.get("discharge_price_factor", 1.1) if m_override is None else m_override)

    charge = get_synthetic_prices(step)
    discharge = m * charge

    T = len(charge); x = np.arange(T)
    ticks = np.arange(0, T+1, 16)
    labels = [f"{(i//4)%24}:00" if i < 96 else "0:00" for i in ticks]

    os.makedirs("analysis_out", exist_ok=True)
    out = out_path or ("analysis_out/fig4a_row.png" if row3 else "analysis_out/fig4a_prices.png")

    if row3:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        titles = ["eMPC", "OCMF", "AFAP/ALAP"]
        for j, a in enumerate(ax):
            a.plot(x, charge, label="Charging", linewidth=2)
            a.plot(x, discharge, label="Discharging", linewidth=2)
            a.set_title(titles[j])
            a.set_xticks(ticks, labels); a.set_xlim(0, T-1); a.grid(True, alpha=.3)
        ax[0].set_ylabel("Price (€/kWh)"); ax[0].legend(loc="upper right")
        plt.tight_layout(); plt.savefig(out, dpi=200)
    else:
        plt.figure(figsize=(6.0, 4.0), dpi=200)
        plt.plot(x, charge, label="Charging", linewidth=2)
        plt.plot(x, discharge, label="Discharging", linewidth=2)
        plt.ylabel("Price (€/kWh)"); plt.xlabel("Time")
        plt.xticks(ticks, labels); plt.xlim(0, T-1)
        plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout()
        plt.savefig(out)
    print("Saved:", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/fig4_case.yaml")
    ap.add_argument("--m", type=float, default=None)
    ap.add_argument("--row3", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(args.cfg, args.m, args.row3, args.out)
