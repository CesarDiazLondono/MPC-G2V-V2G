# scripts/fig4a_from_csv.py
# 复现论文 Fig.4(a)：Charging / Discharging prices（3个并排相同子图）
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 你本地 EV2Gym 的日前电价文件路径（仓库里就有） ---
CSV_PATH = r"E:\code\EV2gym\ev2gym\data\Netherlands_day-ahead-2015-2024.csv"
# 说明：该 CSV 通常是“€/MWh、逐小时”共 24 个点。论文图按 15min 采样，需要插值到 96 点。

def load_dayahead_prices(csv_path, year=2022, month=1, day=17):
    """从 CSV 取某一天的 24 点日前电价（€/MWh），并返回 €/kWh 的 96 点(15min)插值结果"""
    df = pd.read_csv(csv_path)
    # 尝试适配常见列名（不同版本可能是 'date'+'price' 或 'datetime'+'price_eur_mwh'）
    col_dt = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            col_dt = c
            break
    if col_dt is None:
        raise ValueError("未在 CSV 里找到日期列（包含 'date' or 'time' 的列名）。")

    # 解析日期列
    try:
        df[col_dt] = pd.to_datetime(df[col_dt], format="mixed", dayfirst=True, errors="coerce")
    except TypeError:
        df[col_dt] = pd.to_datetime(df[col_dt], dayfirst=True, errors="coerce")
    if df[col_dt].isna().any():
        df = df.dropna(subset=[col_dt])
    # 取当天 00:00-23:00 的 24 个小时值
    date0 = pd.Timestamp(year=year, month=month, day=day)
    df_day = df[(df[col_dt] >= date0) & (df[col_dt] < date0 + pd.Timedelta(days=1))].sort_values(col_dt)

    # 尝试找到价格列
    price_col = None
    for c in df_day.columns:
        name = c.lower()
        if ("price" in name or "eur" in name) and ("mwh" in name or "euro" in name):
            price_col = c
            break
    if price_col is None:
        # 退一步：找名叫 'price' 的列
        price_col = "price"
        if price_col not in df_day.columns:
            raise ValueError("未在 CSV 里找到价格列（含 'price' 与 'MWh' 语义），请看一下文件列名。")

    # 24 个小时值（€/MWh）
    p_mwh_hourly = df_day[price_col].to_numpy().astype(float)
    if len(p_mwh_hourly) != 24:
        raise ValueError(f"当天数据不是 24 个小时值，实际={len(p_mwh_hourly)}。")

    # €/MWh → €/kWh
    p_kwh_hourly = p_mwh_hourly / 1000.0

    # 线性插值到 96 点（15 min）
    p_kwh_quarter = np.repeat(p_kwh_hourly, 4)

    return p_kwh_quarter  # 长度 96，单位 €/kWh

def plot_fig4a(prices_charge, m=1.1, out_path="analysis_out/fig4a_replica.png", titles=False):
    """画三联图（曲线相同，仅为排版需要）。prices_charge：96点 €/kWh"""
    # 规范化与创建输出目录（处理空目录或多余空白/分隔符）
    out_path = os.path.normpath(str(out_path).strip())
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    prices_discharge = m * prices_charge

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    names = ["eMPC", "OCMF", "AFAP/ALAP"] if titles else ["", "", ""]
    x = np.arange(96)
    # 横坐标标注为 0:00, 4:00, 8:00, ..., 24:00
    xticks = np.arange(0, 97, 16)
    xlabels = [f"{(i//4)%24}:00" for i in xticks]

    for ax, name in zip(axes, names):
        ax.step(x, prices_charge, where="post", label="Charging", linewidth=2, color="#1f77b4")
        ax.step(x, prices_discharge, where="post", label="Discharging", linewidth=2, color="#b22222")
        if name:
            ax.set_title(name)
        ax.set_xticks(xticks, xlabels)
        ax.set_xlim(0, 96)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Price (€/kWh)")
    axes[0].set_ylabel("Price (€/kWh)")
    axes[1].set_xlabel("Time")
    axes[0].legend(loc="lower left")
    axes[0].set_ylabel("Price (EUR/kWh)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print("Saved:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH, help="Day-ahead CSV 路径（€/MWh, hourly）")
    ap.add_argument("--date", default="2022-01-17", help="选择日期 YYYY-MM-DD（与你论文用的一致）")
    ap.add_argument("--m", type=float, default=1.1, help="放电价格系数 m")
    ap.add_argument("--out", default="analysis_out/fig4a_replica.png")
    ap.add_argument("--titles", action="store_true", help="是否在子图上加 eMPC/OCMF/AFAP 标题")
    args = ap.parse_args()

    Y, M, D = map(int, args.date.split("-"))
    p_charge = load_dayahead_prices(args.csv, Y, M, D)
    plot_fig4a(p_charge, m=args.m, out_path=args.out, titles=args.titles)
