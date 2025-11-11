# summarize_results.py
import os
import pandas as pd
import matplotlib.pyplot as plt

CSV = "paper_summary.csv"
OUT = "analysis_out"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
print(f"Total rows: {len(df)}")
print("Columns:", list(df.columns))

# 统计汇总（均值/标准差/样本数）
agg = (
    df.groupby(["H", "m"])
      .agg(
          profits_mean=("profits", "mean"),
          profits_std=("profits", "std"),
          satisfaction_mean=("satisfaction", "mean"),
          degradation_mean=("degradation", "mean"),
          served_mean=("n_served", "mean"),
          n_runs=("seed", "count"),
      )
      .reset_index()
      .sort_values(["H", "m"])
)

agg.to_csv(os.path.join(OUT, "paper_summary_agg.csv"), index=False)
print(f"Wrote {os.path.join(OUT, 'paper_summary_agg.csv')}")

# --------- 画图（仅用 matplotlib，默认配色）---------
def line_by_m(col, ylabel, filename):
    for H in sorted(df["H"].unique()):
        sub = df[df["H"] == H].groupby("m")[col].mean()
        plt.figure()
        sub.plot(marker="o")
        plt.title(f"{col} vs m (H={H})")
        plt.xlabel("m (discharge price factor)")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"{filename}_H{H}.png"), dpi=150)
        plt.close()

line_by_m("profits", "Average profits", "profits_vs_m")
line_by_m("satisfaction", "Average user satisfaction", "satisfaction_vs_m")
line_by_m("degradation", "Average battery degradation", "degradation_vs_m")

# H 对比（按 m 展开两条线）
for col, ylabel, fname in [
    ("profits", "Average profits", "profits_H_compare"),
    ("satisfaction", "Average user satisfaction", "satisfaction_H_compare"),
    ("degradation", "Average battery degradation", "degradation_H_compare"),
]:
    plt.figure()
    for H in sorted(df["H"].unique()):
        series = df[df["H"] == H].groupby("m")[col].mean().sort_index()
        plt.plot(series.index, series.values, marker="o", label=f"H={H}")
    plt.xlabel("m (discharge price factor)")
    plt.ylabel(ylabel)
    plt.title(f"{col} vs m (H comparison)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"{fname}.png"), dpi=150)
    plt.close()

print(f"Charts saved in: {OUT}")
