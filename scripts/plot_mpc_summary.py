"""
Quick plots for analysis_out/mpc_summary.csv produced by run_mpc_experiments.py

Usage
  python scripts/plot_mpc_summary.py --csv analysis_out/mpc_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="analysis_out/mpc_summary.csv")
    p.add_argument("--outdir", default="analysis_out")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if df.empty:
        print("Empty CSV; nothing to plot")
        return

    # 1) Profits: mean ± std per algo
    agg = df.groupby("algo").agg(profits_mean=("profits", "mean"), profits_std=("profits", "std"))
    plt.figure(figsize=(4.6, 2.8), dpi=170)
    ax = agg["profits_mean"].plot(kind="bar", yerr=agg["profits_std"], capsize=3, color=["#e377c2", "#1f77b4"])  # type: ignore
    plt.ylabel("Profits (mean ± std)")
    plt.title("MPC Profits by Algorithm")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fout = outdir / "mpc_summary_profits_bar.png"
    plt.savefig(fout); plt.close(); print(f"Saved: {fout.as_posix()}")

    # 2) Profits: boxplot by algo (per-seed distribution)
    plt.figure(figsize=(4.6, 2.8), dpi=170)
    df.boxplot(column="profits", by="algo")
    plt.suptitle("")
    plt.title("Profits distribution by algo")
    plt.ylabel("Profits")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fout = outdir / "mpc_summary_profits_box.png"
    plt.savefig(fout); plt.close(); print(f"Saved: {fout.as_posix()}")

    # 3) Satisfaction: mean ± std per algo
    agg2 = df.groupby("algo").agg(sat_mean=("satisfaction", "mean"), sat_std=("satisfaction", "std"))
    plt.figure(figsize=(4.6, 2.8), dpi=170)
    ax = agg2["sat_mean"].plot(kind="bar", yerr=agg2["sat_std"], capsize=3, color=["#e377c2", "#1f77b4"])  # type: ignore
    plt.ylabel("User satisfaction (mean ± std)")
    plt.title("MPC Satisfaction by Algorithm")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fout = outdir / "mpc_summary_satisfaction_bar.png"
    plt.savefig(fout); plt.close(); print(f"Saved: {fout.as_posix()}")


if __name__ == "__main__":
    main()
