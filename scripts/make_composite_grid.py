"""
Make a 3x3 composite figure from three EV2Gym runs.

Each run directory must contain:
 - EV_Energy_Level.png
 - CS_Current_signals.png
 - Total_Aggregated_Power.png

Usage
  python scripts/make_composite_grid.py \
      --run A=results/sim_2025_10_30_111111 \
      --run B=results/sim_2025_10_30_222222 \
      --run C=results/sim_2025_10_30_333333 \
      --labels "AFAP, PPO, eMPC_V2G" \
      --out analysis_out/composite.png

If no --run is provided, the script picks the latest three subfolders of
`results/` as A, B, C.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from PIL import Image


REQUIRED = [
    "EV_Energy_Level.png",
    "CS_Current_signals.png",
    "Total_Aggregated_Power.png",
]


def dir_ok(d: Path) -> bool:
    return d.is_dir() and all((d / f).exists() for f in REQUIRED)


def pick_latest_three(results_dir: Path) -> List[Path]:
    sub = [p for p in results_dir.iterdir() if dir_ok(p)]
    sub.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return sub[:3]


def parse_runs(args_runs: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in args_runs:
        # form: A=path
        if "=" not in spec:
            raise ValueError(f"Invalid --run spec '{spec}', expected NAME=PATH")
        name, p = spec.split("=", 1)
        out[name.strip()] = Path(p).resolve()
    return out


def load_grid_cells(run_dir: Path) -> List[Image.Image]:
    return [Image.open(run_dir / f).convert("RGB") for f in REQUIRED]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="analysis_out/composite.png")
    parser.add_argument("--results", type=str, default="results", help="Root results dir")
    parser.add_argument("--run", action="append", default=[], help="Named run as NAME=DIR")
    parser.add_argument("--labels", type=str, default="A,B,C", help="Comma labels for columns")
    args = parser.parse_args()

    res_dir = Path(args.results)
    res_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.run:
        runs = parse_runs(args.run)
        # keep insertion order A,B,C if provided like that
        cols = list(runs.items())
    else:
        picked = pick_latest_three(res_dir)
        keys = ["A", "B", "C"]
        cols = list(zip(keys, picked))

    labels = [s.strip() for s in args.labels.split(",")]
    while len(labels) < 3:
        labels.append("")
    labels = labels[:3]

    # Load images: each column is a run, each row is one of REQUIRED in order
    cells: List[List[Image.Image]] = []
    widths: List[int] = []
    heights: List[int] = []
    for _, run_dir in cols:
        imgs = load_grid_cells(run_dir)
        cells.append(imgs)
        widths.append(max(i.width for i in imgs))
        heights.append(sum(i.height for i in imgs))

    col_w = max(widths) if widths else 0
    row_hs = [max(cells[c][r].height for c in range(len(cells))) for r in range(3)]
    total_w = col_w * len(cells)
    total_h = sum(row_hs)

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    y = 0
    for r in range(3):
        x = 0
        for c in range(len(cells)):
            img = cells[c][r]
            if img.width != col_w:
                img = img.resize((col_w, int(img.height * col_w / img.width)), Image.BILINEAR)
            canvas.paste(img, (x, y))
            x += col_w
        y += row_hs[r]

    # Simple header labels (top-left corners of each column)
    try:
        from PIL import ImageDraw, ImageFont  # lazy import
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        for c, lab in enumerate(labels):
            draw.text((10 + c * col_w, 10), lab, fill=(0, 0, 0), font=font)
    except Exception:
        pass

    canvas.save(out_path)
    print(f"Saved composite to: {out_path}")


if __name__ == "__main__":
    main()

