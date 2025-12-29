#!/usr/bin/env python3
"""Plot TensorBoard scalar tags from one or more runs.

Example:
  python scripts/plot_tensorboard_scalars.py \
    logs/train/runs/2025-12-20_02-52-48_ssdlite320_mobilenet_v3_large_vlm_distill_noaug \
    logs/train/runs/2025-12-21_10-11-00_ssdlite320_mobilenet_v3_large_vlm_distill_aug \
    --legends no-aug aug --out-dir plots/tensorboard
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except Exception as exc:  # pragma: no cover - import guard for missing deps
    raise SystemExit(
        "Missing dependency: tensorboard. Install with `pip install tensorboard`."
    ) from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - import guard for missing deps
    raise SystemExit(
        "Missing dependency: matplotlib. Install with `pip install matplotlib`."
    ) from exc


@dataclass
class RunScalars:
    label: str
    log_dir: Path
    event_dir: Path
    scalars: dict[str, tuple[list[int], list[float]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TensorBoard scalar tags from multiple training folders."
    )
    parser.add_argument(
        "log_dirs",
        nargs="+",
        help="Run directories to search for TensorBoard event files.",
    )
    parser.add_argument(
        "--legends",
        nargs="*",
        default=None,
        help="Legend labels, same order as log_dirs. Defaults to folder names.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Scalar tags to plot. Defaults to all scalar tags found.",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/tensorboard",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--format",
        default="svg",
        choices=("svg", "png", "pdf"),
        help="Output file format.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Moving average window size (0 disables smoothing).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="DPI used for raster outputs like PNG.",
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="Print available scalar tags and exit.",
    )
    return parser.parse_args()


def resolve_event_dir(log_dir: Path) -> Path:
    if log_dir.is_file() and log_dir.name.startswith("events.out.tfevents."):
        return log_dir.parent

    event_files = sorted(
        log_dir.rglob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files found under: {log_dir}"
        )

    chosen_dir = event_files[0].parent
    unique_dirs = {path.parent for path in event_files}
    if len(unique_dirs) > 1:
        print(
            f"Note: multiple event directories under {log_dir}, "
            f"using {chosen_dir}",
            file=sys.stderr,
        )
    return chosen_dir


def load_scalars(event_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
    accumulator = EventAccumulator(
        str(event_dir),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    scalars: dict[str, tuple[list[int], list[float]]] = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        steps = [event.step for event in events]
        values = [float(event.value) for event in events]
        scalars[tag] = (steps, values)
    return scalars


def build_runs(log_dirs: Iterable[str], legends: list[str] | None) -> list[RunScalars]:
    runs: list[RunScalars] = []
    for idx, log_dir in enumerate(log_dirs):
        path = Path(log_dir)
        label = legends[idx] if legends else path.name
        event_dir = resolve_event_dir(path)
        scalars = load_scalars(event_dir)
        runs.append(
            RunScalars(
                label=label,
                log_dir=path,
                event_dir=event_dir,
                scalars=scalars,
            )
        )
    return runs


def collect_tags(runs: list[RunScalars]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for run in runs:
        for tag in run.scalars.keys():
            if tag not in seen:
                seen.add(tag)
                ordered.append(tag)
    return ordered


def smooth_values(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed: list[float] = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        window_size = min(idx + 1, window)
        smoothed.append(running_sum / window_size)
    return smoothed


def sanitize_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_") or "scalar"


def plot_tag(
    tag: str,
    runs: list[RunScalars],
    out_dir: Path,
    fmt: str,
    smooth: int,
    dpi: int,
) -> None:
    plt.figure()
    plotted = False
    for run in runs:
        if tag not in run.scalars:
            continue
        steps, values = run.scalars[tag]
        if smooth > 1:
            values = smooth_values(values, smooth)
        plt.plot(steps, values, label=run.label)
        plotted = True
    if not plotted:
        return
    plt.title(tag)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sanitize_tag(tag)}.{fmt}"
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    log_dirs = [Path(path) for path in args.log_dirs]
    if args.legends and len(args.legends) != len(log_dirs):
        raise SystemExit("Number of legends must match number of log_dirs.")

    runs = build_runs(args.log_dirs, args.legends)
    available_tags = collect_tags(runs)
    if args.list_tags:
        for tag in available_tags:
            print(tag)
        return

    tags = args.tags if args.tags else available_tags
    if not tags:
        raise SystemExit("No scalar tags found to plot.")

    out_dir = Path(args.out_dir)
    for tag in tags:
        plot_tag(
            tag=tag,
            runs=runs,
            out_dir=out_dir,
            fmt=args.format,
            smooth=args.smooth,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
