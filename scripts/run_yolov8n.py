#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8n training/eval for comparison.")
    parser.add_argument(
        "--data",
        default="data/kd_datasets/OPIXray/data.yaml",
        help="Path to a YOLO data.yaml file.",
    )
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 weights or model name.")
    parser.add_argument("--imgsz", type=int, default=512, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--device", default="0", help="Device id (e.g., 0) or 'cpu'.")
    parser.add_argument(
        "--mode",
        choices=("train", "val", "test"),
        default="val",
        help="Run mode: train/val/test.",
    )
    parser.add_argument("--project", default="logs/yolov8", help="Ultralytics project dir.")
    parser.add_argument("--name", default="yolov8n_opixray", help="Run name.")
    parser.add_argument("--patience", type=int, default=50, help="Early stop patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow existing run dir.")
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    return parser.parse_args()


def _results_to_dict(results):
    if isinstance(results, dict):
        return results
    if hasattr(results, "results_dict"):
        return results.results_dict
    if hasattr(results, "metrics") and hasattr(results.metrics, "results_dict"):
        return results.metrics.results_dict
    return None


def main() -> int:
    args = _parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Data yaml not found: {data_path}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from exc

    if args.weights:
        model = YOLO(args.weights)
    else:
        model = YOLO("yolov8n.pt")  # default to yolov8n

    common = dict(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
    )

    if args.mode == "train":
        results = model.train(epochs=args.epochs, patience=args.patience, **common)
    elif args.mode == "val":
        results = model.val(**common)
    else:
        results = model.val(split="test", **common)

    metrics = _results_to_dict(results)
    if metrics:
        print(json.dumps(metrics, indent=2, sort_keys=True))
        if args.save_json:
            Path(args.save_json).write_text(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print("No metrics dict found on the results object.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
