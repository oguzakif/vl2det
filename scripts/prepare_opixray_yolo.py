#!/usr/bin/env python3
"""Convert OPIXray raw dataset to YOLO format."""

from __future__ import annotations

import argparse
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.opixray_dataset import (  # noqa: E402
    OPIXRAY_CLASSES,
    extract_opixray_zip,
    prepare_opixray_yolo_dataset,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OPIXray raw dataset to YOLO format.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/kd_datasets/OPIXray_raw"),
        help="Path to extracted OPIXray root (contains train/ and test/).",
    )
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=None,
        help="Optional OPIXray zip file to extract before conversion.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/kd_datasets/OPIXray"),
        help="Output YOLO dataset root.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--link-mode",
        choices=("auto", "symlink", "hardlink", "copy"),
        default="auto",
        help="How to place image files in the output dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory (and extracted raw dir if --zip is used).",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Optional class names list (defaults to OPIXray classes).",
    )
    parser.add_argument(
        "--discover-classes",
        action="store_true",
        help="Discover class names from annotations instead of using defaults.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data_root = args.data_root
    if args.zip_path is not None:
        data_root = extract_opixray_zip(args.zip_path, data_root, overwrite=args.overwrite)

    if args.discover_classes:
        class_names = None
    elif args.class_names:
        class_names = list(args.class_names)
    else:
        class_names = list(OPIXRAY_CLASSES)

    prepare_opixray_yolo_dataset(
        data_root=data_root,
        output_root=args.output_root,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
        class_names=class_names,
    )
    print(f"Wrote YOLO dataset to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
