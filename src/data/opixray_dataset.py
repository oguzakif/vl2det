from __future__ import annotations

import argparse
import os
import random
import shutil
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

OPIXRAY_SUBSETS = ("train", "test")
OPIXRAY_CLASSES = [
    "Folding_Knife",
    "Multi-tool_Knife",
    "Scissor",
    "Straight_Knife",
    "Utility_Knife",
]


@dataclass(frozen=True)
class OPIXrayBox:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int


def parse_opixray_subset(root: Path, subset: str) -> Dict[str, List[OPIXrayBox]]:
    if subset not in OPIXRAY_SUBSETS:
        raise ValueError(f"subset must be one of {OPIXRAY_SUBSETS}, got: {subset!r}")

    annotation_dir = root / subset / f"{subset}_annotation"
    image_dir = root / subset / f"{subset}_image"
    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_to_boxes: Dict[str, List[OPIXrayBox]] = {}
    for ann_path in sorted(annotation_dir.glob("*.txt")):
        for line in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            image_name = parts[0]
            label = parts[1]
            try:
                x1, y1, x2, y2 = map(int, parts[2:6])
            except ValueError as exc:
                raise ValueError(f"Invalid bbox in {ann_path}: {line}") from exc

            image_path = image_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image referenced in {ann_path} not found: {image_path}"
                )

            image_to_boxes.setdefault(image_name, []).append(
                OPIXrayBox(label=label, x1=x1, y1=y1, x2=x2, y2=y2)
            )

    if not image_to_boxes:
        raise RuntimeError(f"No samples collected from {annotation_dir}")
    return image_to_boxes


def extract_opixray_zip(
    zip_path: Union[str, Path], output_root: Union[str, Path], overwrite: bool = False
) -> Path:
    zip_path = Path(zip_path)
    output_root = Path(output_root)
    if output_root.exists():
        if not overwrite:
            return output_root
        _safe_rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_root)
    return output_root


def _safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path(resolved.root):
        raise ValueError(f"Refusing to delete filesystem root: {resolved}")
    shutil.rmtree(resolved)


def _link_or_copy_file(src: Path, dst: Path, mode: str) -> None:
    src_resolved = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if mode in {"symlink", "auto"}:
        try:
            dst.symlink_to(src_resolved)
            return
        except OSError:
            if mode == "symlink":
                raise

    if mode in {"hardlink", "auto"}:
        try:
            os.link(src_resolved, dst)
            return
        except OSError:
            if mode == "hardlink":
                raise

    shutil.copy2(src_resolved, dst)


def _image_size(path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        return _image_size_from_header(path)

    with Image.open(path) as img:
        width, height = img.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size for {path}: {width}x{height}")
    return width, height


def _image_size_from_header(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(32)
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            if len(head) < 24:
                f.seek(0)
                head = f.read(24)
            width, height = struct.unpack(">II", head[16:24])
            return int(width), int(height)

        if head[:2] == b"\xff\xd8":
            f.seek(0)
            return _jpeg_size(f)

        if head[:2] == b"BM":
            f.seek(18)
            width = struct.unpack("<I", f.read(4))[0]
            height = struct.unpack("<I", f.read(4))[0]
            return int(width), int(height)

    raise ValueError(f"Unsupported image format: {path}")


def _jpeg_size(fh) -> Tuple[int, int]:
    if fh.read(2) != b"\xff\xd8":
        raise ValueError("Invalid JPEG header")

    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }

    while True:
        marker_prefix = fh.read(1)
        if not marker_prefix:
            break
        if marker_prefix != b"\xff":
            continue
        marker = fh.read(1)
        if not marker:
            break
        while marker == b"\xff":
            marker = fh.read(1)
        marker_type = marker[0]
        if marker_type in sof_markers:
            _ = fh.read(2)
            _ = fh.read(1)
            height, width = struct.unpack(">HH", fh.read(4))
            return int(width), int(height)
        segment_length = struct.unpack(">H", fh.read(2))[0]
        fh.seek(segment_length - 2, os.SEEK_CUR)

    raise ValueError("JPEG size not found")


def _clamp_bbox(width: int, height: int, bbox: Sequence[int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(int, bbox[:4])
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def _to_yolo_xywh(
    width: int, height: int, x1: int, y1: int, x2: int, y2: int
) -> Tuple[float, float, float, float]:
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    xc = float(x1) + bw / 2.0
    yc = float(y1) + bh / 2.0
    return xc / float(width), yc / float(height), bw / float(width), bh / float(height)


def _primary_label(boxes: Sequence[OPIXrayBox]) -> str:
    counts: Dict[str, int] = {}
    for box in boxes:
        counts[box.label] = counts.get(box.label, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def split_train_val(
    image_to_boxes: Mapping[str, Sequence[OPIXrayBox]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")

    grouped: Dict[str, List[str]] = {}
    for image_name, boxes in image_to_boxes.items():
        grouped.setdefault(_primary_label(boxes), []).append(image_name)

    rng = random.Random(seed)
    train_split: List[str] = []
    val_split: List[str] = []

    for _, items in sorted(grouped.items(), key=lambda kv: kv[0]):
        rng.shuffle(items)
        if len(items) <= 1 or val_ratio == 0:
            train_split.extend(items)
            continue

        val_count = int(round(len(items) * val_ratio))
        val_count = max(1, val_count)
        if val_count >= len(items):
            val_count = len(items) - 1

        val_split.extend(items[:val_count])
        train_split.extend(items[val_count:])

    return sorted(train_split), sorted(val_split)


def _write_data_yaml(dataset_root: Path, names: Sequence[str]) -> Path:
    yaml_path = dataset_root / "data.yaml"
    quoted = [f"'{name}'" for name in names]
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(names)}",
                "names:",
                *[f"  - {name}" for name in quoted],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def prepare_opixray_yolo_dataset(
    *,
    data_root: Union[str, Path],
    output_root: Union[str, Path],
    val_ratio: float = 0.1,
    split_seed: int = 42,
    link_mode: str = "auto",
    overwrite: bool = False,
    class_names: Optional[Sequence[str]] = None,
) -> Path:
    """Converts OPIXray detection annotations into a YOLO-style dataset."""

    if link_mode not in {"auto", "symlink", "hardlink", "copy"}:
        raise ValueError("link_mode must be one of: auto, symlink, hardlink, copy")

    data_root = Path(data_root)
    output_root = Path(output_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    train_ann = parse_opixray_subset(data_root, "train")
    test_ann = parse_opixray_subset(data_root, "test")

    all_boxes: List[OPIXrayBox] = []
    for boxes in list(train_ann.values()) + list(test_ann.values()):
        all_boxes.extend(boxes)
    discovered = sorted({box.label for box in all_boxes})
    names = list(class_names) if class_names is not None else list(discovered)
    if not names:
        raise RuntimeError("No class names discovered from annotations.")

    class_to_id = {name: idx for idx, name in enumerate(names)}

    train_images, val_images = split_train_val(train_ann, val_ratio=val_ratio, seed=split_seed)
    test_images = sorted(test_ann.keys())

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root} (use overwrite=True)"
            )
        _safe_rmtree(output_root)

    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def emit_split(
        *,
        split: str,
        subset: str,
        image_names: Iterable[str],
        annotations: Mapping[str, Sequence[OPIXrayBox]],
    ) -> None:
        src_image_dir = data_root / subset / f"{subset}_image"
        out_images = output_root / "images" / split
        out_labels = output_root / "labels" / split

        for image_name in image_names:
            src_img = src_image_dir / image_name
            dst_img = out_images / image_name
            _link_or_copy_file(src_img, dst_img, mode=link_mode)

            width, height = _image_size(src_img)
            label_path = out_labels / f"{Path(image_name).stem}.txt"

            boxes = annotations.get(image_name, [])
            lines: List[str] = []
            for box in boxes:
                if box.label not in class_to_id:
                    continue
                x1, y1, x2, y2 = _clamp_bbox(
                    width=width,
                    height=height,
                    bbox=(box.x1, box.y1, box.x2, box.y2),
                )
                xc, yc, bw, bh = _to_yolo_xywh(width, height, x1, y1, x2, y2)
                cls_id = class_to_id[box.label]
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    emit_split(split="train", subset="train", image_names=train_images, annotations=train_ann)
    emit_split(split="val", subset="train", image_names=val_images, annotations=train_ann)
    emit_split(split="test", subset="test", image_names=test_images, annotations=test_ann)

    return _write_data_yaml(output_root, names=names)


def find_data_yaml(root: Union[str, Path]) -> Path:
    root = Path(root)
    direct = root / "data.yaml"
    if direct.exists():
        return direct

    matches = sorted(root.rglob("data.yaml"))
    if not matches:
        raise FileNotFoundError(f"No data.yaml found under {root}")
    return matches[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OPIXray in YOLO format.")
    parser.add_argument("--data-root", type=Path, required=True, help="Extracted OPIXray root.")
    parser.add_argument("--output-root", type=Path, required=True, help="YOLO output root.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--link-mode",
        choices=("auto", "symlink", "hardlink", "copy"),
        default="auto",
        help="How to place image files in the output dataset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory.")
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Optional class names list (defaults to dataset labels).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    class_names = args.class_names if args.class_names else None
    prepare_opixray_yolo_dataset(
        data_root=args.data_root,
        output_root=args.output_root,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
