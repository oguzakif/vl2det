#!/usr/bin/env python3
"""Run detection inference on a file or folder and save images + JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from torchvision.transforms import functional as TF

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.detection import DetectionTeacher

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PALETTE = [
    (230, 57, 70),
    (29, 53, 87),
    (69, 123, 157),
    (241, 250, 238),
    (255, 183, 3),
    (251, 133, 0),
    (16, 185, 129),
    (99, 102, 241),
    (244, 114, 182),
    (14, 116, 144),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detection inference with drawing + JSON export.")
    parser.add_argument("input", help="Image file or folder.")
    parser.add_argument("--ckpt", required=True, help="Lightning checkpoint path.")
    parser.add_argument("--arch", required=True, help="Detection arch (e.g., fasterrcnn_mobilenet_v3_large_fpn).")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of foreground classes.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--device", default=None, help="Device (cuda, cpu, or cuda:0).")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="Score threshold.")
    parser.add_argument("--topk", type=int, default=0, help="Keep top-k detections per image (0 disables).")
    parser.add_argument("--label-offset", type=int, default=1, help="Label offset used in training.")
    parser.add_argument("--data-yaml", default=None, help="YOLO data.yaml to read class names.")
    parser.add_argument("--class-names", nargs="*", default=None, help="Override class names list.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    parser.add_argument("--no-background", dest="add_background", action="store_false", help="Disable background class.")
    parser.add_argument("--image-size", type=int, default=None, help="Fixed input size for EfficientDet.")
    parser.add_argument("--box-format", default=None, help="Box format for EfficientDet (xyxy or yxyx).")
    parser.add_argument("--weights", default="DEFAULT", help="Torchvision weights enum or 'DEFAULT'.")
    parser.add_argument("--weights-backbone", default=None, help="Backbone weights enum.")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width.")
    parser.set_defaults(add_background=True)
    return parser.parse_args()


def _collect_images(input_path: Path, recursive: bool) -> List[Tuple[Path, Path]]:
    if input_path.is_file():
        return [(input_path, Path(input_path.name))]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pattern = "**/*" if recursive else "*"
    items: List[Tuple[Path, Path]] = []
    for path in sorted(input_path.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rel = path.relative_to(input_path)
            items.append((path, rel))
    return items


def _parse_class_names(data_yaml: Path) -> List[str]:
    cfg = OmegaConf.load(data_yaml)
    names = cfg.get("names")
    if names is None:
        return []
    if isinstance(names, dict):
        ordered = []
        for key in sorted(names.keys(), key=lambda k: int(k)):
            ordered.append(str(names[key]))
        return ordered
    return [str(name) for name in names]


def _resolve_class_names(args: argparse.Namespace) -> List[str]:
    if args.class_names:
        return list(args.class_names)
    if args.data_yaml:
        return _parse_class_names(Path(args.data_yaml))
    return []


def _color_for_label(label: int) -> Tuple[int, int, int]:
    return PALETTE[label % len(PALETTE)]


def _draw_detections(
    image: Image.Image,
    detections: List[dict],
    line_width: int,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = det["color"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = det["label_name"]
        score = det["score"]
        text = f"{label} {score:.2f}"
        try:
            bbox = draw.textbbox((x1, y1), text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = draw.textsize(text)
        text_x1 = max(0, x1)
        text_y1 = max(0, y1 - text_h - 2)
        text_x2 = text_x1 + text_w + 4
        text_y2 = text_y1 + text_h + 2
        draw.rectangle([text_x1, text_y1, text_x2, text_y2], fill=color)
        draw.text((text_x1 + 2, text_y1 + 1), text, fill=(0, 0, 0))
    return image


def _prepare_detections(
    outputs: dict,
    score_threshold: float,
    topk: int,
    label_offset: int,
    class_names: List[str],
) -> List[dict]:
    boxes = outputs.get("boxes", torch.empty((0, 4)))
    scores = outputs.get("scores", torch.empty((0,)))
    labels = outputs.get("labels", torch.empty((0,), dtype=torch.int64))

    if boxes.numel() == 0:
        return []

    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if topk and scores.numel() > topk:
        values, idx = torch.topk(scores, k=topk)
        scores = values
        boxes = boxes[idx]
        labels = labels[idx]

    detections: List[dict] = []
    for box, score, label in zip(boxes, scores, labels):
        raw_label = int(label.item())
        mapped = raw_label - label_offset
        if mapped < 0:
            mapped = raw_label
        name = str(mapped)
        if 0 <= mapped < len(class_names):
            name = class_names[mapped]
        color = _color_for_label(mapped)
        detections.append(
            {
                "bbox": [float(x) for x in box.tolist()],
                "score": float(score.item()),
                "label": mapped,
                "model_label": raw_label,
                "label_name": name,
                "color": color,
            }
        )
    return detections


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    image_out_dir = out_dir / "images"
    json_out_dir = out_dir / "json"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    json_out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    class_names = _resolve_class_names(args)

    model = DetectionTeacher(
        arch=args.arch,
        num_classes=args.num_classes,
        weights=args.weights,
        weights_backbone=args.weights_backbone,
        add_background=args.add_background,
        ckpt_path=args.ckpt,
        label_offset=args.label_offset,
        image_size=args.image_size,
        box_format=args.box_format,
    )
    model.to(device)
    model.eval()

    items = _collect_images(input_path, args.recursive)
    if not items:
        raise SystemExit(f"No images found in {input_path}")

    for image_path, rel_path in items:
        image = Image.open(image_path).convert("RGB")
        tensor = TF.to_tensor(image).to(device)
        with torch.no_grad():
            outputs = model([tensor])
        if not outputs:
            detections = []
        else:
            detections = _prepare_detections(
                outputs=outputs[0],
                score_threshold=float(args.score_threshold),
                topk=int(args.topk),
                label_offset=int(args.label_offset),
                class_names=class_names,
            )

        rendered = image.copy()
        rendered = _draw_detections(rendered, detections, line_width=args.line_width)

        out_image_path = image_out_dir / rel_path
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(out_image_path)

        out_json_path = json_out_dir / rel_path.with_suffix(".json")
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        json_payload = {
            "image": str(image_path),
            "width": image.width,
            "height": image.height,
            "detections": [
                {
                    "bbox": det["bbox"],
                    "bbox_format": "xyxy",
                    "score": det["score"],
                    "label": det["label"],
                    "label_name": det["label_name"],
                    "model_label": det["model_label"],
                }
                for det in detections
            ],
        }
        out_json_path.write_text(json.dumps(json_payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
