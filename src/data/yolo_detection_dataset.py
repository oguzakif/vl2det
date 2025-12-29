from __future__ import annotations

from pathlib import Path
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import random

from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

import math

def _is_image_file(path: Path, exts: Sequence[str]) -> bool:
    return path.suffix.lower() in exts


def _list_images(root: Path, exts: Sequence[str]) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and _is_image_file(p, exts)])


def _resolve_yaml_path(data_root: Path, data_yaml: str) -> Path:
    yaml_path = Path(data_yaml)
    if not yaml_path.is_absolute():
        yaml_path = data_root / yaml_path
    return yaml_path


def _parse_class_names(classes: Union[Dict, List, DictConfig, ListConfig]) -> List[str]:
    if isinstance(classes, (dict, DictConfig)):
        def _key_as_int(item: Tuple[object, object]) -> int:
            try:
                return int(item[0])
            except (TypeError, ValueError):
                return 0

        return [name for _, name in sorted(classes.items(), key=_key_as_int)]
    return list(classes)


def _resolve_split_entries(
    split_value: Union[str, List, ListConfig], root: Path, exts: Sequence[str]
) -> List[Path]:
    if isinstance(split_value, (list, ListConfig)):
        paths: List[Path] = []
        for entry in split_value:
            paths.extend(_resolve_split_entries(entry, root, exts))
        return paths

    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = root / split_path

    if split_path.is_dir():
        return _list_images(split_path, exts)

    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        entries = []
        for line in split_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entry_path = Path(line)
            if not entry_path.is_absolute():
                entry_path = root / entry_path
            entries.append(entry_path)
        return entries

    if split_path.exists():
        return [split_path]

    return []


class YoloDetectionDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        attributes,
        data_yaml: Optional[str] = None,
        images_dir: Optional[str] = None,
        labels_dir: Optional[str] = None,
        image_extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
        use_teacher: bool = True,
        teacher_image_size: int = 224,
        augment: bool = False,
        hsv_h: float = 0.0,
        hsv_s: float = 0.0,
        hsv_v: float = 0.0,
        degrees: float = 0.0,
        translate: float = 0.0,
        scale: float = 0.0,
        flip_prob: float = 0.5,
        det_teacher_use_raw_images: bool = False,
        label_offset: int = 1,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.attributes = attributes
        self.use_teacher = use_teacher
        self.teacher_image_size = teacher_image_size
        self.augment = bool(augment and split == "train")
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self.flip_prob = flip_prob if split == "train" else 0.0
        self.det_teacher_use_raw_images = bool(det_teacher_use_raw_images)
        self.label_offset = label_offset
        self.image_extensions = image_extensions

        self.yaml_path = _resolve_yaml_path(self.data_root, data_yaml) if data_yaml else None
        yaml_cfg = OmegaConf.load(self.yaml_path) if self.yaml_path else None

        self.class_names = self._resolve_class_names(attributes, yaml_cfg)
        self.class_num = len(self.class_names)

        self.image_paths = self._resolve_image_paths(
            yaml_cfg=yaml_cfg,
            images_dir=images_dir,
        )
        self.labels_dir = self._resolve_labels_dir(labels_dir=labels_dir)

        self.student_transform = transforms.ToTensor()
        self.teacher_transform = transforms.Compose(
            [
                transforms.Resize((teacher_image_size, teacher_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def _resolve_class_names(self, attributes, yaml_cfg) -> List[str]:
        if attributes is not None and getattr(attributes, "classes", None) is not None:
            return _parse_class_names(attributes.classes)

        if yaml_cfg is not None and yaml_cfg.get("names") is not None:
            return _parse_class_names(yaml_cfg.get("names"))

        return []

    def _resolve_image_paths(self, yaml_cfg, images_dir: Optional[str]) -> List[Path]:
        if yaml_cfg is not None and yaml_cfg.get(self.split) is not None:
            yaml_root = Path(yaml_cfg.get("path") or self.data_root)
            if not yaml_root.is_absolute():
                yaml_root = (self.yaml_path.parent / yaml_root).resolve()
            return _resolve_split_entries(yaml_cfg.get(self.split), yaml_root, self.image_extensions)

        if images_dir is None:
            raise ValueError("images_dir is required when data_yaml is not provided.")

        images_path = Path(images_dir)
        if not images_path.is_absolute():
            images_path = self.data_root / images_path
        return _list_images(images_path, self.image_extensions)

    def _resolve_labels_dir(self, labels_dir: Optional[str]) -> Optional[Path]:
        if labels_dir is None:
            return None
        labels_path = Path(labels_dir)
        if not labels_path.is_absolute():
            labels_path = self.data_root / labels_path
        return labels_path

    def _label_path_for_image(self, image_path: Path) -> Path:
        if self.labels_dir is not None:
            return self.labels_dir / f"{image_path.stem}.txt"
        label_path = Path(
            str(image_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        )
        return label_path.with_suffix(".txt")

    def _load_yolo_labels(self, label_path: Path, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        width, height = image_size
        boxes: List[List[float]] = []
        labels: List[int] = []

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                box_w = float(parts[3]) * width
                box_h = float(parts[4]) * height
                x1 = x_center - box_w / 2.0
                y1 = y_center - box_h / 2.0
                x2 = x_center + box_w / 2.0
                y2 = y_center + box_h / 2.0
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id + self.label_offset)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        if boxes_tensor.numel() > 0:
            boxes_tensor[:, 0::2] = boxes_tensor[:, 0::2].clamp(0, width)
            boxes_tensor[:, 1::2] = boxes_tensor[:, 1::2].clamp(0, height)
            keep = (boxes_tensor[:, 2] > boxes_tensor[:, 0]) & (boxes_tensor[:, 3] > boxes_tensor[:, 1])
            boxes_tensor = boxes_tensor[keep]
            labels_tensor = labels_tensor[keep]

        return boxes_tensor, labels_tensor

    def _clip_boxes(
        self,
        boxes: torch.Tensor,
        width: int,
        height: int,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if boxes.numel() == 0:
            return boxes, labels
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]
        if labels is None:
            return boxes, labels
        return boxes, labels[keep]

    def _apply_hsv_jitter(self, image: Image.Image) -> Image.Image:
        if self.hsv_h == 0.0 and self.hsv_s == 0.0 and self.hsv_v == 0.0:
            return image
        hue = random.uniform(-self.hsv_h, self.hsv_h)
        sat = 1.0 + random.uniform(-self.hsv_s, self.hsv_s)
        val = 1.0 + random.uniform(-self.hsv_v, self.hsv_v)
        sat = max(sat, 0.0)
        val = max(val, 0.0)
        image = F.adjust_hue(image, hue)
        image = F.adjust_saturation(image, sat)
        image = F.adjust_brightness(image, val)
        return image

    def _apply_affine(
        self, image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        if self.degrees == 0.0 and self.translate == 0.0 and self.scale == 0.0:
            return image, boxes, labels

        width, height = image.size
        angle = random.uniform(-self.degrees, self.degrees)
        scale = 1.0 + random.uniform(-self.scale, self.scale)
        max_dx = self.translate * width
        max_dy = self.translate * height
        tx = int(round(random.uniform(-max_dx, max_dx)))
        ty = int(round(random.uniform(-max_dy, max_dy)))

        image = F.affine(image, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0])

        if boxes.numel() == 0:
            return image, boxes, labels

        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad) * scale
        sin_a = math.sin(angle_rad) * scale
        cx, cy = width / 2.0, height / 2.0

        corners = torch.stack(
            [
                boxes[:, [0, 1]],
                boxes[:, [2, 1]],
                boxes[:, [2, 3]],
                boxes[:, [0, 3]],
            ],
            dim=1,
        )
        corners[:, :, 0] -= cx
        corners[:, :, 1] -= cy

        x = corners[:, :, 0]
        y = corners[:, :, 1]
        new_x = cos_a * x - sin_a * y + cx + tx
        new_y = sin_a * x + cos_a * y + cy + ty

        new_corners = torch.stack([new_x, new_y], dim=-1)
        new_boxes = torch.zeros_like(boxes)
        new_boxes[:, 0] = new_corners[:, :, 0].min(dim=1).values
        new_boxes[:, 1] = new_corners[:, :, 1].min(dim=1).values
        new_boxes[:, 2] = new_corners[:, :, 0].max(dim=1).values
        new_boxes[:, 3] = new_corners[:, :, 1].max(dim=1).values

        new_boxes, labels = self._clip_boxes(new_boxes, width, height, labels)
        return image, new_boxes, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        raw_image = image.copy()
        width, height = image.size

        label_path = self._label_path_for_image(image_path)
        boxes, labels = self._load_yolo_labels(label_path, (width, height))

        if self.augment:
            image = self._apply_hsv_jitter(image)
            image, boxes, labels = self._apply_affine(image, boxes, labels)
            width, height = image.size

        if self.augment and self.flip_prob > 0.0 and random.random() < self.flip_prob:
            image = F.hflip(image)
            if boxes.numel() > 0:
                x1 = width - boxes[:, 2]
                x2 = width - boxes[:, 0]
                boxes[:, 0] = x1
                boxes[:, 2] = x2
                boxes, labels = self._clip_boxes(boxes, width, height, labels)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if boxes.numel() > 0 else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }

        student_image = self.student_transform(image)
        teacher_image = self.teacher_transform(image) if self.use_teacher else None
        det_teacher_image = (
            self.student_transform(raw_image) if self.det_teacher_use_raw_images else None
        )

        if self.det_teacher_use_raw_images:
            return student_image, target, teacher_image, det_teacher_image
        return student_image, target, teacher_image


def yolo_collate_fn(batch):
    if len(batch[0]) == 4:
        images, targets, teacher_images, det_teacher_images = zip(*batch)
        images_list = list(images)
        targets_list = list(targets)
        teacher_images_tensor = None
        if teacher_images[0] is not None:
            teacher_images_tensor = torch.stack(teacher_images)
        det_teacher_images_list = None
        if det_teacher_images[0] is not None:
            det_teacher_images_list = list(det_teacher_images)
        return images_list, targets_list, teacher_images_tensor, det_teacher_images_list

    images, targets, teacher_images = zip(*batch)
    images_list = list(images)
    targets_list = list(targets)
    if teacher_images[0] is None:
        return images_list, targets_list, None
    return images_list, targets_list, torch.stack(teacher_images)
