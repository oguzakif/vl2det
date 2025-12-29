from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import inspect

from omegaconf import DictConfig, ListConfig

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models.detection as detection_models

from src.models.components.campus import AlignNet, TeacherNet, feature_norm


_EFFICIENTDET_ARCH_MAP = {
    "efficientdet_d0": "tf_efficientdet_d0",
    "efficientdet_d1": "tf_efficientdet_d1",
    "tf_efficientdet_d0": "tf_efficientdet_d0",
    "tf_efficientdet_d1": "tf_efficientdet_d1",
}


def _is_efficientdet_arch(arch: str) -> bool:
    return arch in _EFFICIENTDET_ARCH_MAP


def _normalize_efficientdet_arch(arch: str) -> str:
    return _EFFICIENTDET_ARCH_MAP.get(arch, arch)


def _resolve_image_size(image_size: Optional[object]) -> Optional[Tuple[int, int]]:
    if image_size is None:
        return None
    if isinstance(image_size, (tuple, list, ListConfig)):
        if len(image_size) == 1:
            size = int(image_size[0])
            return size, size
        if len(image_size) != 2:
            raise ValueError("image_size must be an int or a (height, width) tuple.")
        return int(image_size[0]), int(image_size[1])
    size = int(image_size)
    return size, size


def _build_detection_model(
    arch: str,
    num_classes: int,
    weights: Optional[object] = None,
    weights_backbone: Optional[object] = None,
    *,
    label_offset: int = 1,
    image_size: Optional[object] = None,
    box_format: Optional[str] = None,
) -> nn.Module:
    if _is_efficientdet_arch(arch):
        return _build_efficientdet_model(
            arch=arch,
            num_classes=num_classes,
            weights=weights,
            image_size=image_size,
            label_offset=label_offset,
            box_format=box_format,
        )

    builder = getattr(detection_models, arch, None)
    if builder is None:
        raise ValueError(f"Unsupported detection architecture: {arch}")

    if weights is not None:
        model = _build_without_weights(builder, num_classes, weights_backbone)
        state_dict = _get_weights_state_dict(weights)
        _load_state_dict_filtered(model, state_dict)
        return model

    return _build_without_weights(builder, num_classes, weights_backbone)


def _build_without_weights(builder, num_classes, weights_backbone):
    try:
        return builder(weights=None, weights_backbone=weights_backbone, num_classes=num_classes)
    except TypeError:
        return builder(
            pretrained=False,
            pretrained_backbone=weights_backbone is not None,
            num_classes=num_classes,
        )


def _get_weights_state_dict(weights) -> Dict[str, torch.Tensor]:
    if hasattr(weights, "get_state_dict"):
        return weights.get_state_dict(progress=True)
    if isinstance(weights, dict):
        return weights
    raise ValueError("Unsupported weights type; expected a torchvision weights enum or state dict.")


def _load_state_dict_filtered(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(filtered, strict=False)


_WEIGHTS_CLASS_MAP = {
    "ssdlite320_mobilenet_v3_large": "SSDLite320_MobileNet_V3_Large_Weights",
    "ssd300_vgg16": "SSD300_VGG16_Weights",
    "fasterrcnn_resnet50_fpn": "FasterRCNN_ResNet50_FPN_Weights",
    "fasterrcnn_resnet50_fpn_v2": "FasterRCNN_ResNet50_FPN_V2_Weights",
    "fasterrcnn_mobilenet_v3_large_fpn": "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
    "retinanet_resnet50_fpn": "RetinaNet_ResNet50_FPN_Weights",
    "retinanet_resnet50_fpn_v2": "RetinaNet_ResNet50_FPN_V2_Weights",
    "maskrcnn_resnet50_fpn": "MaskRCNN_ResNet50_FPN_Weights",
    "maskrcnn_resnet50_fpn_v2": "MaskRCNN_ResNet50_FPN_V2_Weights",
    "keypointrcnn_resnet50_fpn": "KeypointRCNN_ResNet50_FPN_Weights",
    "fcos_resnet50_fpn": "FCOS_ResNet50_FPN_Weights",
}


def _normalize_weights_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in {"none", "null", ""}:
            return None
        return normalized
    return value


def _resolve_default_weights(arch: str):
    weights_class_name = _WEIGHTS_CLASS_MAP.get(arch)
    if weights_class_name and hasattr(detection_models, weights_class_name):
        return getattr(detection_models, weights_class_name).DEFAULT

    try:
        from torchvision.models import get_model_weights
    except Exception:
        get_model_weights = None

    if get_model_weights is not None:
        try:
            return get_model_weights(arch).DEFAULT
        except Exception:
            pass

    raise ValueError(
        f"Could not resolve DEFAULT weights for '{arch}'. "
        "Set weights to null or pass a torchvision weights enum."
    )


def _resolve_weights(arch: str, weights, weights_backbone) -> Tuple[Optional[object], Optional[object]]:
    weights = _normalize_weights_value(weights)
    weights_backbone = _normalize_weights_value(weights_backbone)

    if isinstance(weights, str):
        if weights.upper() == "DEFAULT":
            weights = _resolve_default_weights(arch)
        elif weights.endswith(".DEFAULT"):
            class_name = weights.split(".")[0]
            if hasattr(detection_models, class_name):
                weights = getattr(detection_models, class_name).DEFAULT
            else:
                raise ValueError(f"Unknown weights class: {class_name}")

    if isinstance(weights_backbone, str):
        if weights_backbone.upper() == "DEFAULT":
            weights_backbone = None
        elif weights_backbone.endswith(".DEFAULT"):
            raise ValueError(
                "weights_backbone expects a backbone weights enum; "
                "pass a resolved enum instance instead."
            )

    return weights, weights_backbone


def _build_effdet_labeler(config: object):
    try:
        from effdet.anchors import Anchors, AnchorLabeler
    except Exception:
        try:
            from effdet.anchor import Anchors, AnchorLabeler
        except Exception:
            return None

    anchors = None
    if hasattr(Anchors, "from_config"):
        try:
            anchors = Anchors.from_config(config)
        except Exception:
            anchors = None

    if anchors is None:
        try:
            anchors = Anchors(config)
        except TypeError:
            anchors = None

    if anchors is None:
        try:
            anchors = Anchors(
                min_level=getattr(config, "min_level", 3),
                max_level=getattr(config, "max_level", 7),
                num_scales=getattr(config, "num_scales", 3),
                aspect_ratios=getattr(config, "aspect_ratios", (1.0, 2.0, 0.5)),
                anchor_scale=getattr(config, "anchor_scale", 4.0),
                image_size=getattr(config, "image_size", None),
            )
        except Exception:
            anchors = None

    labeler = None
    num_classes = int(getattr(config, "num_classes", 0))
    match_thresh = getattr(config, "match_threshold", 0.5)
    unmatched_thresh = getattr(config, "unmatched_threshold", 0.4)

    candidates = []
    try:
        params = inspect.signature(AnchorLabeler).parameters
    except (TypeError, ValueError):
        params = None

    if params is not None:
        if "anchors" in params and "num_classes" in params:
            candidates.append((anchors, num_classes, match_thresh, unmatched_thresh))
            candidates.append((anchors, num_classes))
        if "anchors" in params and "config" in params:
            if list(params.keys())[1] == "anchors":
                candidates.append((config, anchors))
            else:
                candidates.append((anchors, config))
        if "anchors" in params:
            candidates.append((anchors,))
        if "config" in params:
            candidates.append((config,))
        candidates.append(())
    else:
        candidates = [
            (anchors, num_classes, match_thresh, unmatched_thresh),
            (anchors, num_classes),
            (anchors, config),
            (config, anchors),
            (anchors,),
            (config,),
            (),
        ]

    for args in candidates:
        if anchors is None and any(arg is anchors for arg in args):
            continue
        try:
            labeler = AnchorLabeler(*args)
            break
        except Exception:
            continue

    return labeler


class EfficientDetWrapper(nn.Module):
    def __init__(
        self,
        train_model: nn.Module,
        predict_model: nn.Module,
        config: object,
        label_offset: int = 1,
        box_format: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.train_model = train_model
        self.predict_model = predict_model
        self.config = config
        self.label_offset = int(label_offset)
        self.box_format = (box_format or "yxyx").lower()

        self.image_size = _resolve_image_size(getattr(config, "image_size", None))
        self.input_size = self.image_size
        self.image_mean = tuple(getattr(config, "image_mean", (0.485, 0.456, 0.406)))
        self.image_std = tuple(getattr(config, "image_std", (0.229, 0.224, 0.225)))
        self.anchor_labeler = _build_effdet_labeler(config)

    def _resize_image(self, image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _prepare_images(
        self, images: Iterable[torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[Tuple[float, float], ...], Tuple[Tuple[int, int], ...]]:
        if isinstance(images, torch.Tensor):
            batch = images
            orig_h, orig_w = images.shape[-2], images.shape[-1]
            if self.image_size is not None and (orig_h, orig_w) != self.image_size:
                target_h, target_w = self.image_size
                scale_y = target_h / float(orig_h)
                scale_x = target_w / float(orig_w)
                batch = F.interpolate(
                    images,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
                scales = tuple((scale_x, scale_y) for _ in range(images.shape[0]))
                sizes = tuple((target_h, target_w) for _ in range(images.shape[0]))
                return batch, scales, sizes

            scales = tuple((1.0, 1.0) for _ in range(images.shape[0]))
            sizes = tuple((orig_h, orig_w) for _ in range(images.shape[0]))
            return batch, scales, sizes

        image_list = list(images)
        original_sizes = tuple((img.shape[-2], img.shape[-1]) for img in image_list)
        scales: list[Tuple[float, float]] = []
        resized_images = []
        if self.image_size is not None:
            target_h, target_w = self.image_size
            for img, (orig_h, orig_w) in zip(image_list, original_sizes):
                scale_y = target_h / float(orig_h)
                scale_x = target_w / float(orig_w)
                scales.append((scale_x, scale_y))
                if (orig_h, orig_w) != (target_h, target_w):
                    img = self._resize_image(img, (target_h, target_w))
                resized_images.append(img)
            sizes = tuple((target_h, target_w) for _ in image_list)
        else:
            if len({(h, w) for h, w in original_sizes}) > 1:
                raise ValueError(
                    "EfficientDet requires a fixed input size. "
                    "Set model.net.student.image_size to a square size (e.g., 512)."
                )
            scales = [(1.0, 1.0) for _ in image_list]
            resized_images = image_list
            sizes = original_sizes

        batch = torch.stack(resized_images, dim=0)
        return batch, tuple(scales), sizes

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        if not self.image_mean or not self.image_std:
            return images
        mean = torch.tensor(self.image_mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        return (images - mean) / std

    def _resize_targets(
        self,
        targets: Iterable[Dict[str, torch.Tensor]],
        scales: Tuple[Tuple[float, float], ...],
    ) -> list[Dict[str, torch.Tensor]]:
        resized = []
        for target, (scale_x, scale_y) in zip(targets, scales):
            if scale_x == 1.0 and scale_y == 1.0:
                resized.append(target)
                continue
            boxes = target["boxes"].clone()
            if boxes.numel() > 0:
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
            updated = dict(target)
            updated["boxes"] = boxes
            resized.append(updated)
        return resized

    def _to_model_box_format(self, boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        if self.box_format == "yxyx":
            return boxes[:, [1, 0, 3, 2]]
        return boxes

    def _to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        if self.box_format == "yxyx":
            return boxes[:, [1, 0, 3, 2]]
        return boxes

    def _build_effdet_targets(
        self,
        targets: Iterable[Dict[str, torch.Tensor]],
        sizes: Tuple[Tuple[int, int], ...],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        target_list = list(targets)
        batch_size = len(target_list)
        img_size = torch.tensor(sizes, device=device, dtype=torch.float32)
        img_scale = torch.ones((batch_size,), device=device, dtype=torch.float32)

        if self.anchor_labeler is None:
            raise ValueError(
                "EfficientDet training requires anchor labeler support. "
                "Install a recent `effdet` package or provide a custom target encoder."
            )

        boxes_list = [self._to_model_box_format(tgt["boxes"]).detach().cpu() for tgt in target_list]
        labels_list = [(tgt["labels"] - self.label_offset).detach().cpu() for tgt in target_list]

        cls_targets = None
        box_targets = None
        num_pos = None

        if hasattr(self.anchor_labeler, "batch_label_anchors"):
            try:
                cls_targets, box_targets, num_pos = self.anchor_labeler.batch_label_anchors(
                    boxes_list, labels_list
                )
            except Exception:
                max_num = max((b.shape[0] for b in boxes_list), default=0)
                max_num = max(max_num, 1)
                padded_boxes = torch.zeros((batch_size, max_num, 4), dtype=boxes_list[0].dtype)
                padded_labels = torch.full((batch_size, max_num), -1, dtype=labels_list[0].dtype)
                for i, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
                    if boxes.numel() == 0:
                        continue
                    num = boxes.shape[0]
                    padded_boxes[i, :num] = boxes
                    padded_labels[i, :num] = labels
                try:
                    cls_targets, box_targets, num_pos = self.anchor_labeler.batch_label_anchors(
                        padded_boxes, padded_labels
                    )
                except TypeError:
                    cls_targets, box_targets, num_pos = self.anchor_labeler.batch_label_anchors(
                        padded_boxes, padded_labels, sizes
                    )
        elif hasattr(self.anchor_labeler, "label_anchors"):
            cls_targets_list = []
            box_targets_list = []
            num_pos_list = []
            for boxes, labels in zip(boxes_list, labels_list):
                out = self.anchor_labeler.label_anchors(boxes, labels)
                if isinstance(out, dict):
                    cls_targets_list.append(out["cls"])
                    box_targets_list.append(out["bbox"])
                    num_pos_list.append(out.get("label_num_positives", 0))
                else:
                    cls_t, box_t, num = out
                    cls_targets_list.append(cls_t)
                    box_targets_list.append(box_t)
                    num_pos_list.append(num)
            cls_targets = cls_targets_list
            box_targets = box_targets_list
            num_pos = num_pos_list
        elif callable(self.anchor_labeler):
            out = self.anchor_labeler(boxes_list, labels_list)
            if isinstance(out, dict):
                cls_targets = out.get("cls")
                box_targets = out.get("bbox")
                num_pos = out.get("label_num_positives")
            else:
                cls_targets, box_targets, num_pos = out
        else:
            raise ValueError("Unsupported anchor labeler interface for EfficientDet.")

        def _to_tensor(value):
            if torch.is_tensor(value):
                return value.to(device)
            if isinstance(value, (list, tuple)):
                tensors = [v if torch.is_tensor(v) else torch.as_tensor(v) for v in value]
                shapes = {tuple(t.shape) for t in tensors}
                if len(shapes) == 1:
                    return torch.stack([t.to(device) for t in tensors], dim=0)
                return [t.to(device) for t in tensors]
            return torch.as_tensor(value, device=device)

        cls_targets = _to_tensor(cls_targets)
        box_targets = _to_tensor(box_targets)
        num_pos = _to_tensor(num_pos)

        target_dict = {
            "label_num_positives": num_pos,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        if isinstance(cls_targets, dict):
            target_dict.update(cls_targets)
            if isinstance(box_targets, dict):
                target_dict.update(box_targets)
            else:
                target_dict["bbox"] = box_targets
            return target_dict

        if isinstance(box_targets, dict):
            target_dict.update(box_targets)
            target_dict["cls"] = cls_targets
            return target_dict

        num_levels = getattr(self.train_model, "num_levels", None)
        if isinstance(cls_targets, torch.Tensor) and num_levels and cls_targets.shape[0] == num_levels:
            cls_targets = list(cls_targets)
        if isinstance(box_targets, torch.Tensor) and num_levels and box_targets.shape[0] == num_levels:
            box_targets = list(box_targets)

        if isinstance(cls_targets, list) and isinstance(box_targets, list):
            for level, (cls_t, box_t) in enumerate(zip(cls_targets, box_targets)):
                target_dict[f"label_cls_{level}"] = cls_t
                target_dict[f"label_bbox_{level}"] = box_t
            target_dict["cls"] = cls_targets
            target_dict["bbox"] = box_targets
            return target_dict

        target_dict["cls"] = cls_targets
        target_dict["bbox"] = box_targets
        return target_dict

    def _format_predictions(
        self,
        preds,
        scales: Tuple[Tuple[float, float], ...],
    ) -> list[Dict[str, torch.Tensor]]:
        if isinstance(preds, dict) and "boxes" in preds:
            pred_list = [preds]
        elif isinstance(preds, (list, tuple)):
            pred_list = list(preds)
        elif torch.is_tensor(preds):
            pred_list = list(preds)
        else:
            raise ValueError("Unexpected EfficientDet prediction output format.")

        outputs: list[Dict[str, torch.Tensor]] = []
        for idx, det in enumerate(pred_list):
            if isinstance(det, dict) and {"boxes", "scores", "labels"}.issubset(det):
                boxes = det["boxes"]
                scores = det["scores"]
                labels = det["labels"].to(torch.int64)
            else:
                if det.numel() == 0:
                    outputs.append(
                        {
                            "boxes": torch.zeros((0, 4), device=det.device),
                            "scores": torch.zeros((0,), device=det.device),
                            "labels": torch.zeros((0,), dtype=torch.int64, device=det.device),
                        }
                    )
                    continue
                boxes = det[:, 0:4]
                scores = det[:, 4]
                labels = det[:, 5].to(torch.int64)

            boxes = self._to_xyxy(boxes)

            if self.label_offset:
                labels = labels + self.label_offset

            if scales:
                scale_x, scale_y = scales[idx]
                if scale_x != 1.0 or scale_y != 1.0:
                    boxes = boxes.clone()
                    boxes[:, [0, 2]] = boxes[:, [0, 2]] / scale_x
                    boxes[:, [1, 3]] = boxes[:, [1, 3]] / scale_y

            if scores.numel() > 0:
                keep = scores > 0
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            outputs.append({"boxes": boxes, "scores": scores, "labels": labels})

        return outputs

    def forward(self, images, targets=None):
        batch, scales, sizes = self._prepare_images(images)
        batch = self._normalize_images(batch)

        if targets is not None and self.training:
            resized_targets = self._resize_targets(targets, scales)
            effdet_targets = self._build_effdet_targets(resized_targets, sizes, batch.device)
            losses = self.train_model(batch, effdet_targets)
            if isinstance(losses, dict):
                return losses
            if isinstance(losses, (tuple, list)) and losses:
                return {"loss": losses[0]}
            return {"loss": losses}

        preds = self.predict_model(batch)
        return self._format_predictions(preds, scales)

    def extract_features(self, images):
        batch, _, _ = self._prepare_images(images)
        batch = self._normalize_images(batch)
        backbone = getattr(self.train_model, "model", self.train_model)
        features = backbone.backbone(batch)
        if isinstance(features, torch.Tensor):
            feature_map = features
        elif isinstance(features, dict):
            feature_map = list(features.values())[-1]
        else:
            feature_map = features[-1]
        pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        return feature_norm(pooled)


def _build_efficientdet_model(
    arch: str,
    num_classes: int,
    weights: Optional[object],
    image_size: Optional[object],
    label_offset: int,
    box_format: Optional[str],
) -> nn.Module:
    try:
        from effdet import DetBenchPredict, DetBenchTrain, create_model, get_efficientdet_config
    except Exception as exc:
        raise ImportError(
            "EfficientDet support requires the `effdet` package. "
            "Install with: pip install effdet timm"
        ) from exc

    model_name = _normalize_efficientdet_arch(arch)
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    if image_size is not None:
        config.image_size = _resolve_image_size(image_size)

    weights_path = None
    if isinstance(weights, (str, Path)):
        candidate = Path(weights)
        if candidate.exists():
            weights_path = candidate

    pretrained = weights is not None and weights_path is None
    train_model = create_model(
        model_name,
        bench_task="train",
        pretrained=pretrained,
        num_classes=num_classes,
        image_size=config.image_size,
    )

    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "model_state_dict", "model_state"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
        if isinstance(state_dict, dict):
            cleaned = {}
            for key, value in state_dict.items():
                cleaned_key = key
                for prefix in ("model.", "module.", "net.", "student."):
                    if cleaned_key.startswith(prefix):
                        cleaned_key = cleaned_key[len(prefix):]
                        break
                cleaned[cleaned_key] = value
            state_dict = cleaned

        target_model = train_model.model if hasattr(train_model, "model") else train_model
        target_model.load_state_dict(state_dict, strict=False)

    if hasattr(train_model, "model"):
        base_model = train_model.model
        try:
            predict_model = DetBenchPredict(base_model, config)
        except TypeError:
            predict_model = DetBenchPredict(base_model)
    else:
        predict_model = create_model(
            model_name,
            bench_task="predict",
            pretrained=False,
            num_classes=num_classes,
            image_size=config.image_size,
        )
        predict_model.load_state_dict(train_model.state_dict(), strict=False)

    return EfficientDetWrapper(
        train_model,
        predict_model,
        config,
        label_offset=label_offset,
        box_format=box_format,
    )


def _effective_num_classes(arch: str, class_num: int, add_background: bool) -> int:
    if _is_efficientdet_arch(arch):
        return class_num
    return class_num + 1 if add_background else class_num


class DetectionStudent(nn.Module):
    def __init__(self, student, class_num: int) -> None:
        super().__init__()
        add_background = getattr(student, "add_background", True)
        weights = getattr(student, "weights", None)
        weights_backbone = getattr(student, "weights_backbone", None)
        self.label_offset = int(getattr(student, "label_offset", 1))
        image_size = getattr(student, "image_size", None)
        box_format = getattr(student, "box_format", None)

        if _is_efficientdet_arch(student.arch):
            weights = _normalize_weights_value(weights)
            weights_backbone = None
        else:
            weights, weights_backbone = _resolve_weights(student.arch, weights, weights_backbone)

        num_classes = _effective_num_classes(student.arch, class_num, add_background)

        self.model = _build_detection_model(
            arch=student.arch,
            num_classes=num_classes,
            weights=weights,
            weights_backbone=weights_backbone,
            label_offset=self.label_offset,
            image_size=image_size,
            box_format=box_format,
        )
        self.num_features = self._infer_num_features()

    def _infer_num_features(self) -> int:
        if hasattr(self.model, "extract_features"):
            size = getattr(self.model, "input_size", None)
            if size is None:
                size = (320, 320)
            elif isinstance(size, int):
                size = (size, size)
            dummy = torch.zeros(3, size[0], size[1])
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                features = self.model.extract_features([dummy])
            if was_training:
                self.model.train()
            return int(features.shape[1])

        dummy = torch.zeros(3, 320, 320)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            images, _ = self.model.transform([dummy], None)
            features = self.model.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                feature_map = features
            else:
                feature_map = list(features.values())[-1]
            num_features = feature_map.shape[1]
        if was_training:
            self.model.train()
        return num_features

    def extract_features(self, images):
        if hasattr(self.model, "extract_features"):
            return self.model.extract_features(images)

        images, _ = self.model.transform(images, None)
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            feature_map = features
        else:
            feature_map = list(features.values())[-1]
        pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        return feature_norm(pooled)

    def forward(self, images, targets=None):
        return self.model(images, targets)


class DetectionTeacherStudent(nn.Module):
    def __init__(self, teacher, student, data_attributes, use_teacher: bool = True) -> None:
        super().__init__()
        self.teacher = None
        self.align = None
        self.frozen_nlp_features = None
        self.data_attributes = data_attributes
        self.student = DetectionStudent(student, data_attributes.class_num)

        if use_teacher:
            self.teacher = TeacherNet(teacher)
            self.align = AlignNet(self.teacher.last_features_dim, self.student.num_features)
            self.frozen_nlp_features = self._get_frozen_nlp_features(data_attributes)

    def _get_frozen_nlp_features(self, attributes):
        prompt_tmpl = attributes.prompt_tmpl
        classes = getattr(attributes, "classes", {})
        if isinstance(classes, (dict, DictConfig)):
            def _key_as_int(item):
                try:
                    return int(item[0])
                except (TypeError, ValueError):
                    return 0

            class_names = [name for _, name in sorted(classes.items(), key=_key_as_int)]
        else:
            class_names = list(classes)
        text_tokens = self.teacher.tokenizer([prompt_tmpl.format(word) for word in class_names])
        nlp_features = self.teacher.encode_text(text_tokens).detach()
        return feature_norm(nlp_features)

    def forward(self, images, targets=None, teacher_images=None, return_kd: bool = False):
        outputs = self.student(images, targets)
        if self.teacher and return_kd:
            if teacher_images is None:
                raise ValueError("teacher_images must be provided when return_kd is True.")
            clip_img_features = self.teacher(teacher_images)
            frozen_nlp_features = self.frozen_nlp_features.to(clip_img_features.device)
            aligned_img, aligned_nlp = self.align(clip_img_features, frozen_nlp_features)
            student_features = self.student.extract_features(images)
            kd_inputs = (
                student_features,
                None,
                clip_img_features,
                frozen_nlp_features,
                aligned_img,
                aligned_nlp,
            )
            return outputs, kd_inputs
        return outputs


class DetectionTeacher(nn.Module):
    def __init__(
        self,
        arch: str,
        num_classes: int,
        weights: Optional[str] = None,
        weights_backbone: Optional[str] = None,
        add_background: bool = True,
        ckpt_path: Optional[str] = None,
        label_offset: int = 1,
        image_size: Optional[object] = None,
        box_format: Optional[str] = None,
    ) -> None:
        super().__init__()
        label_offset = int(label_offset)
        if _is_efficientdet_arch(arch):
            weights = _normalize_weights_value(weights)
            weights_backbone = None
        else:
            weights, weights_backbone = _resolve_weights(arch, weights, weights_backbone)
        num_classes = _effective_num_classes(arch, num_classes, add_background)
        self.model = _build_detection_model(
            arch=arch,
            num_classes=num_classes,
            weights=weights,
            weights_backbone=weights_backbone,
            label_offset=label_offset,
            image_size=image_size,
            box_format=box_format,
        )

        if ckpt_path:
            try:
                from src.utils.utils import allowlist_checkpoint_globals

                allowlist_checkpoint_globals(ckpt_path)
            except Exception:
                pass

            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            cleaned = {}
            for key, value in state_dict.items():
                cleaned_key = key
                stripped = True
                while stripped:
                    stripped = False
                    for prefix in ("model.", "net.", "student.", "module."):
                        if cleaned_key.startswith(prefix):
                            cleaned_key = cleaned_key[len(prefix):]
                            stripped = True
                            break
                cleaned[cleaned_key] = value
            _load_state_dict_filtered(self.model, cleaned)

        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, images):
        with torch.no_grad():
            return self.model(images)
