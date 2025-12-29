
from typing import Any, Dict, Optional, Tuple

import re
import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from omegaconf import DictConfig, ListConfig
from torchmetrics import MeanMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


class DetectionKDModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        use_teacher: bool,
        kd_criterion,
        det_teacher: Optional[torch.nn.Module] = None,
        use_det_teacher: bool = False,
        warmup_epochs: int = 5,
        det_ramp_epochs: int = 30,
        kd_weight_scale: float = 1.0,
        det_kd_decay_epochs: int = 30,
        det_kd_weight_scale: float = 1.0,
        det_kd_box_weight: float = 1.0,
        det_kd_score_weight: float = 1.0,
        det_kd_iou_thresh: float = 0.5,
        det_kd_score_thresh: float = 0.05,
        det_kd_topk: int = 100,
        det_kd_match_labels: bool = True,
        class_metrics: bool = False,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "kd_criterion", "det_teacher"])

        self.net = net
        self.use_teacher = use_teacher
        if use_teacher:
            self.kd_criterion = kd_criterion

        self.det_teacher = det_teacher
        self.use_det_teacher = bool(use_det_teacher and det_teacher is not None)
        if self.det_teacher is not None:
            self.det_teacher.eval()
            self.det_teacher.requires_grad_(False)

        self.train_loss = MeanMetric()
        self.train_det_loss = MeanMetric()
        if use_teacher:
            self.train_img_loss = MeanMetric()
            self.train_kd_loss = MeanMetric()
        if self.use_det_teacher:
            self.train_det_kd_loss = MeanMetric()

        self.class_metrics = bool(class_metrics)
        self.val_map = MeanAveragePrecision(class_metrics=self.class_metrics)
        self.test_map = MeanAveragePrecision(class_metrics=self.class_metrics)
        self.val_map_50 = None
        self.test_map_50 = None
        if self.class_metrics:
            self.val_map_50 = MeanAveragePrecision(
                iou_thresholds=[0.5], class_metrics=True
            )
            self.test_map_50 = MeanAveragePrecision(
                iou_thresholds=[0.5], class_metrics=True
            )

    @staticmethod
    def _filter_loggable_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        loggable: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    loggable[key] = value
                continue
            if isinstance(value, (float, int)):
                loggable[key] = value
        return loggable

    def forward(self, images, targets=None, teacher_images=None, return_kd: bool = False):
        return self.net(images, targets=targets, teacher_images=teacher_images, return_kd=return_kd)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if len(batch) == 4:
            images, targets, teacher_images, det_teacher_images = batch
        else:
            images, targets, teacher_images = batch
            det_teacher_images = None
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        if teacher_images is not None:
            teacher_images = teacher_images.to(device)
        if det_teacher_images is not None:
            det_teacher_images = [img.to(device) for img in det_teacher_images]
        if len(batch) == 4:
            return images, targets, teacher_images, det_teacher_images
        return images, targets, teacher_images

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.train_det_loss.reset()
        if self.use_teacher:
            self.train_img_loss.reset()
            self.train_kd_loss.reset()
        if self.use_det_teacher:
            self.train_det_kd_loss.reset()

    def _calculate_loss_weights(self, current_epoch: int) -> Tuple[float, float]:
        warmup = max(0, int(self.hparams.warmup_epochs))
        ramp = max(1, int(self.hparams.det_ramp_epochs))

        if current_epoch < warmup:
            return 1.0, 0.0

        progress = (current_epoch - warmup + 1) / ramp
        progress = float(np.clip(progress, 0.0, 1.0))
        kd_weight = float((1.0 - progress) * float(self.hparams.kd_weight_scale))
        return 1.0, kd_weight

    def _calculate_det_kd_weight(self, current_epoch: int) -> float:
        decay_epochs = max(1, int(self.hparams.det_kd_decay_epochs))
        progress = float(current_epoch) / float(decay_epochs)
        weight = float(np.clip(1.0 - progress, 0.0, 1.0))
        return weight * float(self.hparams.det_kd_weight_scale)

    def _student_predict(self, images):
        student_model = None
        if hasattr(self.net, "student") and hasattr(self.net.student, "model"):
            student_model = self.net.student.model
        elif hasattr(self.net, "model"):
            student_model = self.net.model
        else:
            student_model = self.net

        was_training = student_model.training
        student_model.eval()
        outputs = student_model(images)
        if was_training:
            student_model.train()
        return outputs

    def _compute_det_kd_loss(self, images, student_outputs, teacher_outputs) -> torch.Tensor:
        losses = []
        for image, student_out, teacher_out in zip(images, student_outputs, teacher_outputs):
            t_boxes = teacher_out.get("boxes")
            s_boxes = student_out.get("boxes")
            if t_boxes is None or s_boxes is None or t_boxes.numel() == 0 or s_boxes.numel() == 0:
                continue

            t_scores = teacher_out.get("scores")
            t_labels = teacher_out.get("labels")

            if t_scores is not None and self.hparams.det_kd_score_thresh > 0:
                keep = t_scores >= float(self.hparams.det_kd_score_thresh)
                t_boxes = t_boxes[keep]
                t_scores = t_scores[keep]
                if t_labels is not None:
                    t_labels = t_labels[keep]

            if t_scores is not None and self.hparams.det_kd_topk > 0 and t_scores.numel() > self.hparams.det_kd_topk:
                topk = int(self.hparams.det_kd_topk)
                values, idx = torch.topk(t_scores, topk)
                t_scores = values
                t_boxes = t_boxes[idx]
                if t_labels is not None:
                    t_labels = t_labels[idx]

            if t_boxes.numel() == 0:
                continue

            ious = box_iou(t_boxes, s_boxes)
            max_ious, max_idx = ious.max(dim=1)
            mask = max_ious >= float(self.hparams.det_kd_iou_thresh)
            if mask.sum() == 0:
                continue

            if self.hparams.det_kd_match_labels and t_labels is not None and "labels" in student_out:
                s_labels = student_out["labels"][max_idx]
                mask = mask & (t_labels == s_labels)
                if mask.sum() == 0:
                    continue

            matched_idx = max_idx[mask]
            t_boxes = t_boxes[mask]
            s_boxes = s_boxes[matched_idx]

            _, height, width = image.shape
            scale = torch.tensor([width, height, width, height], device=image.device, dtype=s_boxes.dtype)
            t_norm = t_boxes / scale
            s_norm = s_boxes / scale

            box_loss = F.l1_loss(s_norm, t_norm, reduction="none").mean()
            loss = box_loss * float(self.hparams.det_kd_box_weight)

            if (
                self.hparams.det_kd_score_weight > 0
                and t_scores is not None
                and "scores" in student_out
            ):
                s_scores = student_out["scores"][matched_idx]
                t_scores_sel = t_scores[mask]
                score_loss = F.mse_loss(s_scores, t_scores_sel)
                loss = loss + score_loss * float(self.hparams.det_kd_score_weight)

            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=images[0].device)
        return torch.stack(losses).mean()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        if len(batch) == 4:
            images, targets, teacher_images, det_teacher_images = batch
        else:
            images, targets, teacher_images = batch
            det_teacher_images = None
        batch_size = len(images)
        if self.use_teacher:
            loss_dict, kd_inputs = self.forward(
                images, targets=targets, teacher_images=teacher_images, return_kd=True
            )
        else:
            loss_dict = self.forward(images, targets=targets)

        det_loss = sum(loss_dict.values())
        det_weight, kd_weight = self._calculate_loss_weights(self.current_epoch)
        total_loss = det_weight * det_loss

        if self.use_teacher:
            img_loss, kd_loss = self.kd_criterion(kd_inputs)
            total_loss = total_loss + kd_weight * (img_loss + kd_loss) / 2

            self.train_img_loss(img_loss)
            self.train_kd_loss(kd_loss)
            self.log(
                "train/img_loss",
                self.train_img_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            self.log(
                "train/kd_loss",
                self.train_kd_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        if self.use_det_teacher:
            student_outputs = self._student_predict(images)
            det_teacher_inputs = det_teacher_images if det_teacher_images is not None else images
            teacher_outputs = self.det_teacher(det_teacher_inputs)
            det_kd_loss = self._compute_det_kd_loss(images, student_outputs, teacher_outputs)
            det_kd_weight = self._calculate_det_kd_weight(self.current_epoch)
            total_loss = total_loss + det_kd_weight * det_kd_loss

            self.train_det_kd_loss(det_kd_loss)
            self.log(
                "train/det_kd_loss",
                self.train_det_kd_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        self.train_loss(total_loss)
        self.train_det_loss(det_loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/det_loss",
            self.train_det_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for key, value in loss_dict.items():
            if key in {"loss", "det_loss", "total_loss"}:
                continue
            self.log(
                f"train/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        return total_loss

    def validation_step(self, batch, batch_idx: int) -> None:
        if len(batch) == 4:
            images, targets, _, _ = batch
        else:
            images, targets, _ = batch
        outputs = self.forward(images, targets=None, return_kd=False)
        self.val_map.update(outputs, targets)
        if self.val_map_50 is not None:
            self.val_map_50.update(outputs, targets)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_map.compute()
        self._log_per_class_metrics(metrics, prefix="val", metric_label="map50_95")
        if self.val_map_50 is not None:
            metrics_50 = self.val_map_50.compute()
            self._log_per_class_metrics(metrics_50, prefix="val", metric_label="map50")
        loggable = self._filter_loggable_metrics(metrics)
        if "map" in loggable:
            loggable["map_50_95"] = loggable["map"]
        self.log_dict({f"val/{k}": v for k, v in loggable.items()}, prog_bar=True, sync_dist=True)
        self.val_map.reset()
        if self.val_map_50 is not None:
            self.val_map_50.reset()

    def test_step(self, batch, batch_idx: int) -> None:
        if len(batch) == 4:
            images, targets, _, _ = batch
        else:
            images, targets, _ = batch
        outputs = self.forward(images, targets=None, return_kd=False)
        self.test_map.update(outputs, targets)
        if self.test_map_50 is not None:
            self.test_map_50.update(outputs, targets)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_map.compute()
        self._log_per_class_metrics(metrics, prefix="test", metric_label="map50_95")
        if self.test_map_50 is not None:
            metrics_50 = self.test_map_50.compute()
            self._log_per_class_metrics(metrics_50, prefix="test", metric_label="map50")
        loggable = self._filter_loggable_metrics(metrics)
        if "map" in loggable:
            loggable["map_50_95"] = loggable["map"]
        self.log_dict({f"test/{k}": v for k, v in loggable.items()}, prog_bar=True, sync_dist=True)
        self.test_map.reset()
        if self.test_map_50 is not None:
            self.test_map_50.reset()

    def _log_per_class_metrics(
        self, metrics: Dict[str, Any], prefix: str, metric_label: str
    ) -> None:
        per_class = metrics.get("map_per_class")
        if per_class is None:
            return
        if isinstance(per_class, torch.Tensor):
            if per_class.ndim == 0:
                return
            values = per_class.flatten().tolist()
        else:
            try:
                values = list(per_class)
            except TypeError:
                return

        class_ids = metrics.get("classes")
        if isinstance(class_ids, torch.Tensor):
            if class_ids.ndim == 0:
                class_ids = None
            else:
                class_ids = class_ids.flatten().tolist()
        elif class_ids is not None:
            try:
                class_ids = list(class_ids)
            except TypeError:
                class_ids = None

        names = self._resolve_class_names()
        label_offset = self._get_label_offset()
        for idx, val in enumerate(values):
            class_id = class_ids[idx] if class_ids and idx < len(class_ids) else idx
            name = self._class_name_from_id(class_id, names, label_offset)
            self.log(
                f"{prefix}/{metric_label}_{name}",
                float(val),
                prog_bar=False,
                sync_dist=True,
            )

    def _resolve_class_names(self) -> list[str]:
        attributes = getattr(self.net, "data_attributes", None)
        if attributes is None and self.trainer is not None:
            datamodule = getattr(self.trainer, "datamodule", None)
            attributes = getattr(datamodule, "data_train", None)
            attributes = getattr(attributes, "attributes", None)
        if attributes is None:
            return []
        classes = getattr(attributes, "classes", None)
        if classes is None:
            return []
        if isinstance(classes, (dict, DictConfig)):
            def _key_as_int(item: Tuple[object, object]) -> int:
                try:
                    return int(item[0])
                except (TypeError, ValueError):
                    return 0

            return [name for _, name in sorted(classes.items(), key=_key_as_int)]
        if isinstance(classes, (list, ListConfig)):
            return list(classes)
        return []

    def _get_label_offset(self) -> int:
        if hasattr(self.net, "student") and hasattr(self.net.student, "label_offset"):
            try:
                return int(self.net.student.label_offset)
            except (TypeError, ValueError):
                return 0
        return 0

    def _class_name_from_id(
        self, class_id: object, class_names: list[str], label_offset: int
    ) -> str:
        try:
            class_id_int = int(class_id)
        except (TypeError, ValueError):
            class_id_int = None
        name = None
        if class_id_int is not None and class_names:
            idx = class_id_int - label_offset
            if 0 <= idx < len(class_names):
                name = class_names[idx]
            elif 0 <= class_id_int < len(class_names):
                name = class_names[class_id_int]
        if not name:
            name = f"class_{class_id_int}" if class_id_int is not None else "class_unknown"
        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", name)
        return sanitized.strip("_") or "class"

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        if len(batch) == 4:
            images, targets, _, _ = batch
        else:
            images, targets, _ = batch
        outputs = self.forward(images, targets=None, return_kd=False)
        return {"outputs": outputs, "targets": targets}

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/map",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
