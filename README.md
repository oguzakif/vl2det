______________________________________________________________________

<div align="center">

# VL2Lite: Task-Specific Knowledge Distillation from Large Vision-Language Models to Lightweight Networks

<a href="https://pytorch.org/get-started/locally/">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">
</a>
<a href="https://pytorchlightning.ai/">
  <img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white">
</a>
<a href="https://hydra.cc/">
  <img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd">
</a>
<a href="https://github.com/ashleve/lightning-hydra-template">
  <img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray">
</a>
<br>
[![Conference](http://img.shields.io/badge/CVPR%202025-Paper-4b44ce.svg)](#)

</div>

---

## Description

This repository contains the **VL2Lite codebase** plus a detection-focused adaptation used in this project.
It keeps the frozen VLM distillation recipe, but swaps the classifier for torchvision detectors and adds a YOLO-format
detection pipeline with optional detector-teacher KD.

---

## Features

- **Frozen VLM Teacher**: No teacher fine-tuning required  
- **Detection Pipeline**: YOLO-format dataset to torchvision detectors with background label offset  
- **VLM KD for Detection**: Pooled detector features aligned to VLM image/text embeddings  
- **Optional Detector-Teacher KD**: Box/score matching distillation term  
- **Configurable**: Hydra-based setup for custom data, model, or experiment scripts

---

## Detection Adaptation (What Changed)

- **Data**: `src/data/yolo_detection_dataset.py` converts YOLO labels to torchvision targets and can return
  CLIP-normalized images for the VLM teacher; `data.label_offset=1` reserves background label 0.
- **Model**: `src/models/components/detection.py` wraps torchvision detectors, pools backbone features for KD,
  and aligns them with VLM image/text embeddings.
- **Training**: `src/models/detection_kd_module.py` mixes detection loss with VLM KD and an optional detector-teacher KD term.
- **Configs**: `configs/data/detection_yolo.yaml`, `configs/model/kd_detection.yaml`, and `configs/model/detection_teacher.yaml`.

## Installation

1. **Go to the project folder**:
   ```bash
   cd CMP722-ACV/VL2Lite
   ```

2. *(Optional)* **Create conda environment**:
   ```bash
   conda env create -f environment.yaml
   conda activate myenv
   ```

3. **Install PyTorch** per [official instructions](https://pytorch.org/get-started/).

4. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

5. *(Optional)* **Editable install** (enables `train_command` / `eval_command` entry points):
   ```bash
   pip install -e .
   ```

---

## Data Setup

By default, datasets live under `data/kd_datasets/` (see `configs/paths/default.yaml`).
For a custom YOLO dataset, either update `configs/data/attributes/detection_example.yaml` or override on the command line:
```bash
python src/train.py data=detection_yolo \
  data.attributes.data_yaml=/absolute/path/to/data.yaml
```
You can also override folder names:
```bash
python src/train.py data=detection_yolo \
  data.attributes.images_train=images/train data.attributes.labels_train=labels/train
```

---

## Detection (YOLO)

For YOLO-style detection datasets, point the config to your images and labels or a YOLO data YAML file.

Expected structure (default):
```
dataset_root/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Label format (per line): `class_id x_center y_center width height` (normalized).
By default, labels are offset by +1 for background (`data.label_offset=1`).

Override `data.attributes.*` or set `data.attributes.data_yaml` to use a YOLO YAML file.

### Train: VLM KD only (no detector teacher)
```bash
python src/train.py data=detection_yolo model=kd_detection callbacks=detection trainer=gpu logger=tensorboard \
  model.use_det_teacher=false
```

### Train: detector-teacher baseline (no KD)
```bash
python src/train.py data=detection_yolo model=detection_teacher callbacks=detection trainer=gpu logger=tensorboard
```
You can also run `bash scripts/train_detection_teacher.sh` for a simple teacher baseline.

### Train: VLM KD + detector teacher
```bash
python src/train.py data=detection_yolo model=kd_detection callbacks=detection trainer=gpu logger=tensorboard \
  model.use_det_teacher=true \
  model.det_teacher.ckpt_path=logs/train/runs/<run>/checkpoints/epoch_XXX.ckpt
```

### Evaluate a checkpoint
```bash
python src/eval.py data=detection_yolo model=kd_detection ckpt_path=/absolute/path/to/ckpt.ckpt
```

### Optional: EfficientDet students
EfficientDet requires `effdet` and `timm` (included in `requirements.txt`).
```bash
python src/train.py data=detection_yolo model=kd_detection callbacks=detection trainer=gpu \
  model.net.student.arch=efficientdet_d0 model.net.student.image_size=512
```
```bash
python src/train.py data=detection_yolo model=kd_detection callbacks=detection trainer=gpu \
  model.net.student.arch=efficientdet_d1 model.net.student.image_size=640
```

### Plot TensorBoard scalars
```bash
python scripts/plot_tensorboard_scalars.py logs/train/runs/<run1> logs/train/runs/<run2> \
  --legends run1 run2 --out-dir plots/tensorboard
```

---

## Hydra Notes

We use [PyTorch Lightning](https://www.pytorchlightning.ai/) for training loops and [Hydra](https://hydra.cc/) for configuration.
You can override any config from the CLI, for example:
```bash
python src/train.py data=detection_yolo model=kd_detection trainer=gpu \
  data.batch_size=16 trainer.max_epochs=50
```
Pick a config from `configs/experiment/`:
```bash
python src/train.py experiment=experiment_name
```

---

## Configuration

- **trainer** configs in `configs/trainer/`  
- **data** configs in `configs/data/`  
- **model** configs in `configs/model/`  
- **experiment** configs in `configs/experiment/`  

Hydra allows combining or overriding these configs easily.

---

## Acknowledgments

Built upon the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).  
We thank open-source projects (PyTorch, Lightning, Hydra) that enable this work.

---

## Citation

If you use this code or find VL2Lite helpful, please cite:

```bibtex
@misc{jang2025vl2lite,
  title={VL2Lite: Task-Specific Knowledge Distillation from Large Vision-Language Models to Lightweight Networks},
  author={Jang, Jinseong and Ma, Chunfei and Lee, Byeongwon},
  journal={CVPR},
  year={2025}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
Please see the [LICENSE](LICENSE) file for details.

```  
