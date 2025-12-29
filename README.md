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

This repository contains the **official implementation** of the paper:
"VL2Lite: Task-Specific Knowledge Distillation from Large Vision-Language Models to Lightweight Networks".

This repository provides a **PyTorch Lightning + Hydra** codebase for distilling knowledge from large Vision-Language Models (VLMs) (e.g., CLIP-like models) to smaller, resource-efficient neural networks. By leveraging multi-modal embeddings (visual & textual) from a **frozen** VLM, VL2Lite significantly boosts performance in fine-grained classification tasks without extra teacher fine-tuning overhead.

---

## Features

- **Frozen VLM Teacher**: No teacher fine-tuning required  
- **Condensation Layers**: Reduce dimensionality for both image and text embeddings  
- **Multi-Loss**: Classification loss + Visual KD + Linguistic KD  
- **Dynamic Weighting**: Gradually shifts from KD to classification emphasis  
- **Configurable**: Hydra-based setup for custom data, model, or experiment scripts

---

## Installation

1. **Clone project**:
   ```bash
   git clone https://github.com/jsjangAI/VL2Lite
   cd vl2lite
   ```

2. *(Optional)* **Create conda environment**:
   ```bash
   conda create -n vl2lite_env python=3.9
   conda activate vl2lite_env
   ```

3. **Install PyTorch** per [official instructions](https://pytorch.org/get-started/).

4. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Setup

If your dataset is in a different path, create a soft link:
```bash
ln -s ./data/kd_datasets /data/KD_datasets
```
Update `configs/data/` if needed.

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

Example run:
```bash
python src/train.py data=detection_yolo model=kd_detection callbacks=detection trainer=gpu
```
Override `data.attributes.*` or set `data.attributes.data_yaml` to use a YOLO YAML file.

EfficientDet students are supported via the optional `effdet` + `timm` packages:
```bash
pip install effdet timm
```
Example (D0/D1):
```bash
python src/train.py data=detection_yolo model=kd_detection callbacks=detection trainer=gpu \
  model.net.student.arch=efficientdet_d0 model.net.student.image_size=512
```
```bash
python src/train.py data=detection_yolo model=kd_detection callbacks=detection trainer=gpu \
  model.net.student.arch=efficientdet_d1 model.net.student.image_size=640
```

Train a detection teacher first (no KD):
```bash
python src/train.py data=detection_yolo model=detection_teacher callbacks=detection trainer=gpu logger=tensorboard
```
Then distill with the teacher checkpoint:
```bash
python src/train.py data=detection_yolo model=kd_detection callbacks=detection trainer=gpu logger=tensorboard \
  model.det_teacher.ckpt_path=logs/train/runs/<run>/checkpoints/epoch_XXX.ckpt
```
You can also run `bash scripts/train_detection_teacher.sh` for a simple teacher baseline.

---

## How to Run

We use [PyTorch Lightning](https://www.pytorchlightning.ai/) for training loops and [Hydra](https://hydra.cc/) for configuration.

### Basic Commands

- **Train on CPU**:
  ```bash
  python src/train.py trainer=cpu
  ```

- **Train on GPU**:
  ```bash
  python src/train.py trainer=gpu
  ```

### Experiment Configs

Pick a config from `configs/experiment/`:
```bash
python src/train.py experiment=experiment_name
```
and you can override any parameter:
```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
*(See `src/train.sh` for an example script.)*

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
