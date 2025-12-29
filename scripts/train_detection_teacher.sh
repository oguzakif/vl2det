#!/bin/bash
# Train a detection teacher without VLM/detection KD.
# Run from repo root: bash scripts/train_detection_teacher.sh

python src/train.py data=detection_yolo model=detection_teacher callbacks=detection trainer=gpu logger=tensorboard
