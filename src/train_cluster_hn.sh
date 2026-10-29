#!/bin/bash
#SBATCH --job-name=raven-train-hn
#SBATCH --partition=gpuidle
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ryl77lak/AI_MODEL/logs_cluster/%x_%j.log

cd /scratch/ryl77lak/AI_MODEL
module load miniforge3/latest

# Same recipe as train_cluster.sh; only the data (hard negatives added) changes.
yolo detect train \
  model=yolo11n.pt \
  data=configs/bird_detection_hn_cluster.yaml \
  epochs=50 \
  batch=64 \
  imgsz=640 \
  device=0 \
  workers=16 \
  cache=True \
  seed=0 \
  deterministic=True \
  project=runs \
  name=bird_yolo11n_hn_50ep
