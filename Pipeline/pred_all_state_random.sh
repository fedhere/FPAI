#!/bin/bash
#SBATCH --partition=idle
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=5:00:00
#SBATCH --output=cpu_random_pred_yolov7_%x-%j.out

vpkg_require anaconda/2024.02
source activate base
conda activate /lustre/davis/sw/FishPonds/yolov7_qgis_2025/20250124
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
srun python $*
# sbatch -J STATE pred_all_state_random.sh predict_random_grid.py STATE 5000

# srun python ../../YOLO/yolov7/seg/segment/predict.py \
#     --imgsz 1600 \
#     --source $WORKDIR/users/2221/FishPonds/data/ogun_translate/crop/test \
#     --weights $WORKDIR/users/2221/FishPonds/YOLO/yolov7/seg/runs/train-seg/epochs1000/weights/best.pt \
#     --save-txt \
#     --save-conf \
#     --iou-thres 0.5 \
#     --conf-thres 0.5 \
#     --name ogun_pred
