#!/bin/bash
#SBATCH --partition=standard
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-0
#SBATCH --output=feat_rf_pred_%x-%j.out
#SBATCH --error=real_error_feat_rf_pred_%x-%j.log

vpkg_require anaconda/2024.02
source activate base
conda activate /lustre/davis/sw/FishPonds/conda_qgis_2025/20250606/
export XDG_RUNTIME_DIR="/tmp/runtime-taceroc"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}"

#echo "this is a test job names" $SLURM_JOB_NAME
## WEIGHT_PATH = $WORKDIR/yolov7/seg/yolov7-seg.pt
srun python $*
# sbatch -J Bayelsa pred_all_state_features_rf.sh extract_all_features_rf.py Bayelsa



# srun python ../../YOLO/yolov7/seg/segment/predict.py \
#     --imgsz 1600 \
#     --source $WORKDIR/users/2221/FishPonds/data/ogun_translate/crop/test \
#     --weights $WORKDIR/users/2221/FishPonds/YOLO/yolov7/seg/runs/train-seg/epochs1000/weights/best.pt \
#     --save-txt \
#     --save-conf \
#     --iou-thres 0.5 \
#     --conf-thres 0.5 \
#     --name ogun_pred
