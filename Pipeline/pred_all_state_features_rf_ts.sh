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



# srun python create_features_filter_model_time_series.py --STATE Lagos --path_to_savergb final_runs/data/Lagos/Lagos_inter_all_geocoords_wpred.shp --DATA_PATH /lustre/davis/FishPonds_project/share/final_runs/data

# srun python create_features_filter_model_time_series.py --STATE Oyo --path_to_savergb final_runs/data/Oyo/Oyo_inter_all_geocoords_wpred.shp --DATA_PATH /lustre/davis/FishPonds_project/share/final_runs/data


srun python create_features_filter_model_time_series.py --STATE Niger --path_to_savergb final_runs/data/Niger/Niger_inter_all_geocoords_wpred.shp --DATA_PATH /lustre/davis/FishPonds_project/share/final_runs/data

