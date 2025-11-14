# import extract_raster_xyz_google_noqgis_single_Copy1
import create_grid
import intersect_result
import create_features_filter_model
import apply_RF_model
import argparse
from argparse import FileType, ArgumentParser

import numpy as np
import geopandas as gpd
import pandas as pd
import os
import glob
import multiprocessing as mp
from queue import Queue
from threading import Thread

import logging
import time
# import signal
import json
import subprocess
import shutil



t_init = time.time()

parser = ArgumentParser()
pred_files = parser.add_argument_group()

pred_files.add_argument(
    'STATE',#make list of the 37 nigeria states
    choices = ['Abia', 'Adamawa', 'AkwaIbom', 'Anambra', 'Bauchi', 'Bayelsa',
       'Benue', 'Borno', 'CrossRiver', 'Delta', 'Ebonyi', 'Edo', 'Ekiti',
       'Enugu', 'FederalCapitalTerritory', 'Gombe', 'Imo', 'Jigawa',
       'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Kogi', 'Kwara', 'Lagos',
       'Nasarawa', 'Niger', 'Ogun', 'Ondo', 'Osun', 'Oyo', 'Plateau',
       'Rivers', 'Sokoto', 'Taraba', 'Yobe', 'Zamfara'],
    metavar='state',
    help='Define state where you want to extract imgs ',
    type=str)

# pred_files.add_argument(
#     'GRID_NUM',
#     metavar='grid_num',
#     help='Number of grid you want to extract data, check how many the state has',
#     type=int)

# pred_files.add_argument(
#     'TOTAL_PRED',
#     metavar='total_pred',
#     help='Total of predictions on the state',
#     type=int)


pred_files.add_argument(
    '--continue_grid_search',
    dest='continue_grid_search',
    help='True flag if continue with grid search',
    default=False,
    action=argparse.BooleanOptionalAction)



ins = parser.parse_args()

STATE = ins.STATE
# GRID_NUM = ins.GRID_NUM
# TOTAL_PRED = ins.TOTAL_PRED

DATA_PATH = '/lustre/davis/FishPonds_project/share/final_runs/data'
ROOT_NIGERIA_DATA='/lustre/davis/FishPonds_project/share/data/'

ROOT = '/lustre/davis/FishPonds_project/share/'


if not os.path.exists(os.path.join(DATA_PATH, f"{STATE}")):
    os.makedirs(os.path.join(DATA_PATH, f"{STATE}"))
    


grid = create_grid.create_grid(ROOT_NIGERIA_DATA, STATE, load=True)
buffer = gpd.read_file("/lustre/davis/FishPonds_project/share/data/Nigeria/building_buffer/buffer_7_5km_dissolved.shp").to_crs('EPSG:3857')
grid['area_inter'] = grid.apply(lambda row: row['geometry'].intersection(buffer.loc[0, 'geometry']).area, axis=1)
grid_state_buffer = grid[grid['area_inter'] >= 200*200*90*5]
# .index.values
np.random.seed(434)
# print(grid)
# grid_random_index = grid_state_buffer.index.values.copy()[::3] #np.arange(len(grid))
if len(grid_state_buffer.index.values.copy()[::3]) <= 100:
    grid_random_index = grid_state_buffer.index.values.copy()
else:
    grid_random_index = grid_state_buffer.index.values.copy()[::3]
# np.random.shuffle(grid_random_index)
print(f"{STATE} HAS {len(grid_random_index)} ELEMENTS ON GRID")


def process_predictions(DATA_PATH, STATE, path_to_results='', path_to_save=''): # this has to be another job, to activate conda env conda_env_name = '/lustre/davis/sw/FishPonds/conda_qgis_2025/20250606/'
    path_to_save_final = os.path.join(DATA_PATH, f"{STATE}/{STATE}_inter_all_geocoords_wfeatures.shp")
    if os.path.exists(path_to_save_final):
        print('ALL DONE')
    else:
        print("Inspect overlapping, creating data set")
        path_to_intersection = f'/lustre/davis/FishPonds_project/share/final_runs/data/{STATE}/{STATE}_inter_all_geocoords.shp'
        if os.path.exists(path_to_intersection):
            print('intrsection done')
        else:
            path_to_intersection = intersect_result.intersect_results(DATA_PATH, STATE, path_to_results='', path_to_save='')
        path_to_savergb = path_to_intersection.replace(".shp", "_wrgb.shp")
        if os.path.exists(path_to_savergb):
            print('rgb done')
        else:
            print("Extract Features")
            print("Extract RGB")
            command = ['python', 'extract_rgb.py', f'{STATE}', f'{path_to_intersection}'] # this one doesnt work as module
            subprocess.call(command)
            path_to_savergb = path_to_intersection.replace(".shp", "_wrgb.shp")
        dest_state_intersection_wfeatures_path = os.path.join(DATA_PATH, f"{STATE}/temp_all_inter_all_geocoords_wfeatures.shp")
        if os.path.exists(dest_state_intersection_wfeatures_path):
            print('indices done')
        else:
            print("Extract indices")
            # conda_env_name = '/lustre/davis/sw/FishPonds/conda_qgis_2025/20250606/'
            # command = f"conda run -p {conda_env_name} python create_features_filter_model.py --STATE {STATE} --path_to_savergb {path_to_savergb} --DATA_PATH {DATA_PATH} --ROOT_NIGERIA_DATA {ROOT_NIGERIA_DATA}"
            # subprocess.run(command)
            create_features_filter_model.extract_indices(DATA_PATH, path_to_savergb, STATE, ROOT_NIGERIA_DATA)
            create_features_filter_model.add_state_indices(DATA_PATH, STATE, dest_state_intersection_wfeatures_path)
        print("Check preds are within polygon")
        path_to_save_final = os.path.join(DATA_PATH, f"{STATE}/{STATE}_inter_all_geocoords_wfeatures.shp")
        if os.path.exists(path_to_save_final):
            print('ALL done')
        else:
            create_features_filter_model.add_state_indices(DATA_PATH, STATE, dest_state_intersection_wfeatures_path)
            create_features_filter_model.check_preds_within_state(DATA_PATH, STATE, dest_state_intersection_wfeatures_path)
    
    # [os.remove(x) for x in glob.glob(os.path.join(DATA_PATH, f"{STATE}/temp_{STATE}_inter_all_geocoords_wfeatures.*"))]

                
def main():
    process_predictions(DATA_PATH, STATE, path_to_results='', path_to_save='')
    seg_predictions_path = os.path.join(DATA_PATH, f"{STATE}/{STATE}_inter_all_geocoords_wfeatures.shp")
    apply_RF_model.apply_rf(seg_predictions_path=seg_predictions_path, use_state=None, path_to_rf_model='')


if __name__ == "__main__":
    main()
    print(f"Total time = {time.time() - t_init}")
