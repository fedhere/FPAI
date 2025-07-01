# import extract_raster_xyz_google_noqgis_single_Copy1
import create_grid
import intersect_result
import create_features_filter_model

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


# def handler(signum, frame):
#     print('Signal handler called with signal', signum)
#     # with open('DIA_model_statistics_gpu_'+name+'.pkl', 'wb') as f:
#     annot = glob.glob(os.path.join(DATA_PATH, f"{STATE}/*/geocoords/*_geocoords.shp"))
#     if len(annot):
#         dest = gpd.GeoDataFrame()
#         for shape_file in annot:
#             pred_grid = gpd.read_file(shape_file)
#             dest = pd.concat([dest, pred_grid])
#         dest = dest.reset_index(drop=True)
#         dest.to_file(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))
#     exit(0)





t_init = time.time()

parser = ArgumentParser()
pred_files = parser.add_argument_group()

pred_files.add_argument(
    'STATE',#make list of the 37 nigeria states
    # choices = ['Anambra', 'Rivers', 'Abia', 'Adamawa', 'AkwaIbom', 'Bauchi', 'Bayelsa', 'Benue', 'Borno', 'CrossRiver', 'Ebonyi', 'Edo', 'Ekiti',
    #           'Enugu', 'FederalCapitalTerritory', 'Gombe', 'Imo', 'Jigawa', 'Kaduna', 'Katsina', 'Kebbi', 'Kogi', 'Nasarawa',
    #           'Ondo', 'Osun', 'Plateau', 'Sokoto', 'Taraba', 'Yobe', 'Zamfara', 'Ogun', 'Delta', ],
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

# if os.access(os.path.join(DATA_PATH, f"{STATE}"), os.W_OK):
#     STATE = STATE
# else:
#     STATE = STATE+'_2'
# if not os.path.exists(os.path.join(DATA_PATH, f"{STATE}")):
#     os.makedirs(os.path.join(DATA_PATH, f"{STATE}"))

if not os.path.exists(os.path.join(DATA_PATH, f"{STATE}")):
    os.makedirs(os.path.join(DATA_PATH, f"{STATE}"))
    


grid = create_grid.create_grid('/lustre/davis/FishPonds_project/share/data', STATE, load=True)
buffer = gpd.read_file("/lustre/davis/FishPonds_project/share/data/Nigeria/building_buffer/buffer_7_5km_dissolved.shp").to_crs('EPSG:3857')
grid['area_inter'] = grid.apply(lambda row: row['geometry'].intersection(buffer.loc[0, 'geometry']).area, axis=1)
grid_state_buffer = grid[grid['area_inter'] >= 200*200*90*3]
# .index.values
np.random.seed(434)
# print(grid)
grid_random_index = grid_state_buffer.index.values.copy()[:4] #np.arange(len(grid))
# np.random.shuffle(grid_random_index)

try:
    with open('/lustre/davis/FishPonds_project/share/data/Nigeria/grids_ran.json', 'r') as json_file:
        grids_ran = json.load(json_file)
    grid_random_index = [x for x in grid_random_index if x not in grids_ran[STATE][0]]
except KeyError as e:
    print(f"KeyError: The key '{e}' was not found in the dictionary.")
    pass 
# print(list(grid_random_index))
del grid


def work(STATE, GRID_NUM, shared_len, lock):
    command = ['python', 'extract_raster_xyz_google_noqgis_single.py', f'{STATE}', f'{GRID_NUM}', 'final_runs/runs']
    subprocess.call(command)
    with lock:
        if len(glob.glob(os.path.join(DATA_PATH, f"{STATE}/{GRID_NUM}/geocoords/*_geocoords.shp"))):
            shape_predictions = gpd.read_file(glob.glob(os.path.join(DATA_PATH, f"{STATE}/{GRID_NUM}/geocoords/*_geocoords.shp"))[0])
            shared_len.value += len(shape_predictions)
            # if shared_len.value >= TOTAL_PRED:
            #     annot =  glob.glob(os.path.join(DATA_PATH, f"{STATE}/*/geocoords/*_geocoords.shp"))
            #     dest = gpd.GeoDataFrame()
            #     for shape_file in annot:
            #         pred_grid = gpd.read_file(shape_file)
            #         dest = pd.concat([dest, pred_grid])
            #     dest = dest.reset_index(drop=True)
            #     dest.to_file(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))
            #     return None  # Stop processing if target is reached
            
        return shared_len.value

def worker(args):
    """Unpacks arguments for multiprocessing."""
    return work(*args)


def process_predictions(DATA_PATH, STATE, path_to_results='', path_to_save=''):
    print("Inspect overlapping, creating data set")
    path_to_intersection = intersect_result.intersect_results(DATA_PATH, STATE, path_to_results='', path_to_save='')
    print("Extract Features")
    print("Extract RGB")
    dest_state_intersection = create_features_filter_model.extract_rgb(path_to_intersection)
    print("Extract indices")
    dest_state_intersection_wfeatures, state_geometry = create_features_filter_model.extract_indices(dest_state_intersection, STATE)
    print("Check preds are within polygon")
    create_features_filter_model.check_preds_within_state(DATA_PATH, STATE, dest_state_intersection_wfeatures, state_geometry)


def func_manager(STATE, grid_random_index, num_workers):
    with mp.Manager() as manager:
        shared_len = manager.Value('i', 0)  # Shared variable to track length
        lock = manager.Lock()  # Lock for thread safety
    
        with mp.Pool(processes=num_workers) as pool:
            results = []
            tasks = [(STATE, i, shared_len, lock) for i in grid_random_index]  # Predefine task arguments
    
            for result in pool.imap_unordered(worker, tasks):
                if result is None:
                    print("Breaking", flush=True)
                    break  # Stop when the target length is reached
                else:# Stop when the target length is reached
                    results.append(result)


    if len(glob.glob(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))):
        # print(f"Processing complete, predictions needed found. Total predictions generated: {results} and total pred expected {TOTAL_PRED}")
        print(f"Processing complete, predictions needed found. Total predictions generated: {results}")
        process_predictions(DATA_PATH, STATE, path_to_results='', path_to_save='')
        
    else:
        annot = glob.glob(os.path.join(DATA_PATH, f"{STATE}/*/geocoords/*_geocoords.shp"))
        if len(annot):
            dest = gpd.GeoDataFrame()
            for an in annot:
                datas = gpd.read_file(an)
                dest = pd.concat([dest, datas])
            dest = dest.reset_index(drop=True)
            dest.to_file(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))
            # print(f"Processing complete. Total predictions generated: {results}, {dest.shape} and total pred expected {TOTAL_PRED}")
            print(f"Processing complete. Total predictions generated: {results}, {dest.shape}")           
            process_predictions(DATA_PATH, STATE, path_to_results='', path_to_save='')
            
        else:
            dest = gpd.GeoDataFrame({'ids':[0], 'geometry': ['nothing here']})
            dest.to_file(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))
            print(f"NO PREDICTIONS, NOTHING TO SAVE")
    
    
def generate_table(STATE, grid_random_index, num_workers):
    # Number of parallel workers
    if ins.continue_grid_search == False:

        func_manager(STATE, grid_random_index, num_workers)
        
        # annot = glob.glob(os.path.join(DATA_PATH, f"{STATE}/*/geocoords/*_geocoords.shp"))
        # if len(annot):
        #     print(f"Geocoords per grid already exist, creating large shape file")
        #     dest = gpd.GeoDataFrame()
        #     for an in annot:
        #         datas = gpd.read_file(an)
        #         dest = pd.concat([dest, datas])
        #     dest = dest.reset_index(drop=True)
        #     dest.to_file(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))
        #     print(f"Processing complete. Total predictions generated: {results}, {dest.shape} and total pred expected {TOTAL_PRED}")
        #     process_predictions(DATA_PATH, STATE, path_to_results='', path_to_save='')
        # elif len(glob.glob(os.path.join(DATA_PATH, f"{STATE}/*"))) >= 17:
        #     print(f"Geocoords shape file per grid doesn't exist, but state already ran once")
        #     dest = gpd.GeoDataFrame({'ids':[0], 'geometry': ['nothing here']})
        #     dest.to_file(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))
        #     print(f"NO PREDICTIONS, NOTHING TO SAVE")
        # else:
        #     # try:
        #     #     print("Deleting old data")
        #     #     # command_rmstate = ['rm', '-r', f'/lustre/davis/FishPonds_project/share/data/{STATE}']
        #     #     # subprocess.call(command_rmstate)
        #     #     for rruns in glob.glob(f'/lustre/davis/FishPonds_project/share/runs/test_all_{STATE}*'):
        #     #         shutil.rmtree(rruns)
        #     # except:
        #     #     print('Nothing to delete')
        #     func_manager(STATE, grid_random_index, num_workers)
    
    else:
        annot = glob.glob(os.path.join(DATA_PATH, f"{STATE}/*/geocoords/*_geocoords.shp"))
        name_shp = [x.split('/')[-1] for x in annot]
        grids_ran = [int(''.join(filter(str.isdigit, x.split('/')[-1]))) for x in name_shp]
        grid_random_index_n = [x for x in grid_random_index if x not in grids_ran]
        # grid_random_index = grid_state_buffer.index.values.copy()[:2]
        if len(grids_ran) == len(grid_random_index):
            annot = glob.glob(os.path.join(DATA_PATH, f"{STATE}/*/geocoords/*_geocoords.shp"))
            if len(annot):
                dest = gpd.GeoDataFrame()
                for an in annot:
                    datas = gpd.read_file(an)
                    dest = pd.concat([dest, datas])
                dest = dest.reset_index(drop=True)
                dest.to_file(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))
                print(f"Processing complete. Total predictions generated: {results}, {dest.shape}")           
                process_predictions(DATA_PATH, STATE, path_to_results='', path_to_save='')
        else:
            func_manager(STATE, grid_random_index_n, num_workers)
                
def main():

    num_workers = 2  # Number of parallel workers
    generate_table(STATE, grid_random_index, num_workers)


if __name__ == "__main__":
    main()
    print(f"Total time = {time.time() - t_init}")
