# import extract_raster_xyz_google_noqgis_single_Copy1
import create_grid

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
import subprocess


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
    choices = ['Abia', 'Adamawa', 'AkwaIbom', 'Bauchi', 'Bayelsa', 'Benue', 'Borno', 'CrossRiver', 'Ebonyi', 'Edo', 'Ekiti',
              'Enugu', 'FederalCapitalTerritory', 'Gombe', 'Imo', 'Jigawa', 'Kaduna', 'Katsina', 'Kebbi', 'Kogi', 'Nasarawa',
              'Ondo', 'Osun', 'Plateau', 'Sokoto', 'Taraba', 'Yobe', 'Zamfara'],
    metavar='state',
    help='Define state where you want to extract imgs ',
    type=str)

# pred_files.add_argument(
#     'GRID_NUM',
#     metavar='grid_num',
#     help='Number of grid you want to extract data, check how many the state has',
#     type=int)

pred_files.add_argument(
    'TOTAL_PRED',
    metavar='total_pred',
    help='Total of predictions on the state',
    type=int)


ins = parser.parse_args()

STATE = ins.STATE
# GRID_NUM = ins.GRID_NUM
TOTAL_PRED = ins.TOTAL_PRED

DATA_PATH = '/lustre/davis/FishPonds_project/share/data/'


if not os.path.exists(os.path.join(DATA_PATH, f"{STATE}")):
    os.makedirs(os.path.join(DATA_PATH, f"{STATE}"))


grid = create_grid.create_grid(DATA_PATH, STATE, load=True)
np.random.seed(434)
# print(grid)
grid_random_index = np.arange(len(grid))
np.random.shuffle(grid_random_index)
# print(list(grid_random_index))
del grid



# logger = mp.log_to_stderr()
# logger.setLevel(mp.SUBDEBUG)


def work(STATE, GRID_NUM, shared_len, lock):
    command = ['python', 'extract_raster_xyz_google_noqgis_single.py', f'{STATE}', f'{GRID_NUM}', 'runs']
    subprocess.call(command)
    with lock:
        if len(glob.glob(os.path.join(DATA_PATH, f"{STATE}/{GRID_NUM}/geocoords/*_geocoords.shp"))):
            shape_predictions = gpd.read_file(glob.glob(os.path.join(DATA_PATH, f"{STATE}/{GRID_NUM}/geocoords/*_geocoords.shp"))[0])
            shared_len.value += len(shape_predictions)
            if shared_len.value >= TOTAL_PRED:
                annot =  glob.glob(os.path.join(DATA_PATH, f"{STATE}/*/geocoords/*_geocoords.shp"))
                dest = gpd.GeoDataFrame()
                for shape_file in annot:
                    pred_grid = gpd.read_file(shape_file)
                    dest = pd.concat([dest, pred_grid])
                dest = dest.reset_index(drop=True)
                dest.to_file(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))
                return None  # Stop processing if target is reached
        return shared_len.value

def worker(args):
    """Unpacks arguments for multiprocessing."""
    return work(*args)
    
def generate_table(STATE, GRID_NUMs, num_workers):
    # Number of parallel workers


    with mp.Manager() as manager:
        #signal.signal(signal.SIGUSR1, handler)
        #signal.signal(signal.SIGTERM, handler)
        #signal.signal(signal.SIGINT, handler)
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
            print(f"Processing complete, predictions needed found. Total predictions generated: {results} and total pred expected {TOTAL_PRED}")
        else:
            annot = glob.glob(os.path.join(DATA_PATH, f"{STATE}/*/geocoords/*_geocoords.shp"))
            if len(annot):
                dest = gpd.GeoDataFrame()
                for an in annot:
                    datas = gpd.read_file(an)
                    dest = pd.concat([dest, datas])
                dest = dest.reset_index(drop=True)
                dest.to_file(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))
                print(f"Processing complete. Total predictions generated: {results}, {dest.shape} and total pred expected {TOTAL_PRED}")
            else:
                dest = gpd.GeoDataFrame({'ids':[0], 'geometry': ['nothing here']})
                dest.to_file(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))
                print(f"NO PREDICTIONS, NOTHING TO SAVE")

        # print(f"Processing complete, predictions needed Total predictions generated: {results} and total pred expected {TOTAL_PRED}")


    

    # with mp.Manager() as manager:
    #     pool =  mp.Pool(processes=num_workers)
    #     shared_len = manager.Value('i', 0)  # Shared variable to track length
    #     lock = manager.Lock()  # Lock for thread safety
    #     #event = manager.Event()
        
    #     results = []
    #     tasks = [i for i in grid_random_index]  # Predefine task arguments

    #     for i in tasks:
    #         # rr = pool.apply_async(work, (STATE, i, shared_len, lock))
    #         rr = pool.apply_async(work, (STATE, i, shared_len, lock))
    #         # print("partial result")
    #         # print(i, rr.get())
    #         if rr.get() is None:
    #             print("Breaking", flush=True)
    #             #pool.terminate()
    #             break
    #         else:# Stop when the target length is reached
    #             results.append(rr)
    #     pool.close()
    #     pool.join()
    #     r = [x.get() for x in results]
    #     if len(glob.glob(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))):
    #         print(f"Processing complete, predictions needed Total predictions generated: {r} and total pred expected {TOTAL_PRED}")
    #     else:
    #         annot = glob.glob(os.path.join(DATA_PATH, f"{STATE}/*/geocoords/*_geocoords.shp"))
    #         if len(annot):
    #             dest = gpd.GeoDataFrame()
    #             for an in annot:
    #                 datas = gpd.read_file(an)
    #                 dest = pd.concat([dest, datas])
    #             dest = dest.reset_index(drop=True)
    #             dest.to_file(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))
    #             print(f"Processing complete. Total predictions generated: {r}, {dest.shape} and total pred expected {TOTAL_PRED}")
    #         else:
    #             dest = gpd.GeoDataFrame({'ids':[0], 'geometry': ['nothing here']})
    #             dest.to_file(os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp"))
    #             print(f"NO PREDICTIONS, NOTHING TO SAVE")
    
    #     # print("Processing complete. Total tables generated:", results[-1])


#logger = mp.log_to_stderr()
#logger.setLevel(mp.SUBDEBUG)
def main():

    num_workers = 8  # Number of parallel workers
    generate_table(STATE, grid_random_index, num_workers)


if __name__ == "__main__":
    main()
    print(f"Total time = {time.time() - t_init}")
