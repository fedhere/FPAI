from qgis.core import *
from osgeo import gdal
import geopandas as gpd
import rasterio
import rasterio.transform
from rasterio.crs import CRS
from qgis import processing
import cv2
from shapely.geometry import Polygon, box
import numpy as np
import os
import sys
import glob
from shapely.validation import make_valid
from PIL import Image
import utils_mask_to_pixels
import create_grid
import save_predictions
import subprocess


import argparse
from argparse import FileType, ArgumentParser

DATA_PATH = '/lustre/davis/FishPonds_project/share/final_runs/data/'#users/2221/FishPonds/data/'
# MODEL_NAME = 'fine_tune_ogun_delta_combine_freeze_v2' #'epochs1000'
# MODEL_NAME = 'from_zero_ogun_delta_combine_background_v3_v0'
MODEL_NAME = 'fine_tune_final_dataset_freeze_v02_2'



parser = ArgumentParser()
pred_files = parser.add_argument_group()

pred_files.add_argument(
    'STATE',#make list of the 37 nigeria states
    choices = ['Anambra', 'Rivers', 'Abia', 'Adamawa', 'AkwaIbom', 'Bauchi', 'Bayelsa', 'Benue', 'Borno', 'CrossRiver', 'Ebonyi', 'Edo', 'Ekiti',
              'Enugu', 'FederalCapitalTerritory', 'Gombe', 'Imo', 'Jigawa', 'Kaduna', 'Katsina', 'Kebbi', 'Kogi', 'Nasarawa',
              'Ondo', 'Osun', 'Plateau', 'Sokoto', 'Taraba', 'Yobe', 'Zamfara'],
    metavar='state',
    help='Define state where you want to extract imgs ',
    type=str)

pred_files.add_argument(
    'GRID_NUM',
    metavar='grid_num',
    help='Number of grid you want to extract data, check how many the state has',
    type=int)

pred_files.add_argument(
    'RUNS',
    metavar='runs_name',
    help='Name of run folder',
    type=str)

ins = parser.parse_args()

STATE = ins.STATE
GRID_NUM = ins.GRID_NUM
RUNS = ins.RUNS


print('HERE')
# cc = 0
# # def run(DATA_PATH, RUNS, STATE, GRID_NUM):
# if os.path.exists(os.path.join(DATA_PATH, f"{STATE}")):
#     if len(glob.glob(os.path.join(DATA_PATH, f"{STATE}/{GRID_NUM}/geocoords/*_geocoords.shp"))) == 1:
#         print(f'{STATE} and grid number {GRID_NUM} already complete, check shape file in\n{glob.glob(os.path.join(DATA_PATH, f"{STATE}/{GRID_NUM}/geocoords/*_geocoords.shp"))}')
#         cc = cc+1
#         # sys.exit()
# elif os.path.exists(os.path.join(DATA_PATH, f"{STATE}_2")):
#     if len(glob.glob(os.path.join(DATA_PATH, f"{STATE}_2/{GRID_NUM}/geocoords/*_geocoords.shp"))) == 1:
#         print(f'{STATE}_2 and grid number {GRID_NUM} already complete, check shape file in\n{glob.glob(os.path.join(DATA_PATH, f"{STATE}_2/{GRID_NUM}/geocoords/*_geocoords.shp"))}')
#         cc = cc+1
# if cc > 0:
#     sys.exit()

if len(glob.glob(os.path.join(DATA_PATH, f"{STATE}/{GRID_NUM}/geocoords/*_geocoords.shp"))) == 1:
    print(f'{STATE} and grid number {GRID_NUM} already complete, check shape file in\n{glob.glob(os.path.join(DATA_PATH, f"{STATE}/{GRID_NUM}/geocoords/*_geocoords.shp"))}')
    sys.exit()


print('RUNNING')
create_grid.create_grid('/lustre/davis/FishPonds_project/share/data', STATE, GRID_NUM)

# Load QGIS to extract Google XYZ Tiles
os.environ["QT_QPA_PLATFORM"] = "offscreen"

qgs = QgsApplication([], False)
# Load providers
qgs.initQgis()

urls = 'type=xyz&url=http://www.google.cn/maps/vt?lyrs%3Ds@189%26gl%3Dcn%26x%3D%7Bx%7D%26y%3D%7By%7D%26z%3D%7Bz%7D&zmax=19&zmin=0&http-header:referer='

rlayer = QgsRasterLayer(urls, 'GoogleSatellite', 'wms')  

renderer = rlayer.renderer()
provider = rlayer.dataProvider()
#provider
crs = rlayer.crs().toWkt()
pipe = QgsRasterPipe()
pipe.set(provider.clone())
pipe.set(renderer.clone())

layer = QgsVectorLayer(os.path.join("/lustre/davis/FishPonds_project/share/data", f"{STATE}/grid_{STATE}.shp"))
feats = [ feat for feat in layer.getFeatures() ]
print(f"The {STATE} state is divded into {len(feats)} squares of 6km x 6km")
# print(f"Extracting 100x100 images each of 2,000x2,000 pixels from {GRID_NUM} grid...")
print(f"Extracting 30x30 images each of 2,000x2,000 pixels from the number {GRID_NUM} grid...")

width = 2000
height = 2000
#GRID_NUM = 0


if os.path.exists(os.path.join(DATA_PATH, f"{STATE}/{GRID_NUM}/")):
    print(f'{STATE} and {GRID_NUM} grid already exists')
    # sys.exit()
elif os.path.exists(os.path.join(DATA_PATH, f"{STATE}/{GRID_NUM}/geocoords/")):
    print(f'{STATE} and {GRID_NUM} predictions geocoords already exists')
    #sys.exit()
else:
    # create directory to save tif files of grid
    os.makedirs(os.path.join(DATA_PATH, f"{STATE}/{GRID_NUM}"))
# create directory to save geo transformations of each tif
loc_save_transform = os.path.join(DATA_PATH, f"{STATE}/{GRID_NUM}/transformation")
if not os.path.exists(loc_save_transform):
    os.makedirs(loc_save_transform)
save_loc_coord = os.path.join(DATA_PATH, f"{STATE}/{GRID_NUM}/geocoords")
if not os.path.exists(save_loc_coord):
    os.makedirs(save_loc_coord)

k = GRID_NUM
extent_big = feats[k].geometry().boundingBox()
overlap = 400
xmax = extent_big.xMaximum() + 2*(overlap*0.1)
# xmax = extent_big.xMaximum()
h = extent_big.height()
w = extent_big.width()

iter_w = int(round((w/0.1)/width,0))
iter_h = int(round((h/0.1)/height,0))

add_w = (overlap*(iter_w))/(width-overlap)
add_h = (overlap*(iter_h))/(width-overlap)
if add_w < 1:
    add_w = np.ceil(add_w)
if add_h < 1:
    add_h = np.ceil(add_h)
iter_w = iter_w + int(add_w) + 1
iter_h = iter_h + int(add_h) + 1 
print("INTER W AND INTER H")
print(iter_w, iter_h)
# pixel size is 0.1m
sy = width * 0.1 #h / iter_h
sx = height * 0.1 #w / iter_w
#print(sy, sx, xmax)
xmin = xmax - sx
print(f"EXTRACTING TIF FILES FROM XYZ GOOGLE TILES OF {STATE} AND GRID NUM {GRID_NUM}")
for i in range(iter_w):
    # ymax = extent_big.yMaximum()
    ymax = extent_big.yMaximum() + 2*(overlap*0.1)#extent_big.yMaximum()
    ymin = ymax - sy
    for j in range(iter_h):
        # if not os.path.exists(f"/lustre/davis/FishPonds_project/share/{RUNS}/test_all_{STATE}_{GRID_NUM}/{STATE}_g{k}c{i}r{j}"):
        extent = QgsRectangle(xmin, ymin, xmax, ymax)
        # location to save raster files
        #file_writer = QgsRasterFileWriter(f'/data/correct_data/delta_11/delta_{k}{i}{j}.tif')
        file_writer = QgsRasterFileWriter(os.path.join(DATA_PATH, f'{STATE}/{GRID_NUM}/{STATE}_g{k}c{i}r{j}.tif'))
        file_writer.writeRaster(pipe,
                            width,
                            height,
                            extent,
                            rlayer.crs())
        img = os.path.join(DATA_PATH, f'{STATE}/{GRID_NUM}/{STATE}_g{k}c{i}r{j}.tif')
        with rasterio.open(img) as src:
            raster_transform = src.transform
            # print(raster_transform)
    
        with open(os.path.join(loc_save_transform, f"{STATE}_g{k}c{i}r{j}_transform.txt"), "w") as f:
            f.write(str(raster_transform[0:])[1:-1] + '\n')
            f.write(str(src.crs))
        
        filename_new = img.replace('.tif', '.png')
        with Image.open(img) as tif:
            tif.save(filename_new)
        os.remove(img)

        if os.path.exists(f"/lustre/davis/FishPonds_project/share/{RUNS}/test_all_{STATE}_{GRID_NUM}/{STATE}_g{k}c{i}r{j}"):
            command = ['rm', '-r', f"/lustre/davis/FishPonds_project/share/{RUNS}/test_all_{STATE}_{GRID_NUM}/{STATE}_g{k}c{i}r{j}"]
            subprocess.call(command)

        comand = (f"python $WORKDIR/FishPonds_project/share/YOLO/yolov7/seg/segment/predict.py \
        --imgsz 1600 \
        --source {os.path.join(DATA_PATH, f'{STATE}/{GRID_NUM}/{STATE}_g{k}c{i}r{j}.png')} \
        --weights $WORKDIR/FishPonds_project/share/YOLO/yolov7/seg/runs/train-seg/{MODEL_NAME}/weights/best.pt \
        --save-txt \
        --save-conf \
        --iou-thres 0.5 \
        --conf-thres 0.5 \
        --project $WORKDIR/FishPonds_project/share/{RUNS}/test_all_{STATE}_{GRID_NUM}/ \
        --name {STATE}_g{k}c{i}r{j}")
        os.system(comand)
        os.remove(os.path.join(DATA_PATH, f'{STATE}/{GRID_NUM}/{STATE}_g{k}c{i}r{j}.png'))


        save_predictions.save_predictions(RUNS, STATE, GRID_NUM, k, i, j)
      
        ymax = ymin + (overlap*0.1)
        ymin = ymax - sy
    # xmax = xmin
    xmax = xmin + (overlap*0.1)
    xmin = xmax - sx
# del layer # CLEANING UP HERE
# app.exitQgis
qgs.exit()
# read all predictions and save shape file
print(f'PREDICTIONS COMPLETE')

predictions = glob.glob(f"/lustre/davis/FishPonds_project/share/{RUNS}/test_all_{STATE}_{GRID_NUM}/*/labels/*.npz")
if len(predictions):
    name_files = [("_").join(x.split("/")[-1].split(".")[0].split("_")[1:]) for x in predictions]
    # print(name_files)
    path_to_transform = glob.glob(os.path.join(loc_save_transform, "*.txt"))
    # print(path_to_transform)
    
    
    gdf = utils_mask_to_pixels.pixel_to_coord(name_files, predictions, path_to_transform, f"{STATE}_{GRID_NUM}", save_loc_coord=save_loc_coord, split='all', class_cls='P', save=True)
    for i in predictions:
        os.remove(i)
    print(f"test_all_{STATE}_{GRID_NUM} predictions saved as shapefile")
else:
    dest = gpd.GeoDataFrame({'ids':[0], 'geometry': ['nothing here']})
    dest.to_file(os.path.join(save_loc_coord, f"{STATE}_{GRID_NUM}_{P}_geocoords.shp"))
os.system(f"rm -r {loc_save_transform}")
# return# True
        
    # print(f"DONE for {STATE} GRID NUMBER {GRID_NUM}")
