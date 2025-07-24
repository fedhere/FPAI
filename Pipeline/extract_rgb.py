from qgis.core import *
from osgeo import gdal
from qgis import processing
import geopandas as gpd
import pandas as pd
import numpy as np
import os
from PIL import Image

import argparse
from argparse import FileType, ArgumentParser

parser = ArgumentParser()
pred_files = parser.add_argument_group()

pred_files.add_argument(
    'STATE',
    metavar='STATE',
    help='STATE',
    type=str)

pred_files.add_argument(
    'path_to_intersection',
    metavar='path_to_intersection',
    help='Path of file after intersection',
    type=str)




ins = parser.parse_args()

path_to_intersection = ins.path_to_intersection
STATE = ins.STATE

state_intersection = gpd.read_file(path_to_intersection)

dest_state_intersection = pd.concat([state_intersection, state_intersection.bounds, state_intersection.area, state_intersection.length], axis=1)
dest_state_intersection.rename(columns={0: 'area', 1: 'length'}, inplace=True)

dest_state_intersection = dest_state_intersection[dest_state_intersection['area'] > 0]
dest_state_intersection.loc[:, 'inv_isoperimetric_ratio'] = dest_state_intersection.loc[:, 'area']/(dest_state_intersection.loc[:, 'length']**2)


loc_save_data = "/lustre/davis/FishPonds_project/share/final_runs/data"

dest_state_intersection["ave_r"] = np.zeros(len(dest_state_intersection))
dest_state_intersection["ave_g"] = np.zeros(len(dest_state_intersection))
dest_state_intersection["ave_b"] = np.zeros(len(dest_state_intersection))

dest_state_intersection["std_r"] = np.zeros(len(dest_state_intersection))
dest_state_intersection["std_g"] = np.zeros(len(dest_state_intersection))
dest_state_intersection["std_b"] = np.zeros(len(dest_state_intersection))


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


for k in dest_state_intersection.iterfeatures():
    xmin, ymin, xmax, ymax = k['properties']['minx'], k['properties']['miny'], k['properties']['maxx'], k['properties']['maxy']
    
    # h = extent_big.height()
    h = ymax - ymin
    w = xmax - xmin
    iter_w = int(round((w/0.1)/w,0))
    iter_h = int(round((h/0.1)/h,0))
    # pixel size is 0.1m
    sy = w * 0.1 #h / iter_h
    sx = h * 0.1 #w / iter_w

    if k["properties"]["ids"] == 'Sokoto_g81c3r17_1':
        print(xmin, ymin, xmax, ymax, int(w/0.1), int(h/0.1))
    extent = QgsRectangle(xmin, ymin, xmax, ymax)
    # location to save raster files
    file_writer = QgsRasterFileWriter(os.path.join(loc_save_data, f'{k["properties"]["ids"]}.tif'))
    file_writer.writeRaster(pipe,
                            int(round(w/0.1,0)),
                            int(round(h/0.1,0)),
                        extent,
                        rlayer.crs())
    img = os.path.join(loc_save_data, f'{k["properties"]["ids"]}.tif')
    
    filename_new = img.replace('.tif', '.png')
    with Image.open(img) as tif:
        tif.save(filename_new)
    os.remove(img)
    with Image.open(filename_new) as png:
        png = png.convert("RGB")  # Ensure the image is in RGB format
        pixels = np.array(list(png.getdata()))
    os.remove(filename_new)
    
    r, g, b = np.mean(pixels, axis=0)
    r_std, g_std, b_std = np.std(pixels, axis=0)

    dest_state_intersection.loc[int(k['id']), 'ave_r'] = r.item()
    dest_state_intersection.loc[int(k['id']), 'ave_g'] = g.item()
    dest_state_intersection.loc[int(k['id']), 'ave_b'] = b.item()

    dest_state_intersection.loc[int(k['id']), 'std_r'] = r_std.item()
    dest_state_intersection.loc[int(k['id']), 'std_g'] = g_std.item()
    dest_state_intersection.loc[int(k['id']), 'std_b'] = b_std.item()
# QgsApplication.exitQgis()
# path_to_savergb = os.path.join(path_to_intersection.split(f"{STATE}/")[0], f"{STATE}/dest_state_intersection_wrgb.shp")
path_to_savergb = path_to_intersection.replace(".shp", "_wrgb.shp")
dest_state_intersection.to_file(path_to_savergb)
qgs.exit()
