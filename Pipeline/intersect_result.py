# from pylabel import importer
import glob
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from PIL import Image
import pandas as pd
import yaml
import os
from shapely.geometry import box, Polygon, MultiPolygon, Point

from shapely.validation import make_valid
import sys
# from matplotlib import patches
import geopandas as gpd
import rasterio
import rasterio.transform
# from rasterio.plot import show

# import utils_pixel_to_geo
# import json

DATA_PATH = '/lustre/davis/FishPonds_project/share/final_runs/data'
# STATE = 'Abia'

def intersect_results(DATA_PATH='', STATE='', path_to_results='', path_to_save=''):
    """
        Intersect overlapping predictions 
    """
    if path_to_results == '':
        path_to_results = os.path.join(DATA_PATH, f"{STATE}/{STATE}_all_geocoords.shp")
    gdf_file = gpd.read_file(path_to_results)
    gdf_file["geometry"] = gdf_file["geometry"].apply(lambda x: make_valid(x))
    for six, ss in gdf_file.iterrows():
        if ss['geometry'].geom_type != 'Polygon':
            # print(ss.ids)
            # print(len(ss['geometry'].geoms))
            gdf_file.loc[six,'geometry'] = ss['geometry'].geoms[0]
    joined_df = gpd.sjoin(gdf_file, gdf_file, how="left", predicate="intersects").reset_index(drop=True)
    joined_df_r = gpd.sjoin(gdf_file, gdf_file, how="left", predicate="intersects").reset_index(drop=True)
    
    
    intersection = pd.DataFrame()
    ids_intersection_t = []
    ids_intersection = []
    joined_df['flag'] = np.zeros(len(joined_df))
    
    for index, row in joined_df.iterrows():  
        # if row['ids_right'] != row['ids_left']:
        if pd.isnull(row['ids_right']):
            ids_intersection.append(row['ids_left'])
        else:
            t_ogun_poly = gdf_file[gdf_file['ids'] == row['ids_right']]['geometry'].item()
            area_t = t_ogun_poly.area
            ogun_poly = gdf_file[gdf_file['ids'] == row['ids_left']]['geometry'].item()
            area = ogun_poly.area
            polygon_intersection = t_ogun_poly.intersection(ogun_poly).area
            polygon_union = t_ogun_poly.union(ogun_poly).area
            # iou_values.append(polygon_intersection / polygon_union)
            ious = polygon_intersection / polygon_union
            joined_df.loc[index, 'iou'] = ious
            if 0.001 <= ious <= 0.01:
                ids_intersection.append(row['ids_left'])
                ids_intersection_t.append(row['ids_right'])
                joined_df.loc[index, 'flag'] = 1
        if len(joined_df[joined_df['ids_left'] == row['ids_left']]) > 1:
            if row['ids_right'] == row['ids_left']:
                joined_df.loc[index, 'flag'] = 2
            
    ids_intersection = list(set(ids_intersection))
    ids_intersection_t = list(set(ids_intersection_t))
    joined_df_s = joined_df[joined_df['flag'] != 2]
    iou_df = joined_df_s.loc[joined_df_s.dropna().groupby('ids_left')['iou'].idxmax()].sort_index()
    
    for index, row in iou_df.iterrows():
        if row['ids_right'] == row['ids_left']:
            # print("same")
            ids_intersection.append(row['ids_right'])
            ids_intersection = list(set(ids_intersection))
            ids_intersection_t = list(set(ids_intersection_t))
        else:
            if row['iou'] <= 0.1:
                ids_intersection.append(row['ids_left'])
                ids_intersection_t.append(row['ids_right'])
                ids_intersection = list(set(ids_intersection))
                ids_intersection_t = list(set(ids_intersection_t))
            else:
                t_ogun_poly = gdf_file[gdf_file['ids'] == row['ids_right']]['geometry'].item()
                area_t = t_ogun_poly.area
                ogun_poly = gdf_file[gdf_file['ids'] == row['ids_left']]['geometry'].item()
                area = ogun_poly.area
                if area_t >= area:
                    ids_intersection_t.append(row['ids_right'])
                    ids_intersection = list(set(ids_intersection))
                    ids_intersection_t = list(set(ids_intersection_t))
                    if row['ids_left'] in ids_intersection:
                        # print(row['ids_left'])
                        ids_intersection.remove(row['ids_left'])
                    if row['ids_left'] in ids_intersection_t:
                        # print(row['ids_left'])
                        ids_intersection_t.remove(row['ids_left'])
                else:
                    ids_intersection.append(row['ids_left'])
                    ids_intersection = list(set(ids_intersection))
                    ids_intersection_t = list(set(ids_intersection_t))
                    if row['ids_right'] in ids_intersection_t:
                        # print(row['ids_right'])
                        ids_intersection_t.remove(row['ids_right'])
                    if row['ids_right'] in ids_intersection:
                        # print(row['ids_right'])
                        ids_intersection.remove(row['ids_right'])
    
    inter_ogun = gdf_file[gdf_file['ids'].isin(ids_intersection)]
    inter_togun = gdf_file[gdf_file['ids'].isin(ids_intersection_t)]

    dest = pd.concat([inter_ogun, inter_togun]).drop_duplicates().reset_index(drop=True)
    # dest = dest.reset_index(drop=True)

    if path_to_save == '':
        path_to_save = os.path.join(DATA_PATH, f"{STATE}/{STATE}_inter_all_geocoords.shp")
    dest.to_file(path_to_save)

    return path_to_save #dest

