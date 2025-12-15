import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.transform
from rasterio.crs import CRS

import shapely

# from qgis import processing
import cv2
from PIL import Image
from shapely.geometry import Polygon, box
import numpy as np
import os
import sys
import glob
from shapely.validation import make_valid
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN



def find_partial_overlaps(DATA_PATH, path_dest_state_w_pred, STATE, min_overlap_ratio=0.5, keep_largest=True,):
    """
    Find polygons that are mostly within another polygon
    but may have edges outside
    
    Parameters:
    - min_overlap_ratio: minimum proportion of polygon that must be within
    """
    gdf = gpd.read_file(path_dest_state_w_pred)
    gdf = gdf.drop_duplicates(subset='geometry').reset_index(drop=True)
    gdf_sindex = gdf.sindex
    partial_containment = []
    to_remove = set()
    gdf['original_index'] = gdf['ids']
    
    for idx, poly in gdf.iterrows():
        possible_matches_index = list(gdf_sindex.intersection(poly.geometry.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        
        for match_idx, match_poly in possible_matches.iterrows():
            if match_idx == idx:
                continue
            
            # Check for intersection
            if match_poly.geometry.intersects(poly.geometry):
                # Calculate intersection area
                intersection = match_poly.geometry.intersection(poly.geometry)
                
                if not intersection.is_empty:
                    # Calculate what percentage of match_poly is within poly
                    overlap_ratio1 = intersection.area / poly.geometry.area
                    overlap_ratio2 = intersection.area / match_poly.geometry.area
                    
                    # If significant overlap with either polygon
                    if overlap_ratio1 > min_overlap_ratio or overlap_ratio2 > min_overlap_ratio:
                        
                        if keep_largest:
                            # Keep the larger polygon
                            if poly.geometry.area > match_poly.geometry.area:
                                to_remove.add(match_poly['ids'])
                            else:
                                to_remove.add(poly['ids'])

                        else:
                            # Keep the polygon that contains more of the other
                            if overlap_ratio2 > overlap_ratio1:
                                # match_poly is more contained within poly, remove match_poly
                                to_remove.add(match_poly['ids'])
                            else:
                                # poly is more contained within match_poly, remove poly
                                to_remove.add(poly['ids'])
            
    
    # Remove the marked polygons
    gdf_clean = gdf[~gdf['ids'].isin(list(to_remove))]
    gdf_clean = gdf_clean.drop(columns=['original_index'])
    
    print(f"Removed {len(to_remove)} intersecting polygons")
    print(f"Remaining: {len(gdf_clean)} polygons")

    path_to_save = os.path.join(DATA_PATH, f"{STATE}/{STATE}_inter_all_geocoords_clean.shp")
    gdf_clean.reset_index(drop=True).to_file(path_to_save)
    # Find polygons that are at least 70% within another polygon
    # partial_results = find_partial_overlaps(indices_tac_all_nig, min_overlap_ratio=0.7, keep_largest=True)
    # print(f"Found {len(partial_results)} partial containment relationships")
    return gdf_clean.reset_index(drop=True)



def load_data_diff_dates(state_geometry, startDate='2023-11-01', endDate='2023-11-30', features_state=pd.DataFrame()):

    import ee
    # Trigger the authentication flow.
    ee.Authenticate()
    ee.Initialize(project='fishpondsjj')
    # extract sentinel only for the state
    dfall_1km = pd.DataFrame()
    month = startDate.split('-')[1]
    area_of_interest = shapely.to_geojson(state_geometry.buffer(6000).to_crs("epsg:4326").geometry).iloc[0]
    features_aoi = []
    features_aoi.append(ee.Feature(eval(area_of_interest))) 
    fc = ee.FeatureCollection(features_aoi)

    # # define sentinel dates
    # startDate = '2023-11-01';
    # endDate = '2023-12-31';

    s2Collection = ee.ImageCollection('COPERNICUS/S2_SR').select('B2', 'B3', 'B4', 'B8', 'B11', 'B12').filterBounds(fc).filterDate(startDate, endDate).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)).median();


    ndvi = s2Collection.normalizedDifference(['B8', 'B4']).rename('NDVI');
    msavi = s2Collection.expression(
    '(2 * NIR + 1 - sqrt(pow((2 * NIR + 1), 2) - 8 * (NIR - RED))) / 2', {
        'NIR': s2Collection.select('B8'),
        'RED': s2Collection.select('B4')
    }).rename('MSAVI');
    ndwi = s2Collection.normalizedDifference(['B3', 'B8']).rename('NDWI');
    ndbi = s2Collection.normalizedDifference(['B11', 'B8']).rename('NDBI');

    del s2Collection

    # features_state.loc[:,'buffer_geometry_indiv'] = features_state.buffer(1000).to_crs("epsg:4326")
    ####
    # features_pp_1km = []
    for i, batch in enumerate(features_state.groupby(features_state.index // 200)):
        features_pp_1km = []
        indexes = []
        print('len batch 1km')
        print(len(batch[1]))
        for pp in batch[1].iterrows():
            togeojson = shapely.to_geojson(pp[1]['buffer_geometry_indiv'])
            features_pp_1km.append(ee.Feature(eval(togeojson)).set('index', ee.Number(pp[0])))
            indexes.append(pp[0])
    
        clipRegion_fc_1km = ee.FeatureCollection(features_pp_1km)
    
        ndviClipped = ndvi.clipToCollection(clipRegion_fc_1km)
        msaviClipped = msavi.clipToCollection(clipRegion_fc_1km)
        ndwiClipped = ndwi.clipToCollection(clipRegion_fc_1km)
        ndbiClipped = ndbi.clipToCollection(clipRegion_fc_1km)
    
        indicesStack = ndviClipped.addBands([msaviClipped, ndwiClipped, ndbiClipped])
    
        zonalStats = indicesStack.reduceRegions(collection=clipRegion_fc_1km,
          reducer=ee.Reducer.mean(),
          scale=10,
          crs='EPSG:4326',
          maxPixelsPerRegion=3e10
        )
        
        columns_df = ['MSAVI', 'NDBI', 'NDVI', 'NDWI', 'index']
        nested_list = zonalStats.reduceColumns(ee.Reducer.toList(len(columns_df)), columns_df).values().get(0)
        data = nested_list.getInfo()
        columns_df = [f'MSAVI_{month}', f'NDBI_{month}', f'NDVI_{month}', f'NDWI_{month}', 'index']
        df_1km = pd.DataFrame(columns=columns_df, index=indexes)
       
        # df_1km = pd.DataFrame(data, columns=columns_df).fillna(-100)
        df_temp = pd.DataFrame(data, columns=columns_df).fillna(-100).set_index('index')
        df_1km = df_1km.combine_first(df_temp)
        print('len of pond 1km')
        print(df_1km.shape)

        dfall_1km = pd.concat([dfall_1km, df_1km])
        
        del nested_list, data, indicesStack, ndviClipped, ndbiClipped, msaviClipped, ndwiClipped, clipRegion_fc_1km, df_temp
        

    return dfall_1km

def create_buffer_groups(buffer_size, DATA_PATH, path_dest_state_w_pred, STATE):
    ss = buffer_size
    features_state = gpd.read_file(path_dest_state_w_pred)
    coords = features_state.geometry.apply(lambda p: (p.centroid.x, p.centroid.y)).tolist()
    dbscan = DBSCAN(eps=ss, min_samples=3) # n_init is important for robustness
    dbscan.fit(coords)
    clusters = dbscan.fit_predict(coords)
    features_state[f'clus_lab{ss}'] = clusters
    groups = features_state.groupby([f'clus_lab{ss}']).size()
    features_state[f'clus_siz{ss}'] = features_state[f'clus_lab{ss}'].apply(lambda x: groups[x] if x!=-1 else 0)
    # features_state[f'clus_lab{ss}'] = features_state[f'clus_lab{ss}'].apply(lambda x: 1 if x != -1 else 0)
    
    # dest_state_intersection_wfeatures = features_state.set_index('ids').join(df3.set_index('ids')).reset_index().dropna(subset=['ids'])
    path_to_save = os.path.join(DATA_PATH, f"{STATE}/{STATE}_inter_all_geocoords_wcluster_size.shp")
    features_state.to_file(path_to_save)
    return features_state

    
    
def extract_indices(DATA_PATH, path_dest_state_w_pred, STATE, ROOT_NIGERIA_DATA):
    # load state to extract sentinel only for the state, create a buffer of 6km to cover the grid outside the state
    DATA_PATH_ss = ROOT_NIGERIA_DATA
    nigeria_all = gpd.read_file(os.path.join(DATA_PATH_ss, 
                                            "Nigeria/ngaadmbndaadm1osgof/nga_admbnda_adm1_osgof_20161215.shp")).to_crs('EPSG:3857')
    nigeria_all['admin1Name'] = nigeria_all['admin1Name'].apply(lambda x: x.replace(" ", ""))
    state_geometry = nigeria_all[nigeria_all['admin1Name'] == STATE]
    # state_geometry = state_geometry.buffer(6000).to_crs("epsg:4326")
  
    # convert to sentinel crs, calcualte the centroid, to use if the polygon is too small
    
    features_state = gpd.read_file(path_dest_state_w_pred)
    features_state.loc[:,'centroid'] = features_state.geometry.centroid
    features_state.loc[:,'geometry_to_crs'] = features_state.geometry.to_crs("epsg:4326")
    features_state.loc[:,'centroid_to_crs'] = features_state.geometry.centroid.to_crs("epsg:4326")
    features_state.loc[:,'buffer_s_geometry_to_crs'] = features_state.buffer(5).to_crs("epsg:4326")
    print(f'Number of polygons in {STATE} = {len(features_state)}')

    # create a buffer around each predictions and merge everything. THe idea is to have different areas covering multiple polyongs
    buffer_distance = 1500  # Adjust this value as needed
    # Create the buffered polygon
    buffered_polygon = features_state.buffer(buffer_distance)
    features_state.loc[:,'buffer_geometry'] = buffered_polygon
    features_state.loc[:,'buffer_geometry_to_crs'] = features_state.loc[:,'buffer_geometry'].to_crs("epsg:4326")
    dissolved_gdf = features_state['buffer_geometry_to_crs'].union_all()
    # try:
    #     buffer_simple = [box(*dissolved_gdf.geoms[x].bounds) for x in range(len(dissolved_gdf.geoms))]
    # except:
    #     buffer_simple = [box(*dissolved_gdf.bounds)]
    # buffer_simple = gpd.GeoDataFrame({'geometry': buffer_simple}, crs="EPSG:4326")
    features_geometry = features_state.set_geometry("geometry_to_crs")

    

    features_state.loc[:,'buffer_geometry_indiv'] = features_state.buffer(1000).to_crs("epsg:4326")
    print('jan')
    df_1km_1 = load_data_diff_dates(state_geometry, startDate='2024-01-01', endDate='2024-01-31', features_state=features_state)
    dfall_combine = df_1km_1.drop(columns=['index'])
    
    print(dfall_combine.columns)
    print('check shape before combining')
    print(dfall_combine[dfall_combine['MSAVI_01'].isna()].shape, len(dfall_combine))    
    # dest_state_intersection = gpd.read_file(path_dest_state_w_pred)
    dest_state_intersection_wfeatures = features_state.join(dfall_combine)

    print(dest_state_intersection_wfeatures.columns)

    # dest_state_intersection_wfeatures = dest_state_intersection_wfeatures.drop(columns=['split', 'area', 'length', 'inv_isoper',
    #    'ave_r', 'ave_g', 'ave_b', 'std_r', 'std_g', 'std_b', 'MSAVI_pond', 'NDBI_pond',
    #    'NDVI_pond', 'NDWI_pond', 'MSAVI_stat', 'NDBI_state',
    #    'NDVI_state', 'NDWI_state'])
    dest_state_intersection_wfeatures[['MSAVI_01', 'NDBI_01', 
                                       'NDVI_01', 'NDWI_01']] = dest_state_intersection_wfeatures[['MSAVI_01', 'NDBI_01', 
                                                                                                   'NDVI_01', 'NDWI_01']].astype('float32')

    dest_state_intersection_wfeatures = dest_state_intersection_wfeatures[['ids', 'conf', 'split', 'class', 'area', 'length', 'inv_isoper',
       'ave_r', 'ave_g', 'ave_b', 'std_r', 'std_g', 'std_b', 'MSAVI_1km',
       'NDBI_1km', 'NDVI_1km', 'NDWI_1km', 'MSAVI_pond', 'NDBI_pond',
       'NDVI_pond', 'NDWI_pond', 'state', 'MSAVI_stat', 'NDBI_state',
       'NDVI_state', 'NDWI_state', 'clus_lab15', 'clus_siz15', 'geometry', 
       'MSAVI_01', 'NDBI_01', 'NDVI_01', 'NDWI_01']]

    # dest_state_intersection_wfeatures = dest_state_intersection_wfeatures[['ids',
    #                                                                       'MSAVI_01', 'NDBI_01', 'NDVI_01', 'NDWI_01',
    #                                   'MSAVI_02', 'NDBI_02', 'NDVI_02', 'NDWI_02']]
    # dest_state_intersection_wfeatures['Fishpond'] = dest_state_intersection_wfeatures['Fishpond'].astype('int16')
    path_to_save = os.path.join(DATA_PATH, f"{STATE}/{STATE}_inter_all_geocoords_time_series.shp")
    dest_state_intersection_wfeatures.to_file(path_to_save)

    # return dfall_buff, dfall, dfall_1km, dfall_combine, dest_state_intersection_wfeatures

    

if __name__ == "__main__":
     # login to GEE
   
    import argparse
    from argparse import FileType, ArgumentParser

    parser = ArgumentParser()
    pred_files = parser.add_argument_group()
    
    pred_files.add_argument(
        '--STATE',
        metavar='STATE',
        help='STATE',
        type=str)
    
    pred_files.add_argument(
        '--path_to_savergb',
        metavar='path_to_savergb',
        help='Path of file after rgb values',
        type=str)


    pred_files.add_argument(
        '--DATA_PATH',
        metavar='DATA_PATH',
        help='Root of data path to save results',
        type=str)

    
    pred_files.add_argument(
        '--ROOT_NIGERIA_DATA',
        metavar='ROOT_NIGERIA_DATA',
        help='Root of location of Nigeria grid',
        default='/lustre/davis/FishPonds_project/share/data/',
        type=str)

    ins = parser.parse_args()
    
    path_to_savergb = ins.path_to_savergb
    STATE = ins.STATE
    DATA_PATH = ins.DATA_PATH
    ROOT_NIGERIA_DATA = ins.ROOT_NIGERIA_DATA
    extract_indices(DATA_PATH, path_to_savergb, STATE, ROOT_NIGERIA_DATA)