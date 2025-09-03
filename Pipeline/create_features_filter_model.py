# from qgis.core import *
# from osgeo import gdal




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


def extract_indices(DATA_PATH, path_dest_state_intersection, STATE, ROOT_NIGERIA_DATA):

    import ee
    # Trigger the authentication flow.
    ee.Authenticate()
    ee.Initialize(project='fishpondsjj')


    # load state to extract sentinel only for the state, create a buffer of 6km to cover the grid outside the state
    DATA_PATH_ss = ROOT_NIGERIA_DATA
    nigeria_all = gpd.read_file(os.path.join(DATA_PATH_ss, 
                                            "Nigeria/ngaadmbndaadm1osgof/nga_admbnda_adm1_osgof_20161215.shp")).to_crs('EPSG:3857')
    nigeria_all['admin1Name'] = nigeria_all['admin1Name'].apply(lambda x: x.replace(" ", ""))
    state_geometry = nigeria_all[nigeria_all['admin1Name'] == STATE]
    # state_geometry = state_geometry.buffer(6000).to_crs("epsg:4326")
  
    # convert to sentinel crs, calcualte the centroid, to use if the polygon is too small
    
    features_state = gpd.read_file(path_dest_state_intersection)
    features_state.loc[:,'centroid'] = features_state.geometry.centroid
    features_state.loc[:,'geometry_to_crs'] = features_state.geometry.to_crs("epsg:4326")
    features_state.loc[:,'centroid_to_crs'] = features_state.geometry.centroid.to_crs("epsg:4326")
    features_state.loc[:,'buffer_s_geometry_to_crs'] = features_state.buffer(5).to_crs("epsg:4326")
    print(f'Number of polygons in {STATE} = {len(features_state)}')

    # create a buffer around each predictions and merge everything. THe idea is to have different areas covering multiple polyongs
    buffer_distance = 1500  # Adjust this value as needed
    dfall = pd.DataFrame()
    dfall_buff = pd.DataFrame()
    dfall_1km = pd.DataFrame()
    dfall_combine = pd.DataFrame()
    # Create the buffered polygon
    buffered_polygon = features_state.buffer(buffer_distance)
    features_state.loc[:,'buffer_geometry'] = buffered_polygon
    features_state.loc[:,'buffer_geometry_to_crs'] = features_state.loc[:,'buffer_geometry'].to_crs("epsg:4326")
    dissolved_gdf = features_state['buffer_geometry_to_crs'].union_all()
    try:
        buffer_simple = [box(*dissolved_gdf.geoms[x].bounds) for x in range(len(dissolved_gdf.geoms))]
    except:
        buffer_simple = [box(*dissolved_gdf.bounds)]
    buffer_simple = gpd.GeoDataFrame({'geometry': buffer_simple}, crs="EPSG:4326")
    features_geometry = features_state.set_geometry("geometry_to_crs")

    # merge the area(after the buffering and union) with the predictions to see where the predictions are contain

    df3 = gpd.sjoin(features_geometry, buffer_simple, 
                    how="left", 
                    predicate='within').drop_duplicates(subset=['ids'])
    

    # extract sentinel only for the state
    area_of_interest = shapely.to_geojson(state_geometry.buffer(6000).to_crs("epsg:4326").geometry).iloc[0]
    features_aoi = []
    features_aoi.append(ee.Feature(eval(area_of_interest))) 
    fc = ee.FeatureCollection(features_aoi)

    # define sentinel dates
    startDate = '2023-11-01';
    endDate = '2023-12-31';

    s2Collection = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(fc).filterDate(startDate, endDate).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)).median();


    ndvi = s2Collection.normalizedDifference(['B8', 'B4']).rename('NDVI');
    msavi = s2Collection.expression(
    '(2 * NIR + 1 - sqrt(pow((2 * NIR + 1), 2) - 8 * (NIR - RED))) / 2', {
        'NIR': s2Collection.select('B8'),
        'RED': s2Collection.select('B4')
    }).rename('MSAVI');
    ndwi = s2Collection.normalizedDifference(['B3', 'B8']).rename('NDWI');
    ndbi = s2Collection.normalizedDifference(['B11', 'B8']).rename('NDBI');

    del s2Collection

    # loop over the areas that contains the predictions and clip the indices for that area
    print(f"total buffers = {len(np.unique(df3['index_right']))}")
    for buffer_index in np.unique(df3['index_right']):
        print(f'buffer = {buffer_index}')
        polygons_state_intersect = df3[df3['index_right'] == buffer_index]
        area_of_interest_group = shapely.to_geojson(buffer_simple.loc[buffer_index].geometry)
        area_of_interest_group_fc = ee.FeatureCollection([ee.Feature(eval(area_of_interest_group))])

        # clip the area
        ndviClipped = ndvi.clipToCollection(area_of_interest_group_fc)
        msaviClipped = msavi.clipToCollection(area_of_interest_group_fc)
        ndwiClipped = ndwi.clipToCollection(area_of_interest_group_fc)
        ndbiClipped = ndbi.clipToCollection(area_of_interest_group_fc)

        indicesStack = ndviClipped.addBands([msaviClipped, ndwiClipped, ndbiClipped])

        for i, batch in enumerate(polygons_state_intersect.groupby(polygons_state_intersect.index // 200)):
            features_pp = []
            features_pp_buffer_s = []
            indexes = []
            print('len batch')
            print(len(batch[1]))
            for pp in batch[1].iterrows():
                togeojson = shapely.to_geojson(pp[1]['geometry_to_crs'])
                features_pp.append(ee.Feature(eval(togeojson)).set('index', ee.Number(pp[0])))

                togeojson = shapely.to_geojson(pp[1]['buffer_s_geometry_to_crs'])
                features_pp_buffer_s.append(ee.Feature(eval(togeojson)).set('index', ee.Number(pp[0])))

                indexes.append(pp[0])
            
            clipRegion_fc = ee.FeatureCollection(features_pp)
            
    
            zonalStats = indicesStack.reduceRegions(collection=clipRegion_fc,
            reducer=ee.Reducer.mean(),
            scale=10,
            crs='EPSG:4326',
            maxPixelsPerRegion=3e10
            )
             
            columns_df = ['MSAVI', 'NDBI', 'NDVI', 'NDWI', 'index']
    
            nested_list = zonalStats.reduceColumns(ee.Reducer.toList(len(columns_df)), columns_df).values().get(0)
            data = nested_list.getInfo()
            
            columns_df = ['MSAVI_pond', 'NDBI_pond', 'NDVI_pond', 'NDWI_pond', 'index']
            df = pd.DataFrame(columns=columns_df, index=indexes)
            df_temp = pd.DataFrame(data, columns=columns_df).fillna(-100).set_index('index')
            df = df.combine_first(df_temp)
            print('len of pond')
            print(df.shape)
    
            del nested_list, data, clipRegion_fc, df_temp

            clipRegion_fc_s = ee.FeatureCollection(features_pp_buffer_s)

            zonalStats = indicesStack.reduceRegions(collection=clipRegion_fc_s,
            reducer=ee.Reducer.mean(),
            scale=10,
            crs='EPSG:4326',
            maxPixelsPerRegion=3e10
            )
             
            columns_df = ['MSAVI', 'NDBI', 'NDVI', 'NDWI', 'index']
    
            nested_list = zonalStats.reduceColumns(ee.Reducer.toList(len(columns_df)), columns_df).values().get(0)
            data = nested_list.getInfo()
            
            columns_df = ['MSAVI_pond_buff', 'NDBI_pond_buff', 'NDVI_pond_buff', 'NDWI_pond_buff', 'index']
            df_buff = pd.DataFrame(columns=columns_df, index=indexes)
            df_temp = pd.DataFrame(data, columns=columns_df).fillna(-100).set_index('index')
            df_buff = df_buff.combine_first(df_temp)
            # df_buff = pd.DataFrame(data, columns=columns_df).fillna(-100)
            print('len of pond buff')
            print(df_buff.shape)

            del nested_list, data, clipRegion_fc_s, df_temp

            dfall = pd.concat([dfall, df])
            dfall_buff = pd.concat([dfall_buff, df_buff])
            
        del indicesStack, ndviClipped, ndbiClipped, msaviClipped, ndwiClipped

        
    ###
    features_state.loc[:,'buffer_geometry_indiv'] = features_state.buffer(1000).to_crs("epsg:4326")
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
        columns_df = ['MSAVI_1km', 'NDBI_1km', 'NDVI_1km', 'NDWI_1km', 'index']
        df_1km = pd.DataFrame(columns=columns_df, index=indexes)
       
        # df_1km = pd.DataFrame(data, columns=columns_df).fillna(-100)
        df_temp = pd.DataFrame(data, columns=columns_df).fillna(-100).set_index('index')
        df_1km = df_1km.combine_first(df_temp)
        print('len of pond 1km')
        print(df_1km.shape)

        dfall_1km = pd.concat([dfall_1km, df_1km])
        
        del nested_list, data, indicesStack, ndviClipped, ndbiClipped, msaviClipped, ndwiClipped, clipRegion_fc_1km, df_temp
    
    dfall_combine = dfall_1km.drop(columns=['index']).join(dfall).drop(columns=['index']).join(dfall_buff.drop(columns=['index']))
    
    print('check shape before combining')
    print(dfall_combine[dfall_combine['MSAVI_pond'].isna()].shape, len(dfall_combine))    
    dest_state_intersection = gpd.read_file(path_dest_state_intersection)
    dest_state_intersection_wfeatures = dest_state_intersection.join(dfall_combine)

    print(dest_state_intersection_wfeatures.columns)

    dest_state_intersection_wfeatures.loc[dest_state_intersection_wfeatures['MSAVI_pond'].isna(), 'MSAVI_pond'] = dest_state_intersection_wfeatures.loc[dest_state_intersection_wfeatures['MSAVI_pond'].isna(), 'MSAVI_pond_buff']
    dest_state_intersection_wfeatures.loc[dest_state_intersection_wfeatures['NDBI_pond'].isna(), 'NDBI_pond'] = dest_state_intersection_wfeatures.loc[dest_state_intersection_wfeatures['NDBI_pond'].isna(), 'NDBI_pond_buff']
    dest_state_intersection_wfeatures.loc[dest_state_intersection_wfeatures['NDVI_pond'].isna(), 'NDVI_pond'] = dest_state_intersection_wfeatures.loc[dest_state_intersection_wfeatures['NDVI_pond'].isna(), 'NDVI_pond_buff']
    dest_state_intersection_wfeatures.loc[dest_state_intersection_wfeatures['NDWI_pond'].isna(), 'NDWI_pond'] = dest_state_intersection_wfeatures.loc[dest_state_intersection_wfeatures['NDWI_pond'].isna(), 'NDWI_pond_buff']

    
    dest_state_intersection_wfeatures.drop(columns=['MSAVI_pond_buff', 'NDBI_pond_buff',
       'NDVI_pond_buff', 'NDWI_pond_buff'], inplace=True)   
    
    dest_state_intersection_wfeatures = dest_state_intersection_wfeatures.drop(columns=['minx', 'miny', 'maxx', 'maxy'])
    dest_state_intersection_wfeatures[['conf', 'area', 'length', 'inv_isoper', 'ave_r',
       'ave_g', 'ave_b', 'std_r', 'std_g', 'std_b',
       'MSAVI_pond', 'NDBI_pond', 'NDVI_pond', 'NDWI_pond', 'MSAVI_1km', 'NDBI_1km', 'NDVI_1km', 'NDWI_1km']] = dest_state_intersection_wfeatures[['conf', 'area', 'length', 'inv_isoper', 'ave_r',
       'ave_g', 'ave_b', 'std_r', 'std_g', 'std_b',
       'MSAVI_pond', 'NDBI_pond', 'NDVI_pond', 'NDWI_pond', 'MSAVI_1km', 'NDBI_1km', 'NDVI_1km', 'NDWI_1km']].astype('float32')

    # dest_state_intersection_wfeatures['Fishpond'] = dest_state_intersection_wfeatures['Fishpond'].astype('int16')
    path_to_save = os.path.join(DATA_PATH, f"{STATE}/temp_all_inter_all_geocoords_wfeatures.shp")
    dest_state_intersection_wfeatures.to_file(path_to_save)

    # return dfall_buff, dfall, dfall_1km, dfall_combine, dest_state_intersection_wfeatures

def add_state_indices(DATA_PATH, STATE, dest_state_intersection_wfeatures_path='', ROOT_NIGERIA_DATA='/lustre/davis/FishPonds_project/share/data/'):
    dest_state_intersection_wfeatures = gpd.read_file(dest_state_intersection_wfeatures_path)
    dest_state_intersection_wfeatures.loc[:, 'state'] = STATE
    DATA_PATH_ss = '/lustre/davis/FishPonds_project/share/data/'
    indices_states = pd.read_csv(os.path.join(ROOT_NIGERIA_DATA, 'Nigeria/nigeria_indices/indices_perstate_all.csv'))
    dest_state_intersection_wfeatures.loc[:, 'MSAVI_stat'] = indices_states[indices_states['index'] == STATE]['MSAVI'].values[0]
    dest_state_intersection_wfeatures.loc[:, 'NDBI_state'] = indices_states[indices_states['index'] == STATE]['NDBI'].values[0]
    dest_state_intersection_wfeatures.loc[:, 'NDVI_state'] = indices_states[indices_states['index'] == STATE]['NDVI'].values[0]
    dest_state_intersection_wfeatures.loc[:, 'NDWI_state'] = indices_states[indices_states['index'] == STATE]['NDWI'].values[0]

    path_to_save = os.path.join(DATA_PATH, f"{STATE}/temp_all_inter_all_geocoords_wfeatures.shp")
    dest_state_intersection_wfeatures.to_file(path_to_save)
    # FishPonds_project/share/data/Nigeria/nigeria_indices


def check_preds_within_state(DATA_PATH, STATE, dest_state_intersection_wfeatures_path='', ROOT_NIGERIA_DATA='/lustre/davis/FishPonds_project/share/data/', state_geometry=None):

    DATA_PATH_ss = ROOT_NIGERIA_DATA
    nigeria_all = gpd.read_file(os.path.join(DATA_PATH_ss, 
                                            "Nigeria/ngaadmbndaadm1osgof/nga_admbnda_adm1_osgof_20161215.shp")).to_crs('EPSG:3857')
    nigeria_all['admin1Name'] = nigeria_all['admin1Name'].apply(lambda x: x.replace(" ", ""))
    state_geometry = nigeria_all[nigeria_all['admin1Name'] == STATE]

    
    # check all preds are within state
    dest_state_intersection_wfeatures = gpd.read_file(dest_state_intersection_wfeatures_path)
    is_contained = gpd.sjoin(dest_state_intersection_wfeatures, state_geometry, how="inner", predicate="within")

    # drop the columns from the state data
    contains = is_contained.drop(columns=['admin1Name', 'admin1Pcod', 'admin1RefN', 'admin1AltN', 'admin1Al_1',
       'admin0Name', 'admin0Pcod', 'date', 'validOn', 'validTo', 'Shape_Leng',
       'Shape_Area', 'index_right'])
    path_to_save = os.path.join(DATA_PATH, f"{STATE}/{STATE}_inter_all_geocoords_wfeatures.shp")

    # save preds in state as the final dataset
    if len(contains) == len(dest_state_intersection_wfeatures):
        print("all is within state, save this")
        # save this as the final data set dest_state_intersection
        contains_save = dest_state_intersection_wfeatures[dest_state_intersection_wfeatures['ids'].isin(contains['ids'])]
        contains_save.to_file(path_to_save)

    else:
        print('create a new data frame')
        # save the final data set 
        contains_save = dest_state_intersection_wfeatures[dest_state_intersection_wfeatures['ids'].isin(contains['ids'])]
        contains_save.to_file(path_to_save)
        # also save this for backup data set 
        nocontains = dest_state_intersection_wfeatures[~dest_state_intersection_wfeatures['ids'].isin(contains['ids'])]
        nocontains.to_file(path_to_save.replace(f"/{STATE}_", f"/notin_{STATE}_"))
    

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