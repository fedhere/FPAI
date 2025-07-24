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


def extract_rgb(path_to_intersection):

    # from qgis.core import *
    from osgeo import gdal
    from qgis import processing
    state_intersection = gpd.read_file(path_to_intersection)

    dest_state_intersection = pd.concat([state_intersection, state_intersection.bounds, state_intersection.area, state_intersection.length], axis=1)
    dest_state_intersection.rename(columns={0: 'area', 1: 'length'}, inplace=True)

    dest_state_intersection = dest_state_intersection[dest_state_intersection['area'] > 0]
    dest_state_intersection.loc[:, 'inv_isoperimetric_ratio'] = dest_state_intersection.loc[:, 'area']/(dest_state_intersection.loc[:, 'length']**2)


    loc_save_data = ""

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

            
        extent = QgsRectangle(xmin, ymin, xmax, ymax)
        # location to save raster files
        file_writer = QgsRasterFileWriter(os.path.join(loc_save_data, f'{k["properties"]["ids"]}.tif'))
        file_writer.writeRaster(pipe,
                            int(w/0.1),
                            int(h/0.1),
                            extent,
                            rlayer.crs())
        img = os.path.join(os.path.join(loc_save_data, f'{k["properties"]["ids"]}.tif'))
        
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
    QgsProject.instance().clear()
    # QgsApplication.exitQgis()
    qgs.exitQgis()
    qgs.exit()
    return dest_state_intersection
    

def extract_indices(path_dest_state_intersection, STATE):
    # load state to extract sentinel only for the state, create a buffer of 6km to cover the grid outside the state
    DATA_PATH = '/lustre/davis/FishPonds_project/share/data/'
    nigeria_all = gpd.read_file(os.path.join(DATA_PATH, 
                                            "Nigeria/ngaadmbndaadm1osgof/nga_admbnda_adm1_osgof_20161215.shp")).to_crs('EPSG:3857')
    nigeria_all['admin1Name'] = nigeria_all['admin1Name'].apply(lambda x: x.replace(" ", ""))
    state_geometry = nigeria_all[nigeria_all['admin1Name'] == STATE]
    # state_geometry = state_geometry.buffer(6000).to_crs("epsg:4326")
  
    # convert to sentinel crs, calcualte the centroid, to use if the polygon is too small
    
    features_state = gpd.read_file(path_dest_state_intersection)
    features_state.loc[:,'centroid'] = features_state.geometry.centroid
    features_state.loc[:,'geometry_to_crs'] = features_state.geometry.to_crs("epsg:4326")
    features_state.loc[:,'centroid_to_crs'] = features_state.geometry.centroid.to_crs("epsg:4326")
    print(f'Number of polygons in {STATE} = {len(features_state)}')

    # create a buffer around each predictions and merge everything. THe idea is to have different areas covering multiple polyongs
    buffer_distance = 1500  # Adjust this value as needed
    dfall = pd.DataFrame()
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
    

    # login to GEE
    import ee
    # Trigger the authentication flow.
    ee.Authenticate()
    ee.Initialize(project='fishpondsjj')
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

    # loop over the areas that contains the predictions and clip the indices for that area
    print(f'total buffers = {len(buffer_simple.index.values)}')
    for buffer_index in buffer_simple.index.values:
        print(f'buffer = {buffer_index}')
        polygons_state_intersect = df3[df3['index_right'] == buffer_index]
        area_of_interest_group = shapely.to_geojson(buffer_simple.loc[buffer_index].geometry)
        area_of_interest_group_fc = ee.FeatureCollection([ee.Feature(eval(area_of_interest_group))])

        # clip the area
        ndviClipped = ndvi.clip(area_of_interest_group_fc)
        msaviClipped = msavi.clip(area_of_interest_group_fc)
        ndwiClipped = ndwi.clip(area_of_interest_group_fc)
        ndbiClipped = ndbi.clip(area_of_interest_group_fc)

        indicesStack = ndviClipped.addBands([msaviClipped, ndwiClipped, ndbiClipped])
        
        # iterate over the predicitions within the area
        features_pp = []
        features_pp_c = []
        for pp in polygons_state_intersect.iterrows():
            togeojson = shapely.to_geojson(pp[1]['geometry_to_crs'])
            features_pp.append(ee.Feature(eval(togeojson)).set('index', ee.Number(pp[0])))

            togeojson = shapely.to_geojson(pp[1]['centroid_to_crs'])
            features_pp_c.append(ee.Feature(eval(togeojson)).set('index', ee.Number(pp[0])))
            
        clipRegion_fc = ee.FeatureCollection(features_pp)

        zonalStats = indicesStack.reduceRegions(collection=clipRegion_fc,
        reducer=ee.Reducer.mean(),
        scale=10,
        crs='EPSG:4326'
        )
         
        columns_df = ['MSAVI', 'NDBI', 'NDVI', 'NDWI', 'index']

        nested_list = zonalStats.reduceColumns(ee.Reducer.toList(len(columns_df)), columns_df).values().get(0)
        data = nested_list.getInfo()

        df = pd.DataFrame(data, columns=columns_df).fillna(-100)

        del nested_list, data, indicesStack, ndviClipped, ndbiClipped, msaviClipped, ndwiClipped, clipRegion_fc

        # # do the same for the centroids
        # clipRegion_fc_c = ee.FeatureCollection(features_pp_c)

        # zonalStats = indicesStack.reduceRegions(collection=clipRegion_fc_c,
        # reducer=ee.Reducer.mean(),
        # scale=10,
        # crs='EPSG:4326'
        # )

        # nested_list = zonalStats.reduceColumns(ee.Reducer.toList(len(columns_df)), columns_df).values().get(0)
        # data = nested_list.getInfo()
        
        # columns_df = ['MSAVI_c', 'NDBI_c', 'NDVI_c', 'NDWI_c', 'index_c']
        # df_c = pd.DataFrame(data, columns=columns_df).fillna(-100)

        # df = df.set_index('index').join(df_c.set_index('index_c'))
        dfall = pd.concat([dfall, df])
    ###
    buffered_polygon_indivi = features_state.buffer(1000)
    features_state.loc[:,'buffer_geometry_indiv'] = buffered_polygon_indivi
    features_state.loc[:,'buffer_geometry_indiv'] = features_state.loc[:,'buffer_geometry_indiv'].to_crs("epsg:4326")
    ####

    features_pp = []
    features_pp_c = []
    for pp in features_state.iterrows():
        features_pp = []
        togeojson = shapely.to_geojson(pp[1]['buffer_geometry_indiv'])
        features_pp.append(ee.Feature(eval(togeojson)).set('index', ee.Number(pp[0])))

        clipRegion_fc = ee.FeatureCollection(features_pp)
    
        ndviClipped = ndvi.clipToCollection(clipRegion_fc)
        msaviClipped = msavi.clipToCollection(clipRegion_fc)
        ndwiClipped = ndwi.clipToCollection(clipRegion_fc)
        ndbiClipped = ndbi.clipToCollection(clipRegion_fc)
    
        # Resample to 1-meter
        ndvi1m = ndviClipped.resample('bilinear').reproject(crs='EPSG:4326', scale= 1)
        msavi1m = msaviClipped.resample('bilinear').reproject(crs='EPSG:4326', scale= 1)
        ndwi1m = ndwiClipped.resample('bilinear').reproject(crs='EPSG:4326', scale= 1)
        ndbi1m = ndbiClipped.resample('bilinear').reproject(crs='EPSG:4326', scale= 1)
    
        indicesStack = ndviClipped.addBands([msaviClipped, ndwiClipped, ndbiClipped])
    
        zonalStats = indicesStack.reduceRegions(collection=clipRegion_fc,
          reducer=ee.Reducer.mean(),
          scale=10,
          crs='EPSG:4326',
          maxPixelsPerRegion=3e10
        )
        # sample_result = zonalStats.first().getInfo()
        # columns = list(sample_result['properties'].keys()) 
        columns_df = ['MSAVI', 'NDBI', 'NDVI', 'NDWI', 'index']
        nested_list = zonalStats.reduceColumns(ee.Reducer.toList(len(columns_df)), columns_df).values().get(0)
        data = nested_list.getInfo()

        columns_df = ['MSAVI_1km', 'NDBI_1km', 'NDVI_1km', 'NDWI_1km', 'index']
    
        df_1km = pd.DataFrame(data, columns=columns_df).fillna(-100)

        # df = dfall.set_index('index').join(df_1km.set_index('index_1km'))
        dfall_1km = pd.concat([dfall_1km, df_1km])

        del nested_list, data, indicesStack, ndviClipped, ndbiClipped, msaviClipped, ndwiClipped, clipRegion_fc

    dfall_combine = dfall_1km.set_index('index').join(dfall.set_index('index'))
                
        
    dest_state_intersection = gpd.read_file(path_dest_state_intersection)
    dest_state_intersection_wfeatures = dest_state_intersection.join(dfall_combine)

    # if the indices is not return for the polygon, use the indices for the centroid, drop the centroid indices
    # for ii in ['MSAVI', 'NDBI', 'NDVI', 'NDWI']:
    #     dest_state_intersection_wfeatures[ii] = dest_state_intersection_wfeatures[ii].fillna(dest_state_intersection_wfeatures[ii+'_c'])
    # dest_state_intersection_wfeatures = dest_state_intersection_wfeatures.drop(columns=['MSAVI_c', 'NDBI_c', 'NDVI_c', 'NDWI_c'])

    print(dest_state_intersection_wfeatures.columns)
    dest_state_intersection_wfeatures = dest_state_intersection_wfeatures.drop(columns=['minx', 'miny', 'maxx', 'maxy'])
    dest_state_intersection_wfeatures[['conf', 'area', 'length', 'inv_isoper', 'ave_r',
       'ave_g', 'ave_b', 'std_r', 'std_g', 'std_b',
       'MSAVI', 'NDBI', 'NDVI', 'NDWI', 'MSAVI_1km', 'NDBI_1km', 'NDVI_1km', 'NDWI_1km']] = dest_state_intersection_wfeatures[['conf', 'area', 'length', 'inv_isoper', 'ave_r',
       'ave_g', 'ave_b', 'std_r', 'std_g', 'std_b',
       'MSAVI', 'NDBI', 'NDVI', 'NDWI', 'MSAVI_1km', 'NDBI_1km', 'NDVI_1km', 'NDWI_1km']].astype('float32')

    # dest_state_intersection_wfeatures['Fishpond'] = dest_state_intersection_wfeatures['Fishpond'].astype('int16')
    path_to_save = os.path.join(DATA_PATH, f"{STATE}/temp_{STATE}_inter_all_geocoords_wfeatures.shp")
    dest_state_intersection_wfeatures.to_file(path_to_save)
    # return dest_state_intersection_wfeatures, state_geometry


def check_preds_within_state(DATA_PATH, STATE, dest_state_intersection_wfeatures_path='', state_geometry=None):

    DATA_PATH_ss = '/lustre/davis/FishPonds_project/share/data/'
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
        contains.to_file(path_to_save)

    else:
        print('create a new data frame')
        # save the final data set 
        contains.to_file(path_to_save)
        # also save this for backup data set 
        nocontains = dest_state_intersection_wfeatures[~dest_state_intersection_wfeatures['ids'].isin(contains['ids'])]
        nocontains.to_file(path_to_save.replace(f"/{STATE}_", f"/notin_{STATE}_"))
    

if __name__ == "__main__":
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
        'path_to_savergb',
        metavar='path_to_savergb',
        help='Path of file after rgb values',
        type=str)

    ins = parser.parse_args()
    
    path_to_savergb = ins.path_to_savergb
    STATE = ins.STATE
    extract_indices(path_to_savergb, STATE)
