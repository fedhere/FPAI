# from pylabel import importer
import glob
import numpy as np
import os
from shapely.geometry import box, Polygon, MultiPolygon, Point

from shapely.validation import make_valid
import sys
import geopandas as gpd
import rasterio
import rasterio.transform
from rasterio.crs import CRS

SPLIT = 'test'

def save_transformation(name_files, root_to_tif, loc_save_transform=f"delta_geocoord/{SPLIT}/transformation"):
    """
    Raster files are large, save only the transformation needed to convert pixels to geoocordiantes.
    
    Arguments:
    
      name_files: list of tif names
      validation: name tif images
      root_to_tiff: location of the raster files
      loc_save_transform: location of where you want to the transformation to be saved
    
    Outputs:
    
      save the transformation per raster file as a txt file (e.g: 10 raster files, 10 txt files each containing the transformation )
    
      example of txt file:
    
        "0.1, 0.0, 631440.6851349056, 0.0, -0.1, 622789.7903693904, 0.0, 0.0, 1.0
        EPSG:3857
        "
    
        first line: transformation (matrix 3x3)
        second line: crs
    """

    for name_file in name_files: 
      # valid_0_df = df_annon_valid[df_annon_valid['img_filename'] == validation.split("/")[-1]]
      #check thisss!!!
      # name_file = '_'.join(val.split('/')[-1].split('.')[0].split('_')[0:-1])
      print(name_file)
      path_to_tiff = os.path.join(root_to_tif,f"{name_file}_crop.tif")
      if path_to_tiff.endswith(".tif"):
          with rasterio.open(path_to_tiff) as src:
              raster_transform = src.transform
          print(raster_transform)
          with open(os.path.join(loc_save_transform, f"{name_file}_transform.txt"), "w") as f:
              f.write(str(raster_transform[0:])[1:-1] + '\n')
              f.write(str(src.crs))
      else:
          print(f"no tif file of {name_file}")


def load_transform_from_txt(txt_file):
    """
    Loads a raster transformation matrix from a text file.
        example:

        0.1, 0.0, 310217.5765622778, 0.0, -0.1, 747273.0560375599, 0.0, 0.0, 1.0
        EPSG:3857
      
      Input:

        txt_file: location of txt file containing transformation

      Output:

        transform: raster transformation
        CRS: crs of transformation
    
    """

    with open(txt_file, 'r') as f:
        lines = f.readlines()
    # Assuming the transformation matrix is stored as 6 values in the text file
    values = [float(x) for x in lines[0].split(', ')]
    transform = rasterio.Affine(values[0], values[1], values[2], 
                                values[3], values[4], values[5],
                                values[6], values[7], values[8])
    return transform, CRS.from_epsg(int(lines[1].split(":")[-1]))
