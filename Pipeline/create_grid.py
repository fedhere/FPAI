import geopandas as gpd
import rasterio
import rasterio.transform
from rasterio.crs import CRS
from shapely.geometry import Polygon, box
import numpy as np
import os
import sys
import glob



def create_grid(DATA_PATH, STATE, GRID_NUM=0, load=False):
    print(f'CREATING GRID FOR {STATE}')
    if not os.path.exists(os.path.join(DATA_PATH, f"{STATE}/grid_{STATE}.shp")):
        nigeria_all = gpd.read_file(os.path.join(DATA_PATH, "Nigeria/ngaadmbndaadm1osgof/nga_admbnda_adm1_osgof_20161215.shp")).to_crs('EPSG:3857')
        nigeria_all['admin1Name'] = nigeria_all['admin1Name'].apply(lambda x: x.replace(" ", ""))
        state_geometry = nigeria_all[nigeria_all['admin1Name'] == STATE]
        #state_geometry.total_bounds
    
        # create grid in state
        xmin, ymin, xmax, ymax = state_geometry.total_bounds
    
        length = 6000#20000
        wide = 6000#20000
    
        cols = np.arange(xmin, xmax + wide, wide)
        rows = np.arange(ymin, ymax + length, length)
    
        polygons = []
        for x in cols[:-1]:
            for y in rows[:-1]:
                polygons.append(Polygon([(x,y), (x+wide, y), (x+wide, y+length), (x, y+length)]))
                # print(polygons[-1])
    
        grid = gpd.GeoDataFrame({'geometry':polygons}, crs='EPSG:3857')
    
        grid_state = grid.iloc[state_geometry.sjoin_nearest(grid)['index_right']].sort_index().reset_index(drop=True)
        grid_state.to_file(os.path.join(DATA_PATH, f"{STATE}/grid_{STATE}.shp"))
    
    else:
        print(f'Grid for {STATE} already exist, loading it')
        
    if load==True:
        grid = gpd.read_file(os.path.join(DATA_PATH, f"{STATE}/grid_{STATE}.shp"))
        return grid
    