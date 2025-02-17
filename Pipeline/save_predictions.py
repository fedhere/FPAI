# predictions from model 
import utils_mask_to_pixels

import geopandas as gpd
import rasterio
import rasterio.transform
from rasterio.crs import CRS
import cv2
import os
import glob
import numpy as np
def save_predictions(RUNS, STATE, GRID_NUM, k, i, j):

    #$WORKDIR/users/2221/FishPonds/YOLO/yolov7/seg/runs/test_all_{STATE}_{GRID_NUM}/
    path_to_masks_ogun = glob.glob(f"/lustre/davis/FishPonds_project/share/{RUNS}/test_all_{STATE}_{GRID_NUM}/{STATE}_g{k}c{i}r{j}/labels/mask_*")
    path_to_pred = glob.glob(f"/lustre/davis/FishPonds_project/share/{RUNS}/test_all_{STATE}_{GRID_NUM}/{STATE}_g{k}c{i}r{j}/labels/*.txt")
    path_to_imgs = glob.glob(f"/lustre/davis/FishPonds_project/share/{RUNS}/test_all_{STATE}_{GRID_NUM}/{STATE}_g{k}c{i}r{j}/*.png")
    
    if len(path_to_masks_ogun)==0:
        print(f"test_all_{STATE}_{GRID_NUM}/{STATE}_g{k}c{i}r{j} does not have predictions")
        print(os.path.exists(f"/lustre/davis/FishPonds_project/share/{RUNS}/test_all_{STATE}_{GRID_NUM}/{STATE}_g{k}c{i}r{j}/*.png"))
        print(path_to_imgs)
        os.remove(path_to_imgs[0])
    else:
        p_pred = []
        img = cv2.imread(path_to_imgs[0])
        mask_test = np.load(path_to_masks_ogun[0])
        # print("size mask")
        # print(mask_test.shape)
        masks_true_size = np.zeros((len(mask_test), img.shape[0], img.shape[1], 1))
        mask_to_polygons_list = []
    
    
        for ids in range(len(mask_test)):
            masks_true_size[ids] = utils_mask_to_pixels.scale_masks((mask_test.shape[1],mask_test.shape[-1]), np.uint8(mask_test[ids]), (img.shape[0], img.shape[1], 3))
            mask_to_polygon = utils_mask_to_pixels.mask_to_polygons(masks_true_size[ids])
            if len(mask_to_polygon) > 1:
                sizes_len = []
                for ixs in range(len(mask_to_polygon)):
                    sizes_len.append(len(mask_to_polygon[ixs]))
                    mask_to_polygons_list.append(mask_to_polygon[np.argmax(np.array(sizes_len))])
            else:
                mask_to_polygons_list.append(mask_to_polygon[0])
    
        for imt in range(mask_test.shape[0]):
            points = [[mask_to_polygons_list[imt][ixs, 0].item(), mask_to_polygons_list[imt][ixs, 1].item()] for ixs in range(len(mask_to_polygons_list[imt]))] 
            p_pred.append(points)
        
        print(f"test_all_{STATE}_{GRID_NUM}/{STATE}_g{k}c{i}r{j} has {len(p_pred)} predictions ")
        save_p = f"/lustre/davis/FishPonds_project/share/{RUNS}/test_all_{STATE}_{GRID_NUM}/{STATE}_g{k}c{i}r{j}/labels/predictions_{STATE}_g{k}c{i}r{j}.npz"
        np.savez(save_p, *p_pred)
        os.remove(path_to_imgs[0])
        os.remove(path_to_masks_ogun[0])

        # p_conf = []
        # with open(path_to_pred[0], 'r') as f:
        #     lines = f.readlines()
        # # Assuming the transformation matrix is stored as 6 values in the text file
        # for l in lines:
        #     values = [float(x) for x in l.split(' ')]
        #     p_conf.append(values[-1])
        # save_conf = f"/lustre/davis/FishPonds_project/share/{RUNS}/test_all_{STATE}_{GRID_NUM}/{STATE}_g{k}c{i}r{j}/labels/p_conf_{STATE}_g{k}c{i}r{j}.npz"
        # np.savez(save_conf, *p_conf)
