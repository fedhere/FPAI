
import numpy as np
import cv2
import scipy
import os
import geopandas as gpd
import rasterio
import rasterio.transform
import utils_pixel_to_geo
import pandas as pd
from shapely.geometry import box, Polygon, MultiPolygon, Point
import glob

def scale_masks(img1_shape, masks, img0_shape, ratio_pad=None):
    """
    img1_shape: model input shape, [h, w]
    img0_shape: origin pic shape, [h, w, 3]
    masks: [h, w, num]
    resize for the most time
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    tl_pad = int(pad[1]), int(pad[0])  # y, x
    br_pad = int(img1_shape[0] - pad[1]), int(img1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    # masks_h, masks_w, n
    masks = masks[tl_pad[0]:br_pad[0], tl_pad[1]:br_pad[1]]
    # 1, n, masks_h, masks_w
    # masks = masks.permute(2, 0, 1).contiguous()[None, :]
    # # shape = [1, n, masks_h, masks_w] after F.interpolate, so take first element
    # masks = F.interpolate(masks, img0_shape[:2], mode='bilinear', align_corners=False)[0]
    # masks = masks.permute(1, 2, 0).contiguous()
    # masks_h, masks_w, n
    masks = cv2.resize(masks, (img0_shape[1], img0_shape[0]))

    # keepdim
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks

def mask_to_polygons(mask: np.ndarray):
    """
    Converts a binary mask to a list of polygons.

    Parameters:
        mask (np.ndarray): A binary mask represented as a 2D NumPy array of
            shape `(H, W)`, where H and W are the height and width of
            the mask, respectively.

    Returns:
        List[np.ndarray]: A list of polygons, where each polygon is represented by a
            NumPy array of shape `(N, 2)`, containing the `x`, `y` coordinates
            of the points. Polygons with fewer points than `MIN_POLYGON_POINT_COUNT = 3`
            are excluded from the output.
    """

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return [
        np.squeeze(contour, axis=1)
        for contour in contours
        if contour.shape[0] >= 5
    ]

def create_iou_matrix(n_true, n_pred, p_true, p_pred, idx_validation, iou_validation):
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            polygon_true = make_valid(Polygon(p_true[i]))
            polygon_pred = make_valid(Polygon(p_pred[j]))
            # print(i, j)
            polygon_intersection = polygon_true.intersection(polygon_pred).area
            polygon_union = polygon_true.union(polygon_pred).area
            iou_matrix[i, j] = polygon_intersection / polygon_union 
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)
    idx_validation.append([idxs_true, idxs_pred])
    iou_validation.append(iou_matrix)
    # ious_actual = iou_matrix[idxs_true, idxs_pred]
    return idx_validation, iou_validation



def pixel_to_coord(name_files, predictions, path_to_transform, state, save_loc_coord='delta_geocords', split='all', class_cls='all', save=False):
    """
        Convert and SAVE GEOCOORDS IN A GEOPANDAS FILE
        names_files: clean name of the images e.g delta_000_01
        predictions: Location of saved points in npz format, this was done on the "pred_nn" notebook. I have them divided in the TP preds, FN, and FP, three files per image
        path_to_transform: location of the txt file contaiin the transformation pixel to geocords
        state: name of the state, e.g delta or ogun or kwara
        save_loc_coord: where to save the shape file
        split: valid, test or train split, can be all
        class_cls: TP, FN, FP, can be all

        Output:
            a geopandas dataframe
            save a shape file containing the polygon for the split, eg. all the polygons of the test data set

    """
    polygon_s = []
    indexs = []
    p_conf = []
    for ix, pp in enumerate(predictions):
        txt_file = glob.glob('/'.join(pp.split('/')[:-1])+"/*.txt")
        # print(f"TXT FILE PRINT")
        # print(txt_file)
        with open(txt_file[0], 'r') as f:
            lines = f.readlines()
            f.close()
        # print(lines)
        # name = ('_').join(pp.split('/')[-1].split('_')[2:4]) #ogun
        #name = name_files[ix]#('_').join(pp.split('/')[-1].split('_')[2:5]) #delta
        name = ("_").join(pp.split("/")[-1].split(".")[0].split("_")[1:])
        # print(name)
        for ppt in path_to_transform:
            if name+'_' in ppt.split('/')[-1]:
                #p_conf = []
                segmentations = []
                points = np.load(pp)
                transform, CRS_type = utils_pixel_to_geo.load_transform_from_txt(ppt)
                for k in points:
                    poly = [(j[0].item(), j[1].item()) for j in points[k]]
                    xs = np.array(poly)[:, 1]
                    ys = np.array(poly)[:, 0]
                    xs_tmp, ys_tmp = rasterio.transform.xy(transform=transform,
                                                                    rows=xs,
                                                                    cols=ys)
        
                    segmentations.append(np.array([xs_tmp, ys_tmp]))
    
                
                for idx, polygon in enumerate(segmentations):
                    polygon_s.append(Polygon([(segmentations[idx][0][k], segmentations[idx][1][k]) for k in range(len(segmentations[idx][0]))]))
                    indexs.append(name+'_'+str(idx))
                    values = [float(x) for x in lines[idx].split(' ')]
                    p_conf.append(values[-1])
    # print(p_conf)
    # dest = gpd.GeoDataFrame(columns=['id', 'split', 'class', 'feature', 'conf'], geometry='feature', crs=CRS_type)
    d = {'ids': indexs, 'geometry': polygon_s, 'conf': p_conf}
    gdf = gpd.GeoDataFrame(d, crs=CRS_type)
    gdf['split'] = split#'valid'
    gdf['class'] = class_cls#'TP'
    # state = delta3
    if save == True:
        gdf.to_file(os.path.join(save_loc_coord, f"{state}_{class_cls}_geocoords.shp"))

    return gdf
