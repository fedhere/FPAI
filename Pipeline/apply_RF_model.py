import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import ensemble

import joblib

def load_apply_model(model_name, seg_predictions):
    loaded_rf = joblib.load(model_name)
    X = seg_predictions[loaded_rf.feature_names_in_]

    return loaded_rf.predict(X)
    
def apply_rf(seg_predictions_path='', use_state=None, path_to_rf_model=''):
    seg_predictions = gpd.read_file(seg_predictions_path)
    if use_state == None:
        use_state = [True, False]
        for i in range(len(use_state)):
            if use_state[i] == True:
                print('with state')
                model_name = "Tatiana/fishpond_rf_model"
                seg_predictions.loc[:, f'pred_wstate'] = load_apply_model(model_name, seg_predictions)
            else:
                model_name = "Tatiana/fishpond_rf_model_no_index_states"
                seg_predictions.loc[:, f'pred_wostate'] = load_apply_model(model_name, seg_predictions)
    else:
        if use_state:
            print('with state')
            model_name = "Tatiana/fishpond_rf_model"
            seg_predictions.loc[:, f'pred_wstate'] = load_apply_model(model_name, seg_predictions)
        else:
            model_name = "Tatiana/fishpond_rf_model_no_index_states"
            seg_predictions.loc[:, f'pred_wostate'] = load_apply_model(model_name, seg_predictions)

    # loaded_rf = joblib.load(path_to_rf_model)
    # seg_predictions = gpd.read_file(seg_predictions_path)
    
    # X = seg_predictions[rf2.feature_names_in_]
    
    # seg_predictions.loc[:, f'pred_wstate{use_state}'] = loaded_rf.predict(X)
    
    # path_to_save = os.path.join(DATA_PATH, f"{STATE}/{STATE}_inter_all_geocoords_wpred.shp")
    
    seg_predictions.to_file(seg_predictions_path.replace('_wfeatures', '_wpred'))