# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:18:28 2023

@author: min105

generate split json file
"""

import pandas as pd
from natsort import natsorted
import os
import json
import pickle
from collections import OrderedDict
import numpy as np

def find_new_image_label_name (case_names, all_case_names, all_new_label_names):
    IDs = []   
    for c, case_name in enumerate(case_names):
        index = all_case_names.index(case_name)        
        idx = all_new_label_names[index].find('.nii.gz')
        IDs.append(all_new_label_names[index][:idx]) 
                
    return IDs
        
if __name__ == '__main__':
    
   
    root = '/Volumes/{hb-breast-nac-challenge}/work' # '/Volumes/{hb-breast-nac-challenge}/work/min105/scripts/Data_preparation/generate_split_json.py'
    
    data_partition_path = os.path.join(root, 'min105', 'scripts', 'Data_preparation', 'cv_fold_splits.xlsx') 
     
    train_case_names_temp = pd.read_excel(data_partition_path, sheet_name='train_fold_0')['patient_id'].tolist()
    val_case_names_temp = pd.read_excel(data_partition_path, sheet_name='val_fold_0')['patient_id'].tolist()
    
    all_case_names = natsorted(train_case_names_temp + val_case_names_temp)
    
    new_case_path = '/Volumes/{hb-breast-nac-challenge}/work/MedNext_dataset/nnUNet_raw_data_base/nnUNet_raw_data/Task001_breastDCE/labelsTr'
    
    pickle_save_path = '/Volumes/{hb-breast-nac-challenge}/work/MedNext_dataset/nnUNet_preprocessed/Task001_breastDCE/splits_final.pkl'
        
    all_new_label_names = natsorted(os.listdir(new_case_path))
    
    folds = [0, 1, 2, 3, 4]
    Splits = []
    for f, fold in enumerate(folds):
        
        train_case_names = pd.read_excel(data_partition_path, sheet_name='train_fold_' + str(fold))['patient_id'].tolist()
        val_case_names = pd.read_excel(data_partition_path, sheet_name='val_fold_' + str(fold))['patient_id'].tolist()
        
        train_IDs = find_new_image_label_name (train_case_names, all_case_names, all_new_label_names) 
        print('number of train IDs:', len(train_IDs))

        val_IDs = find_new_image_label_name (val_case_names, all_case_names, all_new_label_names) 
        print('number of val IDs:', len(val_IDs))
        
        split = {'train': np.array(train_IDs),
                 'val': np.array(val_IDs)}
        
        Splits.append(OrderedDict(split))
        
    # Pickle the data into a file
    with open(pickle_save_path, 'wb') as file:
        pickle.dump(Splits, file)

    print(Splits)

    