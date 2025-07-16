# -*- coding: utf-8 -*-
"""
Create a 5-fold split for the breast DCE dataset.
"""

import pandas as pd
from natsort import natsorted
import os
import json
import pickle
from collections import OrderedDict
import numpy as np

if __name__ == '__main__':
    
    num_post_contrast_images = 90
    task_name = 'Task012_breastDCE'
   
    root = 'root directory path' 
    
    data_partition_path = 'cv_fold_splits.xlsx'
     
    new_case_path = os.path.join('root directory path/MedNext_dataset_python310/nnUNet_raw_data_base/nnUNet_raw_data',
                                 task_name, 'labelsTr')
    
    pickle_save_path = os.path.join('root directory path/MedNext_dataset_python310/nnUNet_preprocessed',
                                    task_name, 'splits_final.pkl')
    
    all_new_label_names = natsorted([
    name[:-7] for name in os.listdir(new_case_path) if name.endswith('.nii.gz')
    ])
    
    folds = [0, 1, 2, 3, 4]
    Splits = []
    
    for f, fold in enumerate(folds):
        
        train_case_names = pd.read_excel(data_partition_path, sheet_name='train_fold_' + str(fold))['patient_id'].tolist()
        val_case_names = pd.read_excel(data_partition_path, sheet_name='val_fold_' + str(fold))['patient_id'].tolist()
        
        train_IDs = []
        val_IDs = []
        if num_post_contrast_images > 1:
            for j in range(1, num_post_contrast_images+1):
                for case_name in train_case_names:
                    new_name = f"{case_name}_{str(j).zfill(2)}"
                    if new_name in all_new_label_names:
                        train_IDs.append(new_name)
                  
                # # use all augmented validation cases
                # for case_name in val_case_names:
                #     new_name = f"{case_name}_{str(j).zfill(2)}"
                #     if new_name in all_new_label_names:
                #         val_IDs.append(new_name)

            # use only the first postcontrast validation cases
            for case_name in val_case_names:
                new_name = f"{case_name}_{str(1).zfill(2)}"
                if new_name in all_new_label_names:
                    val_IDs.append(new_name)
                    
        else:
            for case_name in train_case_names:
                if case_name in all_new_label_names:
                    train_IDs.append(case_name)

            for case_name in val_case_names:
                if case_name in all_new_label_names:
                    val_IDs.append(case_name)

        # if natsorted(train_IDs+val_IDs) != all_new_label_names:
        #     raise ValueError("The train and validation IDs do not match the expected new label names.")
        
        if all(elem in all_new_label_names for elem in train_IDs+val_IDs):
            print("All elements of train and validation IDs are in all new label names.")
        else:
            print("Some elements of train and validation IDs are missing in all new label names.")

        print(f"Fold {f}: {len(train_IDs)} training cases, {len(val_IDs)} validation cases")

        split = {'train': np.array(train_IDs),
                 'val': np.array(val_IDs)}
        
        Splits.append(OrderedDict(split))
        
    # Pickle the data into a file
    with open(pickle_save_path, 'wb') as file:
        pickle.dump(Splits, file)

    # print(Splits)

    print(f"Successfully saved 5-fold split with {len(Splits)} folds to: {pickle_save_path}")


    