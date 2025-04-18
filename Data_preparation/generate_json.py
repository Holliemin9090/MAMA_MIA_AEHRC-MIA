# This file is to generate json file for nnunet model training
'''
This is following nnunet v1 format as required by mednext repo
'''
import os
import json
from natsort import natsorted

if __name__ == '__main__':
# this is an existing example 
    
    dataset = dict()
    if_DCE_or_DWIADC = 'DCE'
    
    if if_DCE_or_DWIADC=='DCE':
        
        train_label_path = '/Volumes/{hb-breast-nac-challenge}/work/MedNext_dataset/nnUNet_raw_data_base/nnUNet_raw_data/Task001_breastDCE/labelsTr/'
        train_image_paht = '/Volumes/{hb-breast-nac-challenge}/work/MedNext_dataset/nnUNet_raw_data_base/nnUNet_raw_data/Task001_breastDCE/imagesTr/'
        train_case_names = natsorted(os.listdir(train_label_path))
        train_case_names = [name for name in train_case_names if name.endswith('.nii.gz')]
        print(len(train_case_names))
        train_set = []
        for train_case_name in train_case_names:
            train_set.append({"image": "./imagesTr/" + train_case_name, "label": "./labelsTr/" + train_case_name})
            
        # test_image_path = 'Y:/work/breast_cancer_dwi_project/MedNeXt_venv/dataset/nnUNet_raw_data_base/nnUNet_raw_data/Task003_breastDCE/labelsTs/'
        # test_image_names = natsorted(os.listdir(test_image_path))
        # test_set = []
        # for test_image_name in test_image_names:
        #     test_set.append("./imagesTs/" + test_image_name)
        # https://github.com/MIC-DKFZ/MedNeXt/blob/main/documentation/dataset_conversion.md
        # The mednext is based on nnunetv1 at https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1
        dataset = {"name": "Breast_DCE",
                   "description": "Breast tumour segmentation on DCE-MRI",
                   "reference": "Australian E-Health research centre, CSIRO",
                   "licence":"CC-BY-SA 4.0",
                   "relase":"1.0 15/04/2025",
                   "tensorImageSize": "4D",
                   "modality": {"0": "DCE0_0",
                                "1": "DCE1_0",
                                "2": "DCE2_0",
                                            },                
                   "labels": {"0": "background", 
                              "1": "tomour"},                
                   "numTraining": 1506,                
                   "numTest": 0,                   
                   "training": train_set,
                   "test": []                
                   }

            
        with open("/Volumes/{hb-breast-nac-challenge}/work/MedNext_dataset/nnUNet_raw_data_base/nnUNet_raw_data/Task001_breastDCE/dataset.json", "w") as outfile:
            json.dump(dataset, outfile)
            
   
        
        