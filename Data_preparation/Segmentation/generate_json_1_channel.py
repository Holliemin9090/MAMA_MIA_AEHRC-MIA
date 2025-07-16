# This file is to generate json file for nnunet model training
'''
This is following nnunet v1 format as required by mednext repo
'''
import os
import json
from natsort import natsorted

if __name__ == '__main__':

    task_name = 'Task012_breastDCE'
    # num_post_contrast_images = 2 # Number of post-contrast images to generate (1st and 2nd)
    if_normalization_done = False # True  # Whether to normalize the images using the average intensity of the pre-contrast image

    root = 'root directory path'
    json_output_path = os.path.join(root, 'MedNext_dataset_python310', 'nnUNet_raw_data_base', 'nnUNet_raw_data', task_name, 'dataset.json')
    train_label_path = os.path.join(root, 'MedNext_dataset_python310', 'nnUNet_raw_data_base', 'nnUNet_raw_data', task_name, 'labelsTr')
    train_image_path = os.path.join(root, 'MedNext_dataset_python310', 'nnUNet_raw_data_base', 'nnUNet_raw_data', task_name, 'imagesTr')
    
    train_case_names = natsorted(os.listdir(train_label_path))
    train_case_names = [name for name in train_case_names if name.endswith('.nii.gz')]
    print("Number of training cases:", len(train_case_names))

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
    dataset = dict()
    if if_normalization_done:
        # dataset has been normalized by the average intensity of the pre-contrast image
        dataset = {"name": "Breast_DCE",
                    "description": "Breast tumour segmentation on DCE-MRI",
                    "reference": "Australian E-Health research centre, CSIRO",
                    "licence":"CC-BY-SA 4.0",
                    "relase":"1.0 07/06/2025",
                    # "tensorImageSize": "3D",
                    "modality": {"0": "noNorm"},
                    "labels": {"0": "background", 
                               "1": "tumour"},                
                    "numTraining": len(train_case_names),#1506,                
                    "numTest": 0,                   
                    "training": train_set,
                    "test": []                
                    }
    else:
        dataset = {"name": "Breast_DCE",
                    "description": "Breast tumour segmentation on DCE-MRI",
                    "reference": "Australian E-Health research centre, CSIRO",
                    "licence":"CC-BY-SA 4.0",
                    "relase":"1.0 07/06/2025",
                    # "tensorImageSize": "3D",
                    "modality": {"0": "DCE"},
                    "labels": {"0": "background", 
                               "1": "tumour"},                
                    "numTraining": len(train_case_names),#1506,                
                    "numTest": 0,                   
                    "training": train_set,
                    "test": []                
                    }

        
    with open(json_output_path, "w") as outfile:
        json.dump(dataset, outfile)
            
   
        
        