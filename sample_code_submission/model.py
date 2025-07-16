# ==================== MAMA-MIA CHALLENGE SAMPLE SUBMISSION ====================
#
# This is the official sample submission script for the **MAMA-MIA Challenge**, 
# covering both tasks:
#
#   1. Primary Tumour Segmentation (Task 1)
#   2. Treatment Response Classification (Task 2)
#
# ----------------------------- SUBMISSION FORMAT -----------------------------
# Participants must implement a class `Model` with one or two of these methods:
#
#   - `predict_segmentation(output_dir)`: required for Task 1
#       > Must output NIfTI files named `{patient_id}.nii.gz` in a folder
#       > called `pred_segmentations/`
#
#   - `predict_classification(output_dir)`: required for Task 2
#       > Must output a CSV file `predictions.csv` in `output_dir` with columns:
#           - `patient_id`: patient identifier
#           - `pcr`: binary label (1 = pCR, 0 = non-pCR)
#           - `score`: predicted probability (flaot between 0 and 1)
#
#   - `predict_classification(output_dir)`: if a single model handles both tasks
#       > Must output NIfTI files named `{patient_id}.nii.gz` in a folder
#       > called `pred_segmentations/`
#       > Must output a CSV file `predictions.csv` in `output_dir` with columns:
#           - `patient_id`: patient identifier
#           - `pcr`: binary label (1 = pCR, 0 = non-pCR)
#           - `score`: predicted probability (flaot between 0 and 1)
#
# You can submit:
#   - Only Task 1 (implement `predict_segmentation`)
#   - Only Task 2 (implement `predict_classification`)
#   - Both Tasks (implement both methods independently or define `predict_segmentation_and_classification` method)
#
# ------------------------ SANITY-CHECK PHASE ------------------------
#
# ✅ Before entering the validation or test phases, participants must pass the **Sanity-Check phase**.
#   - This phase uses **4 samples from the test set** to ensure your submission pipeline runs correctly.
#   - Submissions in this phase are **not scored**, but must complete successfully within the **20-minute timeout limit**.
#   - Use this phase to debug your pipeline and verify output formats without impacting your submission quota.
#
# 💡 This helps avoid wasted submissions on later phases due to technical errors.
#
# ------------------------ SUBMISSION LIMITATIONS ------------------------
#
# ⚠️ Submission limits are strictly enforced per team:
#   - **One submission per day**
#   - **Up to 15 submissions total on the validation set**
#   - **Only 1 final submission on the test set**
#
# Plan your development and testing accordingly to avoid exhausting submissions prematurely.
#
# ----------------------------- RUNTIME AND RESOURCES -----------------------------
#
# > ⚠️ VERY IMPORTANT: Each image has a **timeout of 5 minutes** on the compute worker.
#   - **Validation Set**: 58 patients → total budget ≈ 290 minutes
#   - **Test Set**: 516 patients → total budget ≈ 2580 minutes
#
# > The compute worker environment is based on the Docker image:
#       `lgarrucho/codabench-gpu:latest`
#
# > You can install additional dependencies via `requirements.txt`.
#   Please ensure all required packages are listed there.
#
# ----------------------------- SEGMENTATION DETAILS -----------------------------
#
# This example uses `nnUNet v2`, which is compatible with the GPU compute worker.
# Note the following nnUNet-specific constraints:
#
# ✅ `predict_from_files_sequential` MUST be used for inference.
#     - This is because nnUNet’s multiprocessing is incompatible with the compute container.
#     - In our environment, a single fold prediction using `predict_from_files_sequential` 
#       takes approximately **1 minute per patient**.
#
# ✅ The model uses **fold 0 only** to reduce runtime.
# 
# ✅ Predictions are post-processed by applying a breast bounding box mask using 
#    metadata provided in the per-patient JSON file.
#
# ----------------------------- CLASSIFICATION DETAILS -----------------------------
#
# If using predicted segmentations for Task 2 classification:
#   - Save them in `self.predicted_segmentations` inside `predict_segmentation()`
#   - You can reuse them in `predict_classification()`
#   - Or perform Task 1 and Task 2 inside `predict_segmentation_and_classification`
#
# ----------------------------- DATASET INTERFACE -----------------------------
# The provided `dataset` object is a `RestrictedDataset` instance and includes:
#
#   - `dataset.get_patient_id_list() → list[str]`  
#         Patient IDs for current split (val/test)
#
#   - `dataset.get_dce_mri_path_list(patient_id) → list[str]`  
#         Paths to all image channels (typically pre and post contrast)
#         - iamge_list[0] corresponds to the pre-contrast image path
#         - iamge_list[1] corresponds to the first post-contrast image path and so on
#
#   - `dataset.read_json_file(patient_id) → dict`  
#         Metadata dictionary per patient, including:
#         - breast bounding box (`primary_lesion.breast_coordinates`)
#         - scanner metadata (`imaging_data`), etc...
#
# Example JSON structure:
# {
#   "patient_id": "XXX_XXX_SXXXX",
#   "primary_lesion": {
#     "breast_coordinates": {
#         "x_min": 1, "x_max": 158,
#         "y_min": 6, "y_max": 276,
#         "z_min": 1, "z_max": 176
#     }
#   },
#   "imaging_data": {
#     "bilateral": true,
#     "dataset": "HOSPITAL_X",
#     "site": "HOSPITAL_X",
#     "scanner_manufacturer": "SIEMENS",
#     "scanner_model": "Aera",
#     "field_strength": 1.5,
#     "echo_time": 1.11,
#     "repetition_time": 3.35
#   }
# }
#
# ----------------------------- RECOMMENDATIONS -----------------------------
# ✅ We recommend to always test your submission first in the Sanity-Check Phase.
#    As in Codabench the phases need to be sequential and they cannot run in parallel,
#    we will open a secondary MAMA-MIA Challenge Codabench page with a permanen Sanity-Check phase.
#   That way you won't lose submission trials to the validation or even wore, the test set.
# ✅ We recommend testing your solution locally and measuring execution time per image.
# ✅ Use lightweight models or limit folds if running nnUNet.
# ✅ Keep all file paths, patient IDs, and formats **exactly** as specified.
# ✅ Ensure your output folders are created correctly (e.g. `pred_segmentations/`)
# ✅ For faster runtime, only select a single image for segmentation.
#
# ------------------------ COPYRIGHT ------------------------------------------
#
# © 2025 Lidia Garrucho. All rights reserved.
# Unauthorized use, reproduction, or distribution of any part of this competition's 
# materials is prohibited without explicit permission.
#
# ------------------------------------------------------------------------------

# === MANDATORY IMPORTS ===
import os
import importlib
import sys#!!!!! change for submission
sys.path.append(os.path.abspath("/app/ingested_program/sample_code_submission"))#!!!!! change for submission
sys.path.append("/app/program/src/mednextv1")

import numpy as np
import pandas as pd
import shutil

# === OPTIONAL IMPORTS: only needed if you modify or extend nnUNet input/output handling ===
# You can remove unused imports above if not needed for your solution

import torch
import SimpleITK as sitk
import joblib
# === Additional imports ===

from data_preprocess import (seg_task_012_preprocess)
from utils import (seg_restore_Task012, read_json, majority_voting)
from radiomics_feature_extraction import (initialize_radiomics_extractors, process_data_and_extract_radiomics)

from nnunet_mednext.inference.predict import predict_from_folder

import json

class Model:
    def __init__(self, dataset):
        """
        Initializes the model with the restricted dataset.
        
        Args:
            dataset (RestrictedDataset): Preloaded dataset instance with controlled access.
        """
        # MANDATOR
        self.dataset = dataset  # Restricted Access to Private Dataset
        self.predicted_segmentations = None  # Optional: stores path to predicted segmentations
        # DC+TopK loss model final checkpoint
        self.seg_model_folder = "/app/ingested_program/sample_code_submission/seg_model/nnUNetTrainerV2_MedNeXt_L_kernel5_lr_5e_6__nnUNetPlansv2.1_trgSp_1x1x1"
        # pyradiomics classification model folder: gt mask + pred mask
        self.cls_model_folder = "/app/ingested_program/sample_code_submission/pcr_model"
       
        self.pcr_threshold_mode = "ensemble"  # Use majority voting for classification thresholding or ensemble
        
    def predict_segmentation(self, output_dir):
        """
        Task 1 — Predict tumor segmentation with nnUNetv2.
        You MUST define this method if participating in Task 1.

        Args:
            output_dir (str): Directory where predictions will be stored.

        Returns:
            str: Path to folder with predicted segmentation masks.
        """

        # === Set required nnUNet paths ===
        # Not strictly mandatory if pre-set in Docker env, but avoids missing variable warnings
        #!!!!! change for submission
        os.environ['nnUNet_raw_data_base'] = "/app/ingested_program/sample_code_submission"
        os.environ['nnUNet_preprocessed'] = "/app/ingested_program/sample_code_submission"
        os.environ['RESULTS_FOLDER'] = "/app/ingested_program/sample_code_submission"
        
        # === 1 Data preparation ===
        # === Build nnUNet-compatible input images folder ===
        nnunet_input_images = os.path.join(output_dir, 'nnunet_input_images')
        if os.path.exists(nnunet_input_images):
            shutil.rmtree(nnunet_input_images)
        # Create the nnUNet input images folder
        os.makedirs(nnunet_input_images, exist_ok=True)
        # === Output folder for raw nnUNet segmentations ===
        output_dir_temp = os.path.join(output_dir, 'pred_segmentations_temp')
        if os.path.exists(output_dir_temp):
            shutil.rmtree(output_dir_temp)
        
        os.makedirs(output_dir_temp, exist_ok=True)
        # === Final output folder for segmentations ===
        output_dir_final = os.path.join(output_dir, 'pred_segmentations')
        if os.path.exists(output_dir_final):
            shutil.rmtree(output_dir_final)
        
        os.makedirs(output_dir_final, exist_ok=True)

        patient_ids = self.dataset.get_patient_id_list()
        for idx, patient_id in enumerate(patient_ids):
            images = self.dataset.get_dce_mri_path_list(patient_id)
            patient_info = self.dataset.read_json_file(patient_id)

            case_metrics = seg_task_012_preprocess(images, patient_id, nnunet_input_images)

        print('===============Data preparation completed!===================')

        # === Call MedNeXt prediction ===
        
        # nnunet_images = [[os.path.join(nnunet_input_images, f)] for f in os.listdir(nnunet_input_images)]
        # IMPORTANT: the only method that works inside the Docker container is predict_from_files_sequential
        # This method will predict all images in the list and save them in the output directory
        prediction = predict_from_folder(model = self.seg_model_folder,
                            input_folder = nnunet_input_images, 
                            output_folder = output_dir_temp, 
                            folds = [0,1,2,3,4], 
                            save_npz = False, # if probability map thresholding generate better results, this is set to True. but when mode is fast, this need to be False
                            num_threads_preprocessing = 6,
                            num_threads_nifti_save = 2, 
                            lowres_segmentations = None, 
                            part_id = 0, 
                            num_parts = 1, 
                            tta = 0,
                            overwrite_existing = True, 
                            mode = "normal",
                            overwrite_all_in_gpu = None,
                            mixed_precision = True,
                            step_size = 0.5, 
                            checkpoint_name = "model_final_checkpoint",# model_best or model_final_checkpoint
                            )#disable_postprocessing = True


        print("Initial Predictions saved to:", os.listdir(output_dir_temp))
        
       # === Postprocessing and generating Final output folder (MANDATORY name) ===
        
        for idx, patient_id in enumerate(patient_ids):
            # process the probability maps
            patient_info = self.dataset.read_json_file(patient_id)
            images = self.dataset.get_dce_mri_path_list(patient_id)
            seg_restore_Task012(images, patient_id, patient_info, output_dir_temp, output_dir_final, thresh=None)

        print("Segmentation task completed!")
        
        self.predicted_segmentations = output_dir_final

        return output_dir_final

    
    def predict_classification(self, output_dir):
        """
        Task 2 — Predict treatment response (pCR).
        You MUST define this method if participating in Task 2.

        Args:
            output_dir (str): Directory to save output predictions.

        Returns:
            pd.DataFrame: DataFrame with patient_id, pcr prediction, and score.
        """
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        patient_ids = self.dataset.get_patient_id_list()
        image_folder_path = os.path.join(output_dir, 'nnunet_input_images')
        seg_temp_folder_path = os.path.join(output_dir, 'pred_segmentations_temp')
        pcr_classification_input_path = os.path.join(output_dir, 'pcr_classification_input')
        output_seg_dir_final = os.path.join(output_dir, 'pred_segmentations')
        predictions = []
        testing_dataset = []

        for patient_id in patient_ids:
            images = self.dataset.get_dce_mri_path_list(patient_id)
            patient_info = self.dataset.read_json_file(patient_id)

            extractor_dce1, extractor_other_dce, extractor_periph = initialize_radiomics_extractors(bin_width=0.5)
            feature_dict = process_data_and_extract_radiomics(images, patient_info, patient_id, output_seg_dir_final, extractor_dce1, extractor_other_dce, extractor_periph)
            # metric = calculate_tumor_size_kinetic(images, patient_id, output_seg_dir_final)
            # kinetics = compute_ordinal_kinetics(metric['TumorIntensities'])
            # feature_dict = {
            #                 "patient_id": patient_id, 
            #                 "TumorRadius_mm": metric["TumorRadius_mm"],
            #                 }
            
            # add all kinetic features
            # feature_dict.update(kinetics)

            if feature_dict is None:
                # If the mask is empty, we cannot use this patient for classification
                print(f"⚠️  Skipping {patient_id}: tumor mask is empty.")
                
                continue
            else:
                testing_dataset.append(feature_dict)
        
        
        testing_dataset = pd.DataFrame(testing_dataset)
        test_X = testing_dataset.drop(columns=['patient_id'])

        test_X.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_X.fillna(-1, inplace=True)

        all_test_preds = []
        threshold_info = read_json(os.path.join(self.cls_model_folder, 'threshold_info.json'))

        if self.pcr_threshold_mode == "majority_voting":
            thresholds = threshold_info[ "majority_voting_thresholds"] 
        elif self.pcr_threshold_mode == "ensemble":
            thresholds = threshold_info[ "ensemble_threshold"]

        for fold in range(5):
            mdl_name = f"model_fold_{str(fold)}.joblib"  # Change this to your model name if needed
            model = joblib.load(os.path.join(self.cls_model_folder,mdl_name))
            test_pred = model.predict_proba(test_X)[:, 1]
            all_test_preds.append(test_pred)
            # using majority voting to get the final prediction 
        
        if self.pcr_threshold_mode == "majority_voting":
            preds, probs = majority_voting (all_test_preds, thresholds)
        elif self.pcr_threshold_mode == "ensemble":
            probs = np.mean(all_test_preds, axis=0)  # Average predictions across folds
            preds = [1 if pred >= thresholds else 0 for pred in probs]
        
            # === MANDATORY output format ===
        predictions_tmp = {
            "patient_id": testing_dataset['patient_id'].tolist(),
            "pcr": preds,
            "score": probs
        }
        prediction_df_tmp = pd.DataFrame(predictions_tmp)

        if prediction_df_tmp["patient_id"].tolist() != list(patient_ids):
            # There may be skipped patients due to empty masks
            prediction_dict_tmp = prediction_df_tmp.set_index('patient_id').to_dict(orient='index')

            final_prediction = []

            for patient_id in patient_ids:
                if patient_id in prediction_dict_tmp:
                    row = {
                        'patient_id': patient_id,
                        'pcr': prediction_dict_tmp[patient_id]['pcr'],
                        'score': prediction_dict_tmp[patient_id]['score']
                    }
                else:
                    row = {
                        'patient_id': patient_id,
                        'pcr': 0,
                        'score': 0.0
                    }
                final_prediction.append(row)

            prediction_df = pd.DataFrame(final_prediction)
        else:
            prediction_df = prediction_df_tmp
           
        prediction_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

        # Clean up temporary files and preprocessed image folder
        if os.path.exists(image_folder_path):
            shutil.rmtree(image_folder_path)
            
        if os.path.exists(seg_temp_folder_path):
            shutil.rmtree(seg_temp_folder_path)
        # shutil.rmtree(pcr_classification_input_path)

        return prediction_df

