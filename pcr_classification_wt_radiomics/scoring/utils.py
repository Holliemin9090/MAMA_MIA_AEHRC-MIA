import os
import sys
# Add the parent directory to the Python path
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.abspath(os.path.join(current_folder, '..'))
sys.path.append(parent_folder_path)
import argparse
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
import json
from scoring.scoring_task2 import calculated_scoring_task2
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")

def calculate_sen_spe_tpr_fpr(predict, true_cate):
    predict = np.asarray(predict).astype(int)
    true_cate = np.asarray(true_cate).astype(int)

    TP = np.sum((predict == 1) & (true_cate == 1))
    TN = np.sum((predict == 0) & (true_cate == 0))
    FP = np.sum((predict == 1) & (true_cate == 0))
    FN = np.sum((predict == 0) & (true_cate == 1))

    # Avoid division by zero
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    tpr = sensitivity
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

    return sensitivity, specificity, tpr, fpr, accuracy

def find_threshold_minimum_distance (tprs, fprs, thresholds, tpr_threshold = -1):
    # minimum distance
    
    if tpr_threshold <= 0:
        metric = np.sqrt((1 - tprs) ** 2 + fprs ** 2)
        min_value = np.min(metric)
        min_indices = np.where(metric == min_value)[0]

        # Pick the index with the maximum threshold among them
        thresholds_at_min = thresholds[min_indices]
        max_thresh_idx = min_indices[np.argmax(thresholds_at_min)]

        return thresholds[max_thresh_idx], tprs[max_thresh_idx], fprs[max_thresh_idx], min_value
    
    else:
        
        tpr_ = tprs[tprs>=tpr_threshold]
        fpr_ = fprs[tprs>=tpr_threshold]
        threshold_ = thresholds[tprs>=tpr_threshold]
        metric = np.sqrt((1-tpr_)**2 + fpr_**2)
        index = np.argmin(metric)
        min_value = np.min(metric)  # Find the minimum value in the metric array
        
        return threshold_[index], tpr_[index], fpr_[index], min_value
    
def majority_voting (all_preds, thresholds):
    '''
    all_preds: list of predictions from each fold (5 lists in on list)
    thresholds: list of thresholds from each fold
    '''
    all_binary_preds = []
    for i in range(len(thresholds)):
        all_binary_preds.append([1 if pred >= thresholds[i] else 0 for pred in all_preds[i]])

    all_binary_preds = np.array(all_binary_preds)
    majority_voting_prob = np.average(all_binary_preds, axis=0) >= 0.5
    majority_voting_pred = majority_voting_prob >= 0.5
    return majority_voting_pred, majority_voting_prob

def get_val_clinical_info_df(clinical_df, val_names, all_new_image_names):
    # Convert to list if not already
    all_new_image_names = list(all_new_image_names)
    
    # Get the indices of val_names in all_new_image_names
    val_indices = [all_new_image_names.index(name) for name in val_names if name in all_new_image_names]
    
    # Extract corresponding rows from clinical_df
    val_clinical_df = clinical_df.iloc[val_indices].reset_index(drop=True)
    
    return val_clinical_df

def get_val_clinical_info_df_new(clinical_df, val_names):
    clinical_df_indexed = clinical_df.set_index('patient_id')
    val_clinical_df = clinical_df_indexed.loc[[name for name in val_names if name in clinical_df_indexed.index]]
    val_clinical_df = val_clinical_df.reset_index()
    return val_clinical_df


def compute_ordinal_kinetics(I, sentinel=-1.0):
    if not isinstance(I, (list, np.ndarray)) or len(I) < 2:
        return {
            "MaxEnhancement": sentinel,
            "TimeToPeak": sentinel,
            "WashInStep": sentinel,
            "WashOutStep": sentinel,
            "EarlyLateRatio": sentinel,
            "AUC": sentinel,
            "SER": sentinel,
        }

    pre = I[0]
    posts = I[1:]

    if pre == 0 or len(posts) == 0:
        return {
            "MaxEnhancement": sentinel,
            "TimeToPeak": sentinel,
            "WashInStep": sentinel,
            "WashOutStep": sentinel,
            "EarlyLateRatio": sentinel,
            "AUC": sentinel,
            "SER": sentinel,
        }

    max_enh = (max(posts) - pre) / pre if pre != 0 else sentinel
    time_to_peak = np.argmax([v - pre for v in posts]) / len(I)

    wash_in = (I[1] - pre) / pre if len(I) > 1 and pre != 0 else sentinel
    wash_out = (I[1] - I[-1]) / pre if len(I) > 2 and pre != 0 else sentinel
    early_late_ratio = I[-1] / I[1] if len(I) > 2 and I[1] != 0 else sentinel
    auc = (sum(posts) - pre) / pre * len(posts) if pre != 0 else sentinel
    ser_denom = I[-1] - pre
    ser = (I[1] - pre) / ser_denom if len(I) > 2 and ser_denom != 0 else sentinel

    return {
        "MaxEnhancement": float(max_enh),
        "TimeToPeak": float(time_to_peak),
        "WashInStep": float(wash_in),
        "WashOutStep": float(wash_out),
        "EarlyLateRatio": float(early_late_ratio),
        "AUC": float(auc),
        "SER": float(ser),
    }


def get_features(clinical_df, infor_json):
    case_names = clinical_df['patient_id'].tolist()
    all_features_pcr = []
    case_name_used = []
    for case_name in case_names:
        pcr_value = clinical_df.loc[clinical_df['patient_id'] == case_name, 'pcr'].values[0]
        data = infor_json[case_name]

        if pcr_value in [0, 1]:
            kinetics = compute_ordinal_kinetics(data['TumorIntensities'])
            feature_dict = {
                            "patient_id": case_name, 
                            "pcr_label": float(pcr_value),
                            "TumorRadius_mm": data["TumorRadius_mm"],
                            }
            
            # add all kinetic features
            feature_dict.update(kinetics)

            all_features_pcr.append(feature_dict)

            case_name_used.append(case_name)
            
    return all_features_pcr, case_name_used

def get_features_radiomics(clinical_df, radiomics_df):
    case_names = clinical_df['patient_id'].tolist()
    all_features_pcr = []
    case_name_used = []
    for case_name in case_names:
        pcr_value = clinical_df.loc[clinical_df['patient_id'] == case_name, 'pcr'].values[0]

        if pcr_value in [0, 1]:
            rad_row = radiomics_df[radiomics_df['CaseID'] == case_name]
           
            feature_dict = rad_row.iloc[0].to_dict()
            feature_dict['pcr'] = pcr_value  # optionally include the label

            all_features_pcr.append(feature_dict)
            case_name_used.append(case_name)

    feature_df = pd.DataFrame(all_features_pcr)
    return feature_df, case_name_used

def get_features_radiomics_wt_prediction_mask(clinical_df, radiomics_df):
    case_names = clinical_df['patient_id'].tolist()
    all_features_pcr = []
    case_name_used = []

    for case_name in case_names:
        pcr_value = clinical_df.loc[clinical_df['patient_id'] == case_name, 'pcr'].values[0]

        if pcr_value in [0, 1]:
            rad_row = radiomics_df[radiomics_df['CaseID'] == case_name]

            if not rad_row.empty:
                dice_score = rad_row['dice_score'].values[0]
                if dice_score > 0.4:
                    feature_dict = rad_row.iloc[0].to_dict()
                    feature_dict.pop('dice_score', None)  # remove dice_score
                    feature_dict['pcr'] = pcr_value        # optionally include the label

                    all_features_pcr.append(feature_dict)
                    case_name_used.append(case_name)

    feature_df = pd.DataFrame(all_features_pcr)
    return feature_df, case_name_used


def find_best_threshold_balanced_accuracy(y_true, y_proba, num_thresholds=100):
    """
    Find the threshold that maximizes balanced accuracy for binary classification.

    Parameters:
    - y_true (array-like): True binary labels (0 or 1).
    - y_proba (array-like): Predicted probabilities for the positive class.
    - num_thresholds (int): Number of threshold points to evaluate (default: 100).

    Returns:
    - best_threshold (float): Threshold that gives the highest balanced accuracy.
    - best_score (float): Best balanced accuracy score.
    - thresholds (np.ndarray): All tested thresholds.
    - scores (np.ndarray): Balanced accuracy scores for each threshold.
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    scores = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = balanced_accuracy_score(y_true, y_pred)
        scores.append(score)

    scores = np.array(scores)
    best_idx = scores.argmax()
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]

    return best_threshold, best_score, thresholds, scores