import os
import sys
# Add the parent directory to the Python path
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.abspath(os.path.join(current_folder, '..'))
sys.path.append(parent_folder_path)
import json
import argparse
import SimpleITK as sitk
import numpy as np
from natsort import natsorted
import pandas as pd
import seaborn as sns
import joblib
import shutil
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, PredefinedSplit
from scoring.scoring_task2 import calculated_scoring_task2
from scoring.utils import (find_threshold_minimum_distance, 
                           calculate_sen_spe_tpr_fpr, 
                           get_val_clinical_info_df_new,
                           majority_voting, get_features_radiomics, find_best_threshold_balanced_accuracy, get_features_radiomics_wt_prediction_mask)

if __name__ == '__main__':

    root = 'root path'
    data_partition_path = os.path.join(root, 'Data_preparation', 'Segmentation', 'cv_fold_splits.xlsx')
    image_path = os.path.join(root, 'dataset', 'images')
    mask_path = os.path.join(root,  'dataset','segmentations', 'expert')
    patient_info_path = os.path.join(root, 'dataset', 'patient_info_files')
    
    # Load JSON
    csv_path = "path to radiomic features/subset_radiomics_features_0.5.csv"# path to radiomic features extracted using the ground truth masks
    radiomics_df = pd.read_csv(csv_path)

    non_feature_cols = ['CaseID', 'pcr']
    num_features = len([col for col in radiomics_df.columns if col not in non_feature_cols])
    print(f"Number of final feature columns: {num_features}") #2237

    '''
    Get the radiomics features extracted using the prediction mask
    '''
    subset_csv_using_pred_mask_path = "path to radiomic features extracted using prediction masks/subset_radiomics_features_using_pred_mask.csv"
    subset_csv_using_pred_mask_df = pd.read_csv(subset_csv_using_pred_mask_path)

    all_train_AUCs = []
    all_val_AUCs = []
    all_val_gt = []
    all_val_pred_prob = []
    
    all_thresholds_voting = []
    all_val_names = []

    threshold_info = {}

    model_save_path = os.path.join('path to model save directory', 'radiomics_gt_and_pred')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    # else:
    #     shutil.rmtree(model_save_path)
    #     os.makedirs(model_save_path, exist_ok=True)
    
    for fold in range(0,5):
        train_clinical_df = pd.read_excel(data_partition_path, sheet_name='train_fold_'+str(fold))
        val_clinical_df = pd.read_excel(data_partition_path, sheet_name='val_fold_'+str(fold))
        training_data, training_cases = get_features_radiomics(train_clinical_df, radiomics_df)
        print(f"Fold {fold} - training_data using gt, shape: {training_data.shape}")
   
        # Add radiomics features extracted using the prediction mask
        training_data_wt_prediction_mask, training_cases_wt_prediction_mask = get_features_radiomics_wt_prediction_mask(train_clinical_df, subset_csv_using_pred_mask_df)
        print(f"Fold {fold} - training_data using prediction mask, shape: {training_data_wt_prediction_mask.shape}")

        training_data = pd.concat([training_data, training_data_wt_prediction_mask], ignore_index=True)
        training_cases = training_cases + training_cases_wt_prediction_mask
        print(f"Fold {fold} - combined training_data shape: {training_data.shape}")

        # Find and fill NaN values
        training_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # nan_counts = training_data.isna().sum()
        # print(nan_counts[nan_counts > 0])  # Print only columns that have NaNs
        training_data.fillna(-1, inplace=True)

        validation_data, validation_cases = get_features_radiomics(val_clinical_df, radiomics_df)
        print(f"Fold {fold} - validation_data using gt, shape: {validation_data.shape}")

        # Add radiomics features extracted using the prediction mask
        validation_data_wt_prediction_mask, validation_cases_wt_prediction_mask = get_features_radiomics_wt_prediction_mask(val_clinical_df, subset_csv_using_pred_mask_df)
        print(f"Fold {fold} - validation_data using prediction mask, shape: {validation_data_wt_prediction_mask.shape}")

        validation_data = pd.concat([validation_data, validation_data_wt_prediction_mask], ignore_index=True)
        validation_cases = validation_cases + validation_cases_wt_prediction_mask
        print(f"Fold {fold} - combined validation_data shape: {validation_data.shape}")

        validation_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        validation_data.fillna(-1, inplace=True)
 
        # Combine training and validation data for all folds
        combined_data = pd.concat([training_data, validation_data], ignore_index=True)

        # Create test_fold array: -1 for training, 0 for validation
        # Example: 0 for validation rows, -1 for training rows
        test_fold = np.array([-1]*len(training_data) + [0]*len(validation_data))
        ps = PredefinedSplit(test_fold)

        # Prepare X and y
        X = combined_data.drop(columns=['CaseID', 'pcr'])
        y = combined_data['pcr']

        # Compute scale_pos_weight for XGBoost
        neg = sum(y == 0)
        pos = sum(y == 1)
        scale = neg / pos

        # Define pipeline randam forest classifier, average auc around 0.614
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('feature_selection', SelectKBest(score_func=f_classif, k=500)),
            ('classifier', RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1))
        ])

        # Run evaluation
        scores = cross_val_score(pipeline, X, y, cv=ps, scoring='roc_auc')
        print(f"AUC Scores: {scores}")

        '''
        Select top features based on the pipeline's feature selection
        '''

        # # Fit the pipeline to the combined data (train+val)
        # pipeline.fit(X, y)

        # # Get feature selector and classifier from the pipeline
        # selector = pipeline.named_steps['feature_selection']
        # classifier = pipeline.named_steps['classifier']

        # # Get scores and selected indices
        # feature_scores = selector.scores_
        # selected_indices = selector.get_support(indices=True)
        # selected_feature_names = X.columns[selected_indices]
        # selected_feature_scores = feature_scores[selected_indices]

        # # Sort features by score (descending)
        # sorted_features = sorted(zip(selected_feature_names, selected_feature_scores), key=lambda x: x[1], reverse=True)

        # # Print top 20 for inspection
        # # print(f"\nTop 500 features for fold {fold}:")
        # # for name, score in sorted_features[:500]:
        # #     print(f"{name}: {score:.4f}")

        # # Optionally: store them for later
        # with open(os.path.join(model_save_path, f'top_features_fold{fold}.txt'), 'w') as f:
        #     for name, score in sorted_features:
        #         f.write(f"{name}\t{score:.4f}\n")

        '''
        Apply the trained pipeline to training and validation data
        '''
        train_X=training_data.drop(columns=['CaseID', 'pcr'])
        train_X.fillna(-1, inplace=True)
        train_y=training_data['pcr']

        pipeline.fit(train_X, train_y)

        # Save the model for the current fold
        model_save_path_fold = os.path.join(model_save_path, f'model_fold_{fold}.joblib')
        joblib.dump(pipeline, model_save_path_fold)

        print(f"Saved model for fold {fold} to {model_save_path_fold}")

        train_y_prob = pipeline.predict_proba(train_X)[:, 1]
        train_auc = roc_auc_score(train_y, train_y_prob)
        all_train_AUCs.append(train_auc)
        print("Training AUC:", train_auc)

        val_X = validation_data.drop(columns=['CaseID', 'pcr'])
        val_X.fillna(-1, inplace=True)
        val_y = validation_data['pcr']
        val_y_prob = pipeline.predict_proba(val_X)[:, 1]

        val_fprs, val_tprs, val_thresholds = roc_curve(val_y, val_y_prob)
        val_roc_auc = auc(val_fprs, val_tprs)
        all_val_AUCs.append(val_roc_auc)
        print("Validation AUC:", val_roc_auc)

        threshold, optimal_tpr, optimal_fpr, min_distance = find_threshold_minimum_distance(val_tprs, val_fprs, val_thresholds)
        # threshold, best_score,_ , scores = find_best_threshold_balanced_accuracy(val_y, val_y_prob, num_thresholds=100)
        all_thresholds_voting.append(threshold)

        all_val_gt.extend(val_y)
        all_val_pred_prob.extend(val_y_prob)
        all_val_names.extend(validation_cases)

    print("Average training AUC:", np.mean(all_train_AUCs))
    print("Average validation AUC:", np.mean(all_val_AUCs))

    ensemble_val_fprs, ensemble_val_tprs, ensemble_val_thresholds = roc_curve(all_val_gt, all_val_pred_prob)
    ensemble_val_roc_auc = auc(ensemble_val_fprs, ensemble_val_tprs)
    ensemble_threshold, optimal_val_tpr, optimal_val_fpr, ensemble_min_distance = find_threshold_minimum_distance(ensemble_val_tprs, ensemble_val_fprs, ensemble_val_thresholds)

    combined_clinical_df = pd.concat([train_clinical_df, val_clinical_df], ignore_index=True)
    all_val_clinical_df = get_val_clinical_info_df_new(combined_clinical_df, all_val_names)

    # Save threshold information
    # ensemble_threshold = 0.34
    ensemble_val_pred = [1 if pred > ensemble_threshold else 0 for pred in all_val_pred_prob]

    ensemble_sensitivity, ensemble_specificity, ensemble_tpr, ensemble_fpr, ensemble_accuracy = calculate_sen_spe_tpr_fpr(
            ensemble_val_pred, all_val_gt)
    print (f"Ensemble Sensitivity: {ensemble_sensitivity}, Specificity: {ensemble_specificity}, TPR: {ensemble_tpr}, FPR: {ensemble_fpr}, Accuracy: {ensemble_accuracy}")
    
    balanced_accuracy, fairness_score, ranking_score = calculated_scoring_task2(
            ensemble_val_pred, all_val_pred_prob, all_val_clinical_df, model_save_path)

    final_pred_df = all_val_clinical_df.copy()
    final_pred_df['pcr_pred'] = ensemble_val_pred
    final_pred_df['pcr_prob'] = all_val_pred_prob
    prediction_excel_path = os.path.join(model_save_path, 'all_val_prediction_df.xlsx')
    final_pred_df.to_excel(prediction_excel_path, index=False)

    # val_voting_pred, majority_voting (all_val_pred_prob_voting, all_thresholds_voting)
    threshold_info['ensemble_threshold'] = ensemble_threshold
    threshold_info['majority_voting_thresholds'] = all_thresholds_voting

    # save the threshold information
    threshold_info_path = os.path.join(model_save_path, 'threshold_info.json')
    with open(threshold_info_path, 'w') as f:
        json.dump(threshold_info, f, indent=4)
    print(f"Threshold information saved to {threshold_info_path}")


    # '''
    # # Find out if offets are needed for the ensemble threshold
    # '''
    
    # # Store metrics at different thresholds
    # threshold_offsets = []
    # metric_records = []

    # # Original optimal threshold (base)
    # base_threshold = ensemble_threshold

    # for offset in np.arange(0.01, 0.31, 0.01):
    #     # Adjust threshold
    #     current_threshold = base_threshold + offset

    #     # Get binary predictions
    #     ensemble_val_pred = [1 if pred > current_threshold else 0 for pred in all_val_pred_prob]

    #     # Compute core metrics
    #     ensemble_sensitivity, ensemble_specificity, ensemble_tpr, ensemble_fpr, ensemble_accuracy = calculate_sen_spe_tpr_fpr(
    #         ensemble_val_pred, all_val_gt)

    #     # Get additional task-specific metrics
    #     balanced_accuracy, fairness_score, ranking_score = calculated_scoring_task2(
    #         ensemble_val_pred, all_val_pred_prob, all_val_clinical_df, model_save_path)

    #     # Store metrics
    #     metric_records.append({
    #         'threshold_offset': round(offset, 2),
    #         'adjusted_threshold': current_threshold,
    #         'sensitivity': ensemble_sensitivity,
    #         'specificity': ensemble_specificity,
    #         'TPR': ensemble_tpr,
    #         'FPR': ensemble_fpr,
    #         'accuracy': ensemble_accuracy,
    #         'balanced_accuracy': balanced_accuracy,
    #         'fairness_score': fairness_score,
    #         'ranking_score': ranking_score
    #     })

    # # Convert to DataFrame
    # df_metrics = pd.DataFrame(metric_records)

    # # Save to Excel
    # excel_save_path = os.path.join(model_save_path, 'threshold_offset_metrics.xlsx')
    # df_metrics.to_excel(excel_save_path, index=False)
    # print(f"Saved threshold offset metrics to: {excel_save_path}")

    


        
