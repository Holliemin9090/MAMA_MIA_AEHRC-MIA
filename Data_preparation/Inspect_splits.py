'''
Inspect the given training testing/validation split and generate more splits (ideal another 4 and stratified)
'''
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def inspect_stratification(train_df, test_df):
    # inspect the dataset, pcr, tumor_subtype stratification
    print("Tumor subtype distribution in TRAIN:")
    print(train_df["tumor_subtype"].value_counts(dropna=False))

    print("\nTumor subtype distribution in TEST:")
    print(test_df["tumor_subtype"].value_counts(dropna=False))

    print("pCR distribution in training set:")
    print(train_df['pcr'].value_counts(dropna=False))

    print("\npCR distribution in test set:")
    print(test_df['pcr'].value_counts(dropna=False))

    print("Dataset distribution in TRAIN:")
    print(train_df["dataset"].value_counts(dropna=False))

    print("\nDataset distribution in TEST:")
    print(test_df["dataset"].value_counts(dropna=False))
    
existing_split_path = '/Volumes/{hb-breast-nac-challenge}/work/dataset/train_test_splits.csv'
existing_split = pd.read_csv(existing_split_path)
existing_train_split = existing_split['train_split'].tolist()
existing_test_split = existing_split['test_split'].tolist()
existing_test_split = [x for x in existing_test_split if str(x) != 'nan']
# print('Existing train split:', existing_train_split)
print('number of existing train split:', len(existing_train_split))

# print('Existing test split:', existing_test_split)
print('number of existing test split:', len(existing_test_split))

# read the clinical information and check if the image information matches
clinical_info_path = '/Volumes/{hb-breast-nac-challenge}/work/dataset/clinical_and_imaging_info.xlsx'

clinical_info = pd.read_excel(clinical_info_path, sheet_name='dataset_info')

# Filter clinical info for train/test splits
train_df = clinical_info[clinical_info['patient_id'].isin(existing_train_split)]
test_df = clinical_info[clinical_info['patient_id'].isin(existing_test_split)]

print('Train DataFrame shape:', train_df.shape)
print('Test DataFrame shape:', test_df.shape)

# Further stratification: split the training data in to 4 folds
train_df_cv = train_df.copy()# only adding this to avoid SettingWithCopyWarning
train_df_cv["stratify_key"] = (
    train_df_cv["dataset"].astype(str) + "_" +
    train_df_cv["pcr"].astype(str) + "_" +
    train_df_cv["tumor_subtype"].astype(str)
)

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

X = train_df_cv.index  # or train_df['patient_id'] if you'd prefer
y = train_df_cv["stratify_key"]

Train_folds = [train_df]
Val_folds = [test_df]
# folds = []
for fold, (train_cv_idx, val_cv_idx) in enumerate(skf.split(X, y)):
    fold_train_df = train_df_cv.iloc[train_cv_idx].copy()
    fold_val_df = train_df_cv.iloc[val_cv_idx].copy()
    fold_train_df["fold"] = fold
    fold_val_df["fold"] = fold
    inspect_stratification(fold_train_df, fold_val_df)
    fold_train_df = fold_train_df.drop(columns=["stratify_key"])
    fold_val_df = fold_val_df.drop(columns=["stratify_key"])

    combined_train = pd.concat([fold_train_df, test_df], ignore_index=True)

    Train_folds.append(combined_train)
    Val_folds.append(fold_val_df)
    
print(len(Train_folds))
# Structure all folds including the existing train/test split

with pd.ExcelWriter("/Volumes/{hb-breast-nac-challenge}/work/min105/scripts/Data_preparation/cv_fold_splits.xlsx", engine="xlsxwriter") as writer:
    for i in range(len(Train_folds)):
        Train_folds[i] = Train_folds[i].sort_values(by="patient_id").reset_index(drop=True)
        Val_folds[i] = Val_folds[i].sort_values(by="patient_id").reset_index(drop=True)

        Train_folds[i].to_excel(writer, sheet_name=f"train_fold_{i}", index=False)
        Val_folds[i].to_excel(writer, sheet_name=f"val_fold_{i}", index=False)

    



