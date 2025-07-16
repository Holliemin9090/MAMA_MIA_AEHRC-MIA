# MAMA-MIA Challenge 2025: MICCAI Participation

This repository contains our code and documentation for the **MAMA-MIA Challenge 2025**, held as part of the MICCAI conference.

## 🧠 Challenge Overview

- **Organized by:** Universitat de Barcelona  
- **Official Website:** [https://www.ub.edu/mama-mia/](https://www.ub.edu/mama-mia/)  
- **Tasks:**  
  1. Primary tumor segmentation in DCE-MRI  
  2. Prediction of pathologic complete response (pCR) to neoadjuvant chemotherapy  
- **Data:** Multiparametric breast MRI from multiple centers  
- **Evaluation Metrics:** Dice (segmentation), AUROC (classification), and patient-wise pCR accuracy

---

## 🛠️ Our Approach

### 🩻 Task 1: Primary Tumor Segmentation with MedNeXt

We used the MedNeXt architecture to segment primary breast tumors from DCE-MRI scans. The following scripts were used to prepare the data:

#### 📁 Data Preparation for Training
- `Data_preparation/Segmentation/Data_preparation_1_channel_as_organizer_baseline.py`  
  Preprocesses single-channel subtraction images for segmentation training.

#### 📄 Dataset JSON Generation
- `Data_preparation/Segmentation/generate_json_1_channel.py`  
  Creates the required dataset description JSON file following nnU-Net conventions.

#### 🔀 Cross-Validation Split Generation
- `Data_preparation/Segmentation/generate_split_json_1_channel.py`  
  Generates a 5-fold cross-validation split file (`splits_final.pkl`) for training and evaluation.

---

_Additional details on model training, evaluation, and Task 2 (pCR prediction) will be added soon._




### 🧠 Models Used
- 3D and 2D **MedNeXt** models for segmentation based on slice thickness
- Ensembling across 5 folds
- Binary classification model using [insert classifier, e.g., ResNet/SwinViT/ConvNeXt] for pCR prediction

### 🔁 Training Strategy
- Cross-validation with 5 folds
- Polynomial LR scheduler and AdamW optimizer
- Custom augmentation pipeline via MONAI

### 🧪 Inference
- Separate pipelines for fine-slice and coarse-slice data
- Thresholding optimized per modality for final prediction

## 📊 Results

| Task | Metric | Score |
|------|--------|-------|
| Segmentation | Dice | [Insert] |
| pCR Prediction | AUROC | [Insert] |

> Final scores and ranks will be updated after the official leaderboard is released.

## 🤝 Acknowledgements

- This work was completed as part of the **MAMA-MIA MICCAI 2025 Challenge**
- Thanks to the organizers for providing the dataset and evaluation tools

## 📄 License

[Insert license, e.g., Apache 2.0, if you're sharing code]

