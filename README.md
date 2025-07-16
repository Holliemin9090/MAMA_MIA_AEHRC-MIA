# MAMA-MIA Challenge 2025: MICCAI Participation

This repository contains our code and documentation for the **MAMA-MIA Challenge 2025**, held as part of the MICCAI conference.

## 🧠 Challenge Overview

- **Organized by:** Universitat de Barcelona  
- **Official Website:** [https://www.ub.edu/mama-mia/](https://www.ub.edu/mama-mia/)  
- **Tasks:**  
  1. Primary tumor segmentation in DCE-MRI  
  2. Prediction of pathologic complete response (pCR) to neoadjuvant chemotherapy  
- **Data:** Training data: A large-scale multicenter breast cancer DCE-MRI benchmark dataset with 1506 cases; Validation data:574 cases from three European centres.  
- **Evaluation Metrics:** Dice, normalized Hausdorff Distance and fairness score (segmentation), balanced accuracy and fairness score (classification).

---

## 🛠️ Our Approach

### 🩻 Task 1: Primary Tumor Segmentation with MedNeXt

We used the MedNeXt architecture to segment primary breast tumors from DCE-MRI scans.

#### 📁 Data Preparation for Training
- `Data_preparation/Segmentation/Data_preparation_1_channel_as_organizer_baseline.py`  
  Preprocesses post-contrast and subtraction images for segmentation training.

#### 📄 Dataset JSON Generation
- `Data_preparation/Segmentation/generate_json_1_channel.py`  
  Creates the required dataset description JSON file following nnU-Netv1/MedNeXt conventions.

#### 🔀 Cross-Validation Split Generation
- `Data_preparation/Segmentation/generate_split_json_1_channel.py`  
  Generates a 5-fold cross-validation split file (`splits_final.pkl`) for training and validation.

#### Model training
1. Train model with DC+CE loss 
- `train_mednext/run_train_Task012_MedNeXt_L_DC_CE_loss.sh`
2. Fine tune model with DC+topk CE loss
- `train_mednext/run_train_Task012_MedNeXt_L_DC_topk_loss.sh`
3. Determine ensemble and postprocessing
- `train_mednext/Determine_ensemble_MedNeXt_L_DC_topk_loss.sh`
---

### 🩻 Task 2: PCR prediction
1. Prepare data for radiomic feature extraction
- `Data_preparation/pcr_classification/Extract_ROI_for_radiomics.py`
2. Extract radiomic features (using PyRadiomics) and kinetic features
- `pcr_classification_wt_radiomics/Extract_radiomics.py`
3. Train a Random Forest classifier for pCR classification
- `pcr_classification_wt_radiomics/pcr_classification_radiomics.py`


### Submission inference
The inference scripts for running the trained networks are in `sample_code_submission`


## 📊 Results

> Final scores and ranks will be updated after the official leaderboard is released.

## 🤝 Acknowledgements

- This work was completed as part of the **MAMA-MIA MICCAI 2025 Challenge**
- Thanks to the organizers for providing the dataset and evaluation tools

## 📄 License

[Apache 2.0]

## Reference
@article{garrucho2025,
  title={A large-scale multicenter breast cancer DCE-MRI benchmark dataset with expert segmentations},
  author={Garrucho, Lidia and Kushibar, Kaisar and Reidel, Claire-Anne and Joshi, Smriti and Osuala, Richard and Tsirikoglou, Apostolia and Bobowicz, Maciej and Riego, Javier del and Catanese, Alessandro and Gwoździewicz, Katarzyna and Cosaka, Maria-Laura and Abo-Elhoda, Pasant M and Tantawy, Sara W and Sakrana, Shorouq S and Shawky-Abdelfatah, Norhan O and Salem, Amr Muhammad Abdo and Kozana, Androniki and Divjak, Eugen and Ivanac, Gordana and Nikiforaki, Katerina and Klontzas, Michail E and García-Dosdá, Rosa and Gulsun-Akpinar, Meltem and Lafcı, Oğuz and Mann, Ritse and Martín-Isla, Carlos and Prior, Fred and Marias, Kostas and Starmans, Martijn P A and Strand, Fredrik and Díaz, Oliver and Igual, Laura and Lekadir, Karim},
  journal = {Scientific Data},
  year = {2025},
  doi = {10.1038/s41597-025-04707-4},
  pages = {453},
  number = {1},
  volume = {12}
}


@inproceedings{roy2023mednext,
  title={Mednext: transformer-driven scaling of convnets for medical image segmentation},
  author={Roy, Saikat and Koehler, Gregor and Ulrich, Constantin and Baumgartner, Michael and Petersen, Jens and Isensee, Fabian and Jaeger, Paul F and Maier-Hein, Klaus H},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={405--415},
  year={2023},
  organization={Springer}
}


@article{van2017computational,
  title={Computational radiomics system to decode the radiographic phenotype},
  author={Van Griethuysen, Joost JM and Fedorov, Andriy and Parmar, Chintan and Hosny, Ahmed and Aucoin, Nicole and Narayan, Vivek and Beets-Tan, Regina GH and Fillion-Robin, Jean-Christophe and Pieper, Steve and Aerts, Hugo JWL},
  journal={Cancer research},
  volume={77},
  number={21},
  pages={e104--e107},
  year={2017},
  publisher={American Association for Cancer Research}
}
