# Challenge data and description download
Done
# Initiate a python 3.12 env and installed mednext and its dependencies
Mednext (the version used in EPVS challenge) has been installed but to use the SwinunetR, more libraries may be needed.
After the mednext is installed, additional library like surface_distance (https://github.com/google-deepmind/surface-distance/tree/master) monai, timm, CoTr (https://github.com/YtongXie/CoTr) need to be installed to enable full funtionality. 
'MedNeXt-mod/nnunet_mednext/training/network_training/nnUNet_variants/architectural_variants/nnUNetTrainerV2_noDeepSupervision.py' need to add num_epoch and loss function arguments in it.
The above libs were installed sequentially, when installing timm, it did try to reinstall new torch and other libraries, but it appears that it has not problem running.

# I ran a Data_analysis to cross-check if the downloaded images matches the image and clinical infor spread sheet.
I noticed in ISPY2 cohort, there are a lot of cases where the image size (row and column) does not match the image size recorded in the spreadsheet. This is probably because the image size in the spread sheet is the full image before cropping them into unilateral breast.

# Apart from the image and segmentation folder, there is also a folder called patient_info_files
This folder has json files of axillary informantion. Acording to the organizer, information such as breast coordinates of the primary tumour, field strength, echo time will be available for all cases including holdout test sets.

# Data preparation for MedNeXt
I have extracted training images and labels into the required format (cropped with the breast coordinates of the primary tumour). Data.json file has been generated. Data intergrity check and preprocessing (3d) is being carried out.

The organizer has provided a one-fold train-test split. Additionaly I generated 4 folds, so there are 5 folds (stratified with cancer subtype, pcr and cohorts) for cross validation. This split information has been saved into cv-split spreadsheet.

# The organizer has release codebench and basic env docker (17 Apr)
codebench: https://www.codabench.org/competitions/7425/
base docker: lgarrucho/codabench-gpu:latest

Liz has inspected the docker env. They use Python3.10.12 and the requirement file to replicate their env is at /datasets/hb-breast-nac-challenge/work/submission_related/requirements.txt

I have been training the mednext and swinunetr on my previous python3.12 env.
But now I have installed miniconda in the bowen storage and made a python3.10 env.

source /datasets/work/hb-breast-nac-challenge/work/miniconda3/etc/profile.d/conda.sh
conda activate /datasets/work/hb-breast-nac-challenge/work/min105/myenv_python310

The organizer's env is to be replicated and mednext is to be installed.
# The organizer's env has been replicated
extra lib need to be installed
argparse-1.4.0
nnunet 1.7.0
monai (this will downgrade numpy)
CoTr
mednext-mod
natsort
surface-distance

