#!/bin/bash

#SBATCH --account=OD-231108
#SBATCH --job-name="train mednext_l"
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=slurm_files/train_mednext_l_%A_%a_DC_and_topk_loss.log

#module load cuda/12.1.0
#module load cudnn/8.9.5-cu12


source /datasets/work/hb-breast-nac-challenge/work/min105/myenv/bin/activate
PYTHONPATH="/datasets/work/hb-breast-nac-challenge/work/min105/myenv/lib/python3.12/site-packages/":"${PYTHONPATH}"
export PYTHONPATH

export nnUNet_raw_data_base=/datasets/work/hb-breast-nac-challenge/work/MedNext_dataset/nnUNet_raw_data_base
export nnUNet_preprocessed=/datasets/work/hb-breast-nac-challenge/work/MedNext_dataset/nnUNet_preprocessed
export RESULTS_FOLDER=/datasets/work/hb-breast-nac-challenge/work/MedNext_dataset/nnUNet_results_DC_and_topk_loss

cd /datasets/work/hb-breast-nac-challenge/work/min105/myenv/src/mednextv1

# export nnUNet_compile=F

INDEX="$(($SLURM_ARRAY_TASK_ID))"
echo $INDEX

## train with kernel size 3
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel3_lr_5e_4 Task001_breastDCE $INDEX 300 DC_and_topk_loss -p nnUNetPlansv2.1_trgSp_1x1x1 --npz

#pretrained_weights_path="/datasets/work/hb-neuro-feobv/work/EPVS_code/MedNeXt_env/dataset/nnUNet_results_no_sampling_DC_and_CE_loss/nnUNet/3d_fullres/Task001_T1wT2w/nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_noRes/fold_${INDEX}/model_final_checkpoint.model"

## train with kernel size 5 using upkernel
#mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel5 Task001_T1wT2w $INDEX 300 DC_and_CE_loss -p nnUNetPlansv2.1_noRes -pretrained_weights $pretrained_weights_path -resample_weights

