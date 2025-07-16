#!/bin/bash

#SBATCH --job-name="test mednext"
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1



export nnUNet_raw_data_base=root_path/MedNext_dataset_python310/nnUNet_raw_data_base
export nnUNet_preprocessed=root_path/MedNext_dataset_python310/nnUNet_preprocessed
export RESULTS_FOLDER=root_path/MedNext_dataset_python310/nnUNet_results_DC_and_topk_loss

cd [mednext path]

start=$(date +%s)


mednextv1_find_best_configuration -m 3d_fullres -t 012 -tr nnUNetTrainerV2_MedNeXt_L_kernel5_lr_5e_6 -ctr nnUNetTrainerV2_MedNeXt_L_kernel5_lr_5e_6 -pl nnUNetPlansv2.1_trgSp_1x1x1 -f 0 1 2 3 4


end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"


 




