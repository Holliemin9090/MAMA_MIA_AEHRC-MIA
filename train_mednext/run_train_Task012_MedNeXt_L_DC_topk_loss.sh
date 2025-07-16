#!/bin/bash


#SBATCH --job-name="train mednext_l"
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --mail-type=END,FAIL



# Print Python environment information
echo "Using Python at: $(which python)"
python --version

export nnUNet_raw_data_base=root_path/nnUNet_raw_data_base
export nnUNet_preprocessed=root_path/nnUNet_preprocessed
export RESULTS_FOLDER=root_path/nnUNet_results_DC_and_topk_loss


cd [mednext path]


INDEX="$(($SLURM_ARRAY_TASK_ID))"
echo $INDEX

# Start timing
start_time=$(date +%s)


# restart the training with smaller learning rate
pretrained_weights_path="root_path/nnUNet_results_DC_and_CE_loss/nnUNet/3d_fullres/Task012_breastDCE/nnUNetTrainerV2_MedNeXt_L_kernel5_lr_5e_4__nnUNetPlansv2.1_trgSp_1x1x1/fold_${INDEX}/model_final_checkpoint.model"

# # # train with kernel size 5 using upkernel
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel5_lr_5e_6 Task012_breastDCE $INDEX 300 DC_and_topk_loss -c -p nnUNetPlansv2.1_trgSp_1x1x1 --npz -pretrained_weights $pretrained_weights_path


# End timing
end_time=$(date +%s)
runtime=$((end_time - start_time))

# Print runtime in seconds and human-readable format
echo "Total execution time: ${runtime} seconds"
printf "Execution time: %02dh:%02dm:%02ds\n" $((runtime/3600)) $(( (runtime%3600)/60 )) $((runtime%60))