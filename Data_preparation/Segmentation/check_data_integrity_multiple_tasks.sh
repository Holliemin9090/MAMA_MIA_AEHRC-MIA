#!/bin/bash

#SBATCH --job-name="preprocess_tasks_array"
#SBATCH --time=4-00:00:00
#SBATCH --array=12


# Set nnU-Net paths
export nnUNet_raw_data_base=root_path/MedNext_dataset_python310/nnUNet_raw_data_base
export nnUNet_preprocessed=root_path/MedNext_dataset_python310/nnUNet_preprocessed
export RESULTS_FOLDER=root_path/MedNext_dataset_python310/nnUNet_results_DC_and_CE_loss

cd [mednext path]

# Record start time
start_time=$(date +%s)
echo "🚀 Job started at: $(date) | SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Run preprocessing for the current task index
mednextv1_plan_and_preprocess -t $SLURM_ARRAY_TASK_ID -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 
# -pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1

# Record end time
end_time=$(date +%s)
duration=$((end_time - start_time))

# Report timing
echo "✅ Job completed at: $(date)"
printf "⏱️ Total runtime: %02d:%02d:%02d (hh:mm:ss)\n" $((duration/3600)) $(((duration%3600)/60)) $((duration%60))
