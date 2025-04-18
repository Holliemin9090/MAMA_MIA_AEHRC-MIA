#!/bin/bash

#SBATCH --account=OD-231108
#SBATCH --job-name="set variables and integrity check"
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm_files/DCE001_2d_integrity_out.txt
#SBATCH --error=slurm_files/DCE001_2d_integrity_error.txt

source /datasets/work/hb-breast-nac-challenge/work/min105/myenv/bin/activate
PYTHONPATH="/datasets/work/hb-breast-nac-challenge/work/min105/myenv/lib/python3.12/site-packages/":"${PYTHONPATH}"
export PYTHONPATH

export nnUNet_raw_data_base=/datasets/work/hb-breast-nac-challenge/work/MedNext_dataset/nnUNet_raw_data_base
export nnUNet_preprocessed=/datasets/work/hb-breast-nac-challenge/work/MedNext_dataset/nnUNet_preprocessed
export RESULTS_FOLDER=/datasets/work/hb-breast-nac-challenge/work/MedNext_dataset/nnUNet_results

cd /datasets/work/hb-breast-nac-challenge/work/min105/myenv/src/mednextv1

# mednextv1_plan_and_preprocess -t 1 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 

mednextv1_plan_and_preprocess -t 1 -pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1
