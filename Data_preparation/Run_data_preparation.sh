#!/bin/bash
#SBATCH --account=OD-231108
#SBATCH --job-name="data preparation"
#SBATCH --time=22:00:00
#SBATCH --output=/datasets/work/hb-breast-nac-challenge/work/min105/slurm_files/data_preparation_out.txt
#SBATCH --error=/datasets/work/hb-breast-nac-challenge/work/min105/slurm_files/data_preparation_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G

source /datasets/work/hb-breast-nac-challenge/work/miniconda3/etc/profile.d/conda.sh
conda activate /datasets/work/hb-breast-nac-challenge/work/min105/myenv_python310

date

cd /datasets/work/hb-breast-nac-challenge/work/min105/scripts_python310/Data_preparation

start=$(date +%s)

srun -n1 python Data_preparation.py

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
