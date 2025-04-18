#!/bin/bash
#!/datasets/work/hb-breast-nac-challenge/work/min105/myenv/bin python3

#SBATCH --account=OD-231108
#SBATCH --job-name="data inspection"
#SBATCH --time=22:00:00
#SBATCH --output=slurm_files/data_inspection_out.txt
#SBATCH --error=slurm_files/data_inspection_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G

# module load python/3.12.0

source /datasets/work/hb-breast-nac-challenge/work/min105/myenv/bin/activate

# Print current date
date

PYTHONPATH="/datasets/work/hb-breast-nac-challenge/work/min105/myenv/lib/python3.12/site-packages/":"${PYTHONPATH}"
export PYTHONPATH

cd /datasets/work/hb-breast-nac-challenge/work/min105/scripts/Data_preparation

start=$(date +%s)

srun -n1 python3.12 Data_analysis.py

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
