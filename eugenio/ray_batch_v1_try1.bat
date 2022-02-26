#!/bin/bash
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1-23:00:00
#SBATCH --account=bkrishna_437
module purge
module load gcc/8.3.0
module load python/3.6.8
python3 SA_experiment_RAY_no_randomness_v1_try1.py



