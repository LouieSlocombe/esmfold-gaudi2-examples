#!/bin/bash

#SBATCH --job-name=esmfold
#SBATCH -N 1
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH --time=0-01:00:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --export=NONE
#SBATCH --exclusive

cd $SLURM_SUBMIT_DIR

ENV_NAME="gaudi-pytorch-diffusion-1.22.0.740"

module load mamba/latest
source activate $ENV_NAME

export PT_HPU_LAZY_MODE=1

srun /packages/envs/$ENV_NAME/bin/python run_esmfold.py