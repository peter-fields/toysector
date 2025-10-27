#!/bin/bash

#SBATCH --job-name=fit_toysector_L2
#SBATCH --array=1-5
#SBATCH --output=outerr/fit_%A_%a.out
#SBATCH --error=outerr/fit_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=pi-sepalmer



module load julia/1.7.3
module load cudnn/11.2
module load cuda/11.2
module load python/anaconda-2020.11



# Add lines here to run your computations.
JULIA_CUDA_USE_BINARYBUILDER=false julia fit_pw_rbm_to_all_datasets_at_1temp_L2.jl $SLURM_ARRAY_TASK_ID

