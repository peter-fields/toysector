#!/bin/bash

#SBATCH --job-name=entropy_toy_temps
#SBATCH --array=1-10
#SBATCH --output=outerr_ent/entbest_%A_%a.out
#SBATCH --error=outerr_ent/entbest_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=pi-sepalmer



module load julia/1.7.3
module load cudnn/8.3.1


# Add lines here to run your computations.
julia best_toy_frac_func_entropy.jl $SLURM_ARRAY_TASK_ID
