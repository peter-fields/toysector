#!/bin/bash

#SBATCH --job-name=entropy_toy
#SBATCH --array=1-10
#SBATCH --output=outerr_ent/ent_%A_%a.out
#SBATCH --error=outerr_ent/ent_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=8
#SBATCH --mem=16G
#SBATCH --account=pi-sepalmer



module load julia/1.7.2
module load python/anaconda-2020.02


# Add lines here to run your computations.
julia toy_frac_func_entropy.jl $SLURM_ARRAY_TASK_ID
