#!/bin/bash

#SBATCH --job-name=entropy_and_fpos
#SBATCH --array=1-9
#SBATCH --output=outerr_ent/ent_%A_%a.out
#SBATCH --error=outerr_ent/ent_%A_%a.err
#SBATCH --time=1-12:00:00
#SBATCH --partition=dali
#SBATCH --ntasks=8
#SBATCH --mem=16G
#SBATCH --account=pi-sepalmer



module load julia/1.7.2
module load python/anaconda-2020.02



# Add lines here to run your computations.
julia calc_ents_and_fpos.jl $SLURM_ARRAY_TASK_ID
