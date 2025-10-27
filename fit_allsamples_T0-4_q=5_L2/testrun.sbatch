#!/bin/bash

#SBATCH --job-name=fit_toysector_nsamples500
#SBATCH --array=1-1,
#SBATCH --output=outerr/fit_%A_%a.out
#SBATCH --error=outerr/fit_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --partition=sepalmer
#SBATCH --ntasks=4
#SBATCH --mem=8G



module load julia/1.7.2
module load cudnn/8.1.0+cuda-11.2
module load cuda/11.2
module load python/anaconda-2020.02



# Add lines here to run your computations.
JULIA_CUDA_USE_BINARYBUILDER=false julia fit_pw_rbm_srbm_to_1_dataset_testrun.jl $SLURM_ARRAY_TASK_ID

