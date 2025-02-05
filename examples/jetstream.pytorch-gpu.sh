#!/usr/bin/env bash
export SINGULARITY_MODULE=singularity/1.3.4
export GITHUB_REPO=$HOME/NVIDIA-Singularity-Containers
export ENV_OUPUT_FILE_NAME=$SLURM_JOB_ID.environment.txt

module reset
module load $SINGUALRITY_MODULE 

## Print relevant environment and resource variables
module list >> $ENV_OUTPUT_FILE_NAME
printenv >> $ENV_OUTPUT_FILE_NAME
nvidia-smi >> $ENV_OUTPUT_FILE_NAME

# Load and run container
apptainer exec pytorch_24.11-py3.sif python -c 'import torch; print(torch.cuda.is_available())'

