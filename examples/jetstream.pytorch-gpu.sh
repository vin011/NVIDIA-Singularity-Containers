#!/usr/bin/env bash
export SINGULARITY_MODULE=singularity/1.3.4
export CONTAINER_DIR=$HOME/NVIDIA-Singularity-Containers/sif_files
export ENV_OUPUT_FILE_NAME=$SLURM_JOB_ID.environment.txt

module reset
module load $SINGUALRITY_MODULE 

## Print relevant environment and resource variables
module list >> $ENV_OUTPUT_FILE_NAME
printenv >> $ENV_OUTPUT_FILE_NAME
nvidia-smi >> $ENV_OUTPUT_FILE_NAME

# Load and run container
apptainer exec $CONTAINER_DIR/pytorch_24.11-py3.sif python -c 'import torch; print(f"CUDA: {torch.cuda.is_available()}")'

