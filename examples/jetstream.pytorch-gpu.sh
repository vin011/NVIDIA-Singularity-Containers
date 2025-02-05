#!/usr/bin/env bash
export SINGULARITY_MODULE=apptainer
export PYTHON_MODULE=python/3.8.12/7zdjza7
export GITHUB_REPO=$HOME/NVIDIA-Singularity-Containers
export ENV_OUPUT_FILE_NAME=$SLURM_JOB_ID.environment.txt

module reset
module load $SINGUALRITY_MODULE $PYTHON_MODULE

cd /scratch

## Print relevant environment and resource variables
module list >> $ENV_OUTPUT_FILE_NAME
printenv >> $ENV_OUTPUT_FILE_NAME
nvidia-smi >> $ENV_OUTPUT_FILE_NAME

# Load and run container
tar -xzvf $GITHUB_REPO/pytorch_24.11-py3.sif.gz

singularity exec --bind /scratch pytorch_24.11-py3.sif "python3 -c 'import torch; print(torch.cuda.is_available()')"

