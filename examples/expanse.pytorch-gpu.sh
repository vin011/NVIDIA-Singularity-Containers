#!/usr/bin/env bash
### Please adjust the following variables accordingly
#SBATCH --job-name=pytorch-gpu
#SBATCH --account=XXXXX
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=pytorch-gpu.o%j.%N

export SINGULARITY_MODULE=singularitypro/3.11
export PYTHON_MODULE=python/3.8.12/7zdjza7
export GITHUB_REPO=$HOME/NVIDIA-Singularity-Containers
export ENV_OUPUT_FILE_NAME=$SLURM_JOB_ID.environment.txt

module reset
module load $SINGUALRITY_MODULE $PYTHON_MODULE

cd /scratch/$USER/job_$SLURM_JOBID

## Print relevant environment and resource variables
module list >> $ENV_OUTPUT_FILE_NAME
printenv >> $ENV_OUTPUT_FILE_NAME
nvidia-smi >> $ENV_OUTPUT_FILE_NAME

# Load and run container
tar -xzvf $GITHUB_REPO/pytorch_24.11-py3.sif.gz

singularity exec --bind /expanse,/scratch pytorch_24.11-py3.sif "python3 -c 'import torch; print(torch.cuda.is_available()')"

