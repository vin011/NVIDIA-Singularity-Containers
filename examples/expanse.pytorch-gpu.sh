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
export CONTAINER_DIR=$HOME/NVIDIA-Singularity-Containers/sif_files
export ENV_OUPUT_FILE_NAME=$SLURM_JOB_ID.environment.txt

module reset
module load $SINGUALRITY_MODULE

cd /scratch/$USER/job_$SLURM_JOBID

## Print relevant environment and resource variables
module list >> $ENV_OUTPUT_FILE_NAME
printenv >> $ENV_OUTPUT_FILE_NAME
nvidia-smi >> $ENV_OUTPUT_FILE_NAME

# Load and run container
singularity exec --bind /expanse,/scratch $CONTAINER_DIR/pytorch_24.11-py3.sif 'import torch; print(f"TORCH & CUDA: {torch.cuda.is_available()}")'
