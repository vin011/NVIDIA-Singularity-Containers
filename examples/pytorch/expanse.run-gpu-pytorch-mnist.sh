#!/usr/bin/env bash
### Please adjust the following variables accordingly
#SBATCH --job-name=pytorch-gpu
#SBATCH --account=XXXXXX
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=pytorch-gpu.o%j.%N

export SINGULARITY_MODULE=singularitypro/4.1.2
export CONTAINER_DIR=$HOME/pytorch_23.02-py3.sif
export GITHUB_REPO_DIR=$HOME/NVIDIA-Singularity-Containers

module reset
module load $SINGULARITY_MODULE

#### We will run from the scratch directory local to the node
cd /scratch/$USER/job_$SLURM_JOBID

## Print relevant environment and resource variables
module list 
printenv 
nvidia-smi

time -p singularity exec --bind /expanse,/scratch --nv $CONTAINER_DIR python3 $GITHUB_REPO_DIR/examples/pytorch/pytorch.mnist.py
