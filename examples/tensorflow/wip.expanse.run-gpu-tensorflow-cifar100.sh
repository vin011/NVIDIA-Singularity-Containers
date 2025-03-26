#!/usr/bin/env bash
### Please adjust the following variables accordingly
#SBATCH --job-name=tensorflow-gpu
#SBATCH --account=sds196
#SBATCH --partition=gpu
#SBATCH --gpus=3
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=tensorflow-cifar100-gpu.o%j.%N

export SINGULARITY_MODULE=singularitypro/4.1.2
#export CONTAINER_DIR=$HOME/tensorflow_23.02-py3.sif
export CONTAINER_DIR=$HOME/NVIDIA-Singularity-Containers/sif_files/tensorflow_23.02-py3.sif
export GITHUB_REPO_DIR=$HOME/NVIDIA-Singularity-Containers

module reset
module load $SINGULARITY_MODULE

#### We will run from the scratch directory local to the node
cd /scratch/$USER/job_$SLURM_JOBID

## Print relevant environment and resource variables
module list 
printenv 
nvidia-smi 

export TF_USE_LEGACY_KERAS=1

time -p singularity exec --bind /expanse,/scratch --nv $CONTAINER_DIR "python3 $GITHUB_REPO_DIR/examples/tensorflow/tensorflow.cifar100.py --classes 10 --precision fp32 --epochs 1 --batch_size 256 --accelerator gpu --savekeras True"

#time -p singularity exec --bind /expanse,/scratch --nv $CONTAINER_DIR "python3 $GITHUB_REPO_DIR/examples/tensorflow/tensorflow.cifar100.py --data_dir /expanse/lustre/scratch/mkandes/temp_project/datasets/sdsc10.zip --classes 10 --precision fp32 --epochs 1 --batch_size 256 --accelerator gpu --savekeras True"
