#!/usr/bin/env bash
export SINGULARITY_MODULE=singularity/1.1.2
export CONTAINER_DIR=$HOME/tensorflow_23.10-py3.sif
export GITHUB_REPO_DIR=$HOME/NVIDIA-Singularity-Containers

module reset
module load $SINGULARITY_MODULE

## Print relevant environment and resource variables
module list 
printenv 
nvidia-smi 

time -p singularity exec --nv $CONTAINER_DIR python3 $GITHUB_REPO_DIR/examples/tensorflow/tensorflow.cifar10.py --classes 100 --precision fp32 --epochs 1 --batch_size 256 --accelerator gpu --savekeras True
