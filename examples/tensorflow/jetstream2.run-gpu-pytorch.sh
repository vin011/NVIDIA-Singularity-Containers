#!/usr/bin/env bash
export SINGULARITY_MODULE=singularity/1.1.2
export CONTAINER_DIR=$HOME/tensorflow_23.10-py3.sif

module reset
module load $SINGULARITY_MODULE

## Print relevant environment and resource variables
module list 
printenv 
nvidia-smi 

time -p singularity exec --nv $CONTAINER_DIR python3 $GITHUB_REPO_DIR/examples/tensorflow/tensorflow.cifar.py
